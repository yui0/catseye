//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

/* cats_rnnt.h - unified RNN-T decoding and experimental native ASR pipeline.
 * Includes tokenization, beam search, ONNX bridge, and WAV inference.
 * ReazonSpeech real-model compatibility remains unverified.
 */
#ifndef CATS_RNNT_H
#define CATS_RNNT_H
#include "cats_speech.h"
#include "cats_onnx.h"

/* ========================================================================
 * Token ID decoding and byte BPE
 * ======================================================================== */

/* Cat's eye / ReazonSpeech token ID -> UTF-8 text. Dependency-free C11.
 * Supports sherpa-onnx 'token id' text files and its byte-BPE alphabet.
 * Not an encoder; runs only after an ASR decoder produced token ids.
 * MPL-2.0.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef CATS_GPU
#include "cats_gpu.h"   /* optional GPU GEMM; the CPU kernels remain the fallback */
#define CS_GPU_FORGET() cs_gpu_forget_weights()
#else
#define CS_GPU_FORGET() ((void)0)
#endif

typedef struct {char **text;size_t count;} cs_vocab;
static inline void cs_vocab_free(cs_vocab*v){
 if(!v)return;
 for(size_t i=0;i<v->count;i++)free(v->text[i]);
 free(v->text);memset(v,0,sizeof(*v));
}
static inline int cs_vocab_load(cs_vocab*v,const char*path){
 if(!v||!path)return -1;
 memset(v,0,sizeof(*v));FILE*f=fopen(path,"rb");if(!f)return -1;
 char line[8192];int rc=-1;
 while(fgets(line,sizeof(line),f)){
   size_t n=strlen(line);if(n&&!strchr(line,'\n')&&!feof(f))goto done;
   while(n&&(line[n-1]=='\r'||line[n-1]=='\n'||line[n-1]==' '||line[n-1]=='\t'))line[--n]=0;
   if(!n)continue;
   /* Format: <symbol><space><integer id>. A sole integer means space. */
   char*end=line+n;
   while(end>line&&!isspace((unsigned char)end[-1]))end--;
   char *idstr=end;
   char *piece=line;
   if(end==line)piece=" ";
   else{while(end>line&&isspace((unsigned char)end[-1]))end--;*end=0;}
   if(!*idstr)goto done;
   char*tail;long id=strtol(idstr,&tail,10);
   if(*tail||id<0||id>=65536||!piece[0])goto done;
   if((size_t)id>=v->count){
     size_t cap=(size_t)id+1;if(cap>SIZE_MAX/sizeof(char*))goto done;
     char**p=(char**)realloc(v->text,cap*sizeof(char*));if(!p)goto done;
     memset(p+v->count,0,(cap-v->count)*sizeof(char*));v->text=p;v->count=cap;
   }
   if(v->text[id])goto done;
   size_t len=strlen(piece);v->text[id]=(char*)malloc(len+1);if(!v->text[id])goto done;
   memcpy(v->text[id],piece,len+1);
 }
 if(ferror(f)||v->count==0)goto done;
 rc=0;
done:fclose(f);if(rc)cs_vocab_free(v);return rc;
}
/* Resolve reserved symbols such as <unk> from model vocabulary. */
static inline int cs_vocab_find(const cs_vocab*v,const char*symbol){
 if(!v||!symbol)return -1;
 for(size_t i=0;i<v->count;i++)if(v->text[i]&&!strcmp(v->text[i],symbol))return (int)i;
 return -1;
}
/* Exact sherpa-onnx v1.13.6 bbpe.cc glyph order: 0..30 and 127..255.
 * (byte 31 does not have an entry in the upstream table).
 * 32..126 map to ASCII directly. */
static const char cs_bbpe_0_30[] = "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĞğ";
static const char cs_bbpe_127_255[] =
"ĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĴĵĶķĸĹĺĻļĽľŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦ";
static inline int cs_utf8_next(const unsigned char**p,uint32_t*cp){
 const unsigned char*s=*p;
 if(!*s)return 0;
 uint32_t x;int n;
 if(s[0]<0x80){x=s[0];n=1;}
 else if(s[0]>=0xc2&&s[0]<0xe0){x=s[0]&31;n=2;}
 else if(s[0]>=0xe0&&s[0]<0xf0){x=s[0]&15;n=3;}
 else if(s[0]>=0xf0&&s[0]<=0xf4){x=s[0]&7;n=4;}
 else return -1;
 for(int i=1;i<n;i++){if(!s[i]||(s[i]&0xc0)!=0x80)return -1;x=(x<<6)|(s[i]&63);}
 if((n==2&&x<128)||(n==3&&x<2048)||(n==4&&x<65536)||
     (x>=0xd800&&x<=0xdfff)||x>0x10ffff)return -1;
 *p+=n;*cp=x;return n;
}
static inline int cs_bbpe_lookup_table(uint32_t cp,const char*table,int base){
 const unsigned char*p=(const unsigned char*)table;int pos=0;
 while(*p){uint32_t v;if(cs_utf8_next(&p,&v)<0)return -1;if(v==cp)return base+pos;pos++;}
 return -1;
}
static inline int cs_bbpe_byte(uint32_t cp){
 if(cp>=32&&cp<=126)return (int)cp;
 if(cp==0x2047)return 32; /* sherpa-onnx: ⁇ -> space */
 int v=cs_bbpe_lookup_table(cp,cs_bbpe_0_30,0);
 if(v>=0)return v;
 return cs_bbpe_lookup_table(cp,cs_bbpe_127_255,127);
}
static inline int cs_emit_bytes(char*out,size_t cap,size_t*used,
                                  const char*src,size_t size){
 if(size>=cap || *used>cap-size-1)return -1;
 memcpy(out+*used,src,size);*used+=size;out[*used]=0;return 0;
}
/* mode: 1=byte-BPE (ReazonSpeech); 0=SentencePiece-like.
 * Errors on invalid token IDs, output overflow or malformed UTF-8.
 * Blank/specials must be omitted by the decoding caller. */
static inline int cs_vocab_decode(const cs_vocab*v,const int*ids,size_t n,
                                  int mode,char*out,size_t cap){
 if(!v||(!ids&&n)||!out||!cap)return -1;
 size_t used=0;out[0]=0;
 for(size_t i=0;i<n;i++){
   if(ids[i]<0||(size_t)ids[i]>=v->count||!v->text[ids[i]])return -1;
   const char*text=v->text[ids[i]];
   const unsigned char *p=(const unsigned char*)text;
   while(*p){
     const unsigned char*start=p;uint32_t cp;int bytes=cs_utf8_next(&p,&cp);
     if(bytes<0)return -1;
     if(cp==0x2581){ /* ▁ */
       if(!mode){if(cs_emit_bytes(out,cap,&used," ",1))return -1;}
       else if(used&&out[used-1]!=' '&&isprint((unsigned char)out[used-1])){
         if(cs_emit_bytes(out,cap,&used," ",1))return -1;
       }
       continue;
     }
     if(mode){
       int byte=cs_bbpe_byte(cp);
       if(byte>=0){
         char ch=(char)byte;
         if(byte==0)return -1; /* text API, not a binary buffer */
         if(cs_emit_bytes(out,cap,&used,&ch,1))return -1;
       }else if(cp<128 && isprint((unsigned char)cp)){
         if(cs_emit_bytes(out,cap,&used,(const char*)start,(size_t)bytes))return -1;
       }else return -1;
     }else {
       if(cs_emit_bytes(out,cap,&used,(const char*)start,(size_t)bytes))return -1;
     }
   }
 }
 return (int)used;
}


/* ========================================================================
 * RNN-T modified beam search
 * ======================================================================== */

/* Dependency-free reference modified beam search for offline RNN-T.
 * Aligns with sherpa-onnx v1.13.6 offline decoder: one symbol/blank
 * expansion per encoder frame; identical hypotheses merged by log-add.
 * No LM, hotwords, blank penalty, or parallel streams. MPL-2.0.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

typedef struct {
  int *ids;
  size_t len;
  double score;
  uint64_t hash;
} cs_beam_hyp;

typedef struct {
  int parent, symbol; /* symbol == -1: blank, also unk */
  double score;
  uint64_t hash;
  size_t len;
} cs_beam_candidate;

static inline double cs_logadd(double x,double y){
  if(x<y){double t=x;x=y;y=t;}
  return !isfinite(y)?x:x+log1p(exp(y-x));
}
static inline uint64_t cs_hash_append(uint64_t h,int sym){
  uint32_t x=(uint32_t)sym;
  for(int k=0;k<4;k++){h^=(x>>(k*8))&255u;h*=UINT64_C(1099511628211);}
  return h;
}
static inline int cs_beam_same(const cs_beam_hyp*old,const cs_beam_candidate*c,
                                const cs_beam_hyp*parent){
  if(old->len!=c->len||old->hash!=c->hash)return 0;
  if(c->symbol<0)return old->len==parent->len &&
      (!parent->len || !memcmp(old->ids,parent->ids,parent->len*sizeof(int)));
  return old->len==parent->len+1 &&
      (!parent->len || !memcmp(old->ids,parent->ids,parent->len*sizeof(int))) &&
      old->ids[parent->len]==c->symbol;
}
/* Compare candidate histories before pruning, so blank/unk and colliding
   paths are merged correctly instead of losing probability mass. */
static inline int cs_beam_candidate_same(const cs_beam_candidate*a,
                const cs_beam_candidate*b,const cs_beam_hyp*parents){
  if(a->len!=b->len||a->hash!=b->hash)return 0;
  const cs_beam_hyp *x=&parents[a->parent],*y=&parents[b->parent];
  if(a->symbol<0&&b->symbol<0)return x->len==y->len &&
      (!x->len || !memcmp(x->ids,y->ids,x->len*sizeof(int)));
  if(a->symbol>=0&&b->symbol>=0)return a->symbol==b->symbol &&
      x->len==y->len &&
      (!x->len || !memcmp(x->ids,y->ids,x->len*sizeof(int)));
  const cs_beam_candidate *extend=a->symbol>=0?a:b;
  const cs_beam_hyp *ex=&parents[extend->parent];
  const cs_beam_hyp *st=&parents[a->symbol<0?a->parent:b->parent];
  return st->len==ex->len+1 &&
      (!ex->len || !memcmp(st->ids,ex->ids,ex->len*sizeof(int))) &&
      st->ids[ex->len]==extend->symbol;
}
static inline int cs_beam_hash_cmp(const void*a,const void*b){
  const cs_beam_candidate*x=a,*y=b;
  if(x->hash!=y->hash)return (x->hash>y->hash)-(x->hash<y->hash);
  return (x->len>y->len)-(x->len<y->len);
}
static inline int cs_beam_cmp(const void *a,const void*b){
  const cs_beam_candidate*x=a,*y=b;
  return (x->score<y->score)-(x->score>y->score);
}
static inline int cs_beam_hyp_cmp(const void*a,const void*b){
  const cs_beam_hyp*x=a,*y=b;
  return (x->score<y->score)-(x->score>y->score);
}
/* Returns number of generated token ids or -1 on failure.
 * The initial tokens are -1,...,-1,0 (sherpa-onnx convention).
 * `unk_id` is treated like blank when it is >=0.
 * Work grows as O(frames * beam * vocab * beam) plus a sort;
 * beam <= 16, vocab <= 65536.  Intended for correctness first.
 */
static inline int cs_rnnt_modified_beam(const cs_rnnt *m,const float *encoded,
                size_t frames,int encoder_dim,int beam_size,int unk_id,
                int *out,size_t out_cap){
  if(!m||!m->predictor||!m->joiner||(!encoded&&frames)||!out||
      beam_size<1||beam_size>16||m->vocabulary<2||m->vocabulary>65536||
      m->blank<0||m->blank>=m->vocabulary||m->context_size<1||
      m->context_size>32||m->predictor_dim<1||m->predictor_dim>65536||
      encoder_dim<1||out_cap>INT32_MAX||
      (size_t)beam_size>(SIZE_MAX/(size_t)m->vocabulary))return -1;
  size_t nc=(size_t)beam_size*m->vocabulary;
  if(nc>1048576u)return -1;
  cs_beam_hyp old[16]={{0}},next[16]={{0}};
  cs_beam_candidate *cand=(cs_beam_candidate*)malloc(nc*sizeof(*cand));
  float *prediction=(float*)malloc((size_t)m->predictor_dim*sizeof(float));
  float *logits=(float*)malloc((size_t)m->vocabulary*sizeof(float));
  if(!cand||!prediction||!logits){free(cand);free(prediction);free(logits);return -1;}
  int count=1,answer=-1;
  old[0].score=0;old[0].hash=UINT64_C(1469598103934665603);
  for(size_t t=0;t<frames;t++){
    size_t used=0;
    for(int h=0;h<count;h++){
      int ctx[32];
      for(int i=0;i<m->context_size;i++)ctx[i]=i==m->context_size-1?0:-1;
      /* history is initial [-1 ... 0], then generated tokens */
      size_t skip=old[h].len>(size_t)m->context_size?
                    old[h].len-(size_t)m->context_size:0;
      for(size_t i=skip;i<old[h].len;i++){
        memmove(ctx,ctx+1,(size_t)(m->context_size-1)*sizeof(*ctx));
        ctx[m->context_size-1]=old[h].ids[i];
      }
      if(m->predictor(m->user,ctx,m->context_size,prediction)||
         m->joiner(m->user,encoded+t*(size_t)encoder_dim,prediction,logits))goto done;
      float max=-INFINITY;
      for(int i=0;i<m->vocabulary;i++)if(logits[i]>max)max=logits[i];
      if(!isfinite(max))goto done;
      double z=0;for(int i=0;i<m->vocabulary;i++)z+=exp((double)logits[i]-max);
      if(!(z>0))goto done;
      double norm=max+log(z);
      for(int i=0;i<m->vocabulary;i++){
        int sym=(i==m->blank||i==unk_id)?-1:i;
        if(sym>=0&&old[h].len>=out_cap)continue;
        cs_beam_candidate c={h,sym,old[h].score+logits[i]-norm,
          sym<0?old[h].hash:cs_hash_append(old[h].hash,sym),
          old[h].len+(sym>=0)};
        cand[used++]=c;
      }
    }
    /* Log-add duplicate output histories before top-k pruning. */
    qsort(cand,used,sizeof(*cand),cs_beam_hash_cmp);
    size_t unique=0;
    for(size_t k=0;k<used;k++){
      size_t at=unique;
      while(at>0 && cand[at-1].hash==cand[k].hash &&
            cand[at-1].len==cand[k].len)at--;
      size_t match=unique;
      for(size_t r=at;r<unique;r++)
        if(cs_beam_candidate_same(&cand[r],&cand[k],old)){match=r;break;}
      if(match!=unique)cand[match].score=cs_logadd(cand[match].score,cand[k].score);
      else {if(unique!=k)cand[unique]=cand[k];unique++;}
    }
    qsort(cand,unique,sizeof(*cand),cs_beam_cmp);
    int nnext=0;
    for(size_t k=0;k<unique&&nnext<beam_size;k++){
      const cs_beam_candidate*c=&cand[k];
      cs_beam_hyp *n=&next[nnext++];
      n->len=c->len;n->score=c->score;n->hash=c->hash;
      n->ids=(int*)malloc((n->len?n->len:1)*sizeof(int));
      if(!n->ids)goto done;
      const cs_beam_hyp*p=&old[c->parent];
      if(p->len)memcpy(n->ids,p->ids,p->len*sizeof(int));
      if(c->symbol>=0)n->ids[p->len]=c->symbol;
    }
    if(!nnext)goto done;
    qsort(next,nnext,sizeof(*next),cs_beam_hyp_cmp);
    for(int h=0;h<count;h++){free(old[h].ids);old[h].ids=NULL;}
    for(int h=0;h<nnext;h++){old[h]=next[h];next[h].ids=NULL;}
    count=nnext;
  }
  /* sherpa-onnx GetMostProbable(true): final length normalization by
     (context_size + emitted symbols), but intermediate top-k is raw score. */
  int best=0;
  for(int h=1;h<count;h++)
    if(old[h].score/(m->context_size+old[h].len) >
       old[best].score/(m->context_size+old[best].len))best=h;
  if(old[best].len>out_cap)goto done;
  if(old[best].len)memcpy(out,old[best].ids,old[best].len*sizeof(int));
  answer=(int)old[best].len;
done:
  for(int h=0;h<16;h++){free(old[h].ids);free(next[h].ids);}
  free(cand);free(prediction);free(logits);
  return answer;
}


/* ========================================================================
 * RNN-T encoder/decoder/joiner ONNX bridge
 * ======================================================================== */

/* Bridge Cat's eye's reference RNN-T greedy decoder to native ONNX sessions.
 * Decoder context: [1,context_size] int64; output flattened predictor state.
 * Joiner: [1,1,E] + [1,1,D], flattened logits [vocabulary].
 * The real Zipformer has model-specific shape/state details, which MUST be
 * verified against the exported encoder/decoder/joiner before use.
 * This is a reference integration, not a validated ReazonSpeech recognizer.
 * MPL-2.0.
 */

typedef struct {
  cx_session *decoder, *joiner;
  const char *decoder_in, *decoder_out;
  const char *joiner_encoder_in, *joiner_decoder_in, *joiner_out;
  int enc_dim, pred_dim, vocab;
  /* A recurrent streaming encoder requires its own graph and state outside
     this adapter. The adapter accepts already encoded acoustic frames. */
} cx_rnnt_adapter;

static inline int cx_rnnt_feed_vector(cx_session *s,const char *name,
                                       const float *x,int n,int type,int default_nd){
  if(!s||!name||!x||n<1)return -1;
  int k=cx_session_input_index(&s->graph,name);if(k<0)return -1;
  const cx_value_meta *m=&s->graph.input_meta[k];
  int nd=m->nd>=1?m->nd:default_nd;
  if(nd<1||nd>CX_DIMS)return -1;
  int64_t d[CX_DIMS];
  for(int i=0;i<nd;i++){
    d[i]=i==nd-1?n:1;
    if(m->nd>=1&&m->d[i]>=0&&m->d[i]!=d[i])return -1;
  }
  return cx_session_feed(s,name,d,nd,type,x,(uint64_t)n);
}
static inline int cx_rnnt_predict(void*opaque,const int*ctx,int context_size,float*out){
  cx_rnnt_adapter*a=opaque;
  if(!a||!a->decoder||!ctx||!out||context_size<1||context_size>32)return -1;
  float tokens[32];for(int i=0;i<context_size;i++){
    if(ctx[i]<-1||ctx[i]>16777216)return -1;
    tokens[i]=(float)ctx[i];
  }
  if(cx_rnnt_feed_vector(a->decoder,a->decoder_in,tokens,context_size,7,2))return -1;
  if(cx_session_run(a->decoder))return -1;
  const cx_tensor*t=cx_session_output(a->decoder,a->decoder_out);
  if(!t||t->n!=(uint64_t)a->pred_dim)return -1;
  memcpy(out,t->v,(size_t)a->pred_dim*sizeof(float));
  return 0;
}
static inline int cx_rnnt_join(void*opaque,const float*frame,const float*prediction,float*logits){
  cx_rnnt_adapter*a=opaque;
  if(!a||!a->joiner||!frame||!prediction||!logits)return -1;
  if(cx_rnnt_feed_vector(a->joiner,a->joiner_encoder_in,frame,a->enc_dim,1,3)||
     cx_rnnt_feed_vector(a->joiner,a->joiner_decoder_in,prediction,a->pred_dim,1,3))return -1;
  if(cx_session_run(a->joiner))return -1;
  const cx_tensor*t=cx_session_output(a->joiner,a->joiner_out);
  if(!t||t->n!=(uint64_t)a->vocab)return -1;
  memcpy(logits,t->v,(size_t)a->vocab*sizeof(float));
  return 0;
}
static inline int cx_rnnt_decode(cx_rnnt_adapter*adapter,const float*encoded,
                                  size_t frames,int context_size,int blank,int max_symbols,
                                  int*output,size_t output_capacity){
  if(!adapter||!adapter->decoder||!adapter->joiner||!encoded||!output)return -1;
  cs_rnnt m={0};
  m.vocabulary=adapter->vocab;m.blank=blank;
  m.context_size=context_size;m.predictor_dim=adapter->pred_dim;
  m.max_symbols_per_frame=max_symbols;m.user=adapter;
  m.predictor=cx_rnnt_predict;m.joiner=cx_rnnt_join;
  return cs_rnnt_greedy(&m,encoded,frames,adapter->enc_dim,blank,output,output_capacity);
}
/* sherpa-onnx offline ReazonSpeech compatible one-symbol-per-frame search.
   Blank=0; initial decoder context contains -1 padding + final 0. */
static inline int cx_rnnt_decode_beam(cx_rnnt_adapter*a,const float*enc,
                    size_t frames,int context_size,int blank,int beam_size,
                    int unk_id,int*out,size_t capacity){
  if(!a||!a->decoder||!a->joiner||!out)return -1;
  cs_rnnt m={0};m.vocabulary=a->vocab;m.blank=blank;
  m.context_size=context_size;m.predictor_dim=a->pred_dim;
  m.user=a;m.predictor=cx_rnnt_predict;m.joiner=cx_rnnt_join;
  return cs_rnnt_modified_beam(&m,enc,frames,a->enc_dim,beam_size,unk_id,out,capacity);
}
/* Encoded acoustic frames -> token IDs -> UTF-8, using native ONNX
   decoder/joiner and the Hayamimi symbol-table parser. The acoustic
   encoder itself must already have produced `enc` (not integrated yet). */
static inline int cx_rnnt_decode_text(cx_rnnt_adapter*a,const float*enc,
              size_t frames,int context_size,int blank,int beam,int unk,
              const cs_vocab*vocab,int bbpe,char*text,size_t cap){
  if(!vocab||!text||cap<1||!enc)return -1;
  /* 4096 emitted tokens per utterance is a deliberate bound. */
  int *ids=(int*)malloc(4096*sizeof(int));if(!ids)return -1;
  int n=cx_rnnt_decode_beam(a,enc,frames,context_size,blank,beam,unk,ids,4096);
  int written=n>=0?cs_vocab_decode(vocab,ids,(size_t)n,bbpe,text,cap):-1;
  free(ids);return written;
}


/* ========================================================================
 * Native-C ASR audio processing pipeline
 * ======================================================================== */

/* Cat's eye native offline RNN-T audio pipeline (experimental).
 * Standard C + libm; no Python, sherpa-onnx or ONNX Runtime linkage.
 * Uses the PARTIAL ONNX interpreter. Only synthetic-model tests pass so far;
 * real Zipformer/ReazonSpeech compatibility is NOT established.
 * MPL-2.0, 2026.
 */

typedef struct {
  cx_session encoder, decoder, joiner;
  cs_vocab vocab;
  cx_rnnt_adapter adapter;
  int context, beam, unk;
  char error[256];
} cx_native_asr;

static inline void cx_native_close(cx_native_asr *a) {
  if(!a)return;
  cs_vocab_free(&a->vocab);
  cx_session_close(&a->joiner);
  cx_session_close(&a->decoder);
  cx_session_close(&a->encoder);
  memset(&a->adapter,0,sizeof(a->adapter));
}
static inline int cx_native_last_dim(const cx_value_meta *m) {
  return m && m->nd>0 && m->d[m->nd-1]>0 && m->d[m->nd-1]<INT_MAX ? (int)m->d[m->nd-1] : -1;
}
static inline int cx_native_supported(cx_native_asr *a,FILE*report){
  cx_graph *g[3]={&a->encoder.graph,&a->decoder.graph,&a->joiner.graph};
  const char *what[3]={"encoder","decoder","joiner"};
  int ok=1;
  for(int j=0;j<3;j++){
    if(report)fprintf(report,"%s: %zu nodes\n",what[j],g[j]->nn);
    if(cx_graph_preflight(g[j],report))ok=0;
    if(report)cx_graph_interfaces(g[j],report);
  }
  return ok?0:-1;
}
/* Pure-C model bundle opening. Real model shapes must be inspected and compared
   with sherpa-onnx. graph interfaces are taken in their declared order. */
static inline int cx_native_open(cx_native_asr*a,const char*encoder,
    const char*decoder,const char*joiner,const char*tokens,
    int context_override,int beam) {
  if(!a||!encoder||!decoder||!joiner||!tokens||beam<1||beam>16)return -1;
  memset(a,0,sizeof(*a));
  if(cx_session_open(&a->encoder,encoder)){snprintf(a->error,sizeof(a->error),"cannot open encoder");return -1;}
  if(cx_session_open(&a->decoder,decoder)){snprintf(a->error,sizeof(a->error),"cannot open decoder");return -1;}
  if(cx_session_open(&a->joiner,joiner)){snprintf(a->error,sizeof(a->error),"cannot open joiner");return -1;}
  if(cs_vocab_load(&a->vocab,tokens)){snprintf(a->error,sizeof(a->error),"cannot read tokens.txt");return -1;}
  cx_graph*e=&a->encoder.graph,*d=&a->decoder.graph,*j=&a->joiner.graph;
  if(e->ninput!=2||e->noutput!=2||d->ninput!=1||d->noutput!=1||j->ninput!=2||j->noutput!=1) {
    snprintf(a->error,sizeof(a->error),"unexpected model input/output count");return -1;
  }
  if(e->input_meta[0].type && e->input_meta[0].type!=1){snprintf(a->error,sizeof(a->error),"encoder features must be FP32");return -1;}
  if(e->input_meta[1].type!=6 && e->input_meta[1].type!=7){snprintf(a->error,sizeof(a->error),"encoder length must be int32/int64");return -1;}
  if(d->input_meta[0].type!=6 && d->input_meta[0].type!=7){snprintf(a->error,sizeof(a->error),"decoder tokens must be int32/int64");return -1;}
  /* The real Sherpa exporter records context_size/vocab_size in
     decoder ModelProto.metadata_props. Its context axis may be dynamic. */
  int meta_ctx=0,meta_vocab=0;
  int have_ctx=cx_graph_metadata_int(d,"context_size",&meta_ctx);
  int have_vocab=cx_graph_metadata_int(d,"vocab_size",&meta_vocab);
  if(have_ctx<0||have_vocab<0){
    snprintf(a->error,sizeof(a->error),"invalid decoder metadata integer");return -1;
  }
  int inferred_ctx=cx_native_last_dim(&d->input_meta[0]);
  if(have_ctx>0 && inferred_ctx>0 && meta_ctx!=inferred_ctx){
    snprintf(a->error,sizeof(a->error),"decoder context shape/metadata mismatch");return -1;
  }
  if(context_override>0 && have_ctx>0 && context_override!=meta_ctx){
    snprintf(a->error,sizeof(a->error),"-C conflicts with decoder metadata context_size");return -1;
  }
  a->context=context_override>0?context_override:(have_ctx>0?meta_ctx:inferred_ctx);
  if(a->context<1||a->context>32 || (context_override>0 && inferred_ctx>0 && context_override!=inferred_ctx)){
    snprintf(a->error,sizeof(a->error),"cannot resolve decoder context length (use -C)");return -1;
  }
  int vocab=cx_native_last_dim(&j->output_meta[0]);
  if(vocab<1){if(a->vocab.count>INT_MAX)return -1;vocab=(int)a->vocab.count;}
  if(have_vocab>0 && (meta_vocab!=vocab || a->vocab.count<(size_t)meta_vocab)){
    snprintf(a->error,sizeof(a->error),"decoder vocab_size disagrees with joiner/tokens.txt");return -1;
  }
  if(vocab<2||vocab>65536 || a->vocab.count<(size_t)vocab){
    snprintf(a->error,sizeof(a->error),"joiner vocabulary differs from tokens.txt");return -1;
  }
  /* Decoder and joiner are called repeatedly for every beam/frame.
     Re-reading and expanding weights from disk on each call was prohibitive. */
  a->decoder.cache_weights=1;
  a->joiner.cache_weights=1;
  a->adapter.decoder=&a->decoder;a->adapter.joiner=&a->joiner;
  a->adapter.decoder_in=d->input[0];a->adapter.decoder_out=d->output[0];
  a->adapter.joiner_encoder_in=j->input[0];a->adapter.joiner_decoder_in=j->input[1];
  a->adapter.joiner_out=j->output[0];a->adapter.vocab=vocab;
  a->adapter.enc_dim=cx_native_last_dim(&e->output_meta[0]);
  a->adapter.pred_dim=cx_native_last_dim(&d->output_meta[0]);
  if(a->adapter.enc_dim<1||a->adapter.pred_dim<1){
    snprintf(a->error,sizeof(a->error),"encoder/decoder output last dim not static");return -1;
  }
  a->beam=beam;
  a->unk=cs_vocab_find(&a->vocab,"<unk>");
  return 0;
}
/* Feature frames -> encoder ONNX -> RNN-T decoder/joiner -> UTF-8.
   Returns number of UTF-8 bytes or -1; text remains valid on success. */
static inline int cx_native_transcribe_features(cx_native_asr*a,
    const cs_features*f,char*text,size_t capacity) {
  if(!a||!f||!f->data||f->dim!=80||!f->frames||f->frames>INT32_MAX||!text||!capacity)return -1;
  cx_graph*g=&a->encoder.graph;
  if(g->input_meta[0].nd>=0 && g->input_meta[0].nd!=3)return -1;
  int64_t feat_dims[]={1,(int64_t)f->frames,80}, len_dims[]={1};
  float length=(float)f->frames;
  if(cx_session_feed(&a->encoder,g->input[0],feat_dims,3,1,f->data,f->frames*80)||
     cx_session_feed(&a->encoder,g->input[1],len_dims,1,g->input_meta[1].type,&length,1))return -1;
  if(cx_session_run(&a->encoder))return -1;
  const cx_tensor*enc=cx_session_output(&a->encoder,g->output[0]);
  const cx_tensor*enc_len=cx_session_output(&a->encoder,g->output[1]);
  if(!enc||!enc_len||enc_len->n!=1||enc->nd!=3||enc->d[0]!=1||
     enc->d[2]!=a->adapter.enc_dim||enc->d[1]<0||
     (enc_len->type!=6&&enc_len->type!=7))return -1;
  int64_t frames=0;
  if(cx_i64(enc_len->v[0],&frames)||frames<0||frames>enc->d[1])return -1;
  if(frames==0){text[0]=0;return 0;}
  return cx_rnnt_decode_text(&a->adapter,enc->v,(size_t)frames,
            a->context,0,a->beam,a->unk,&a->vocab,1,text,capacity);
}
static inline int cx_native_transcribe_wav(cx_native_asr*a,
    const char*wav,char*text,size_t capacity) {
  cs_wave wave={0};cs_features f={0};
  if(!a||!wav||cs_wave_read(wav,&wave))return -1;
  int rc=cs_fbank80(wave.pcm,wave.count,wave.rate,&f);
  cs_wave_free(&wave);
  if(rc)return -1;
  rc=cx_native_transcribe_features(a,&f,text,capacity);
  cs_fbank_free(&f);
  return rc;
}
/* WAV -> energy-based VAD segments with <=1 second REAL audio preroll.
   Unlike original Hayamimi this does not use Silero ONNX VAD.  Text chunks
   are concatenated with newlines, each ASR call has independent RNN-T state.
   Out is empty for detected silence. Explicitly opt in via -V. */
static inline int cx_native_transcribe_wav_segmented(cx_native_asr *a,
    const char *wav,char *text,size_t capacity,size_t *segments){
  if(!a||!wav||!text||!capacity)return -1;
  text[0]=0;
  if(segments)*segments=0;
  cs_wave wav_data={0};cs_segment *parts=NULL;size_t count=0;
  if(cs_wave_read(wav,&wav_data))return -1;
  if(cs_vad_split(wav_data.pcm,wav_data.count,wav_data.rate,&parts,&count)){
    cs_wave_free(&wav_data);return -1;
  }
  size_t written=0;
  int rc=0;
  for(size_t i=0;i<count;i++){
    const cs_segment *seg=&parts[i];
    cs_features f={0};
    if(seg->end<=seg->begin || seg->end>wav_data.count ||
       cs_fbank80(wav_data.pcm+seg->begin,seg->end-seg->begin,wav_data.rate,&f)){
      rc=-1;break;
    }
    if(i>0){
      if(written+2>capacity){cs_fbank_free(&f);rc=-1;break;}
      text[written++]='\n';text[written]=0;
    }
    rc=cx_native_transcribe_features(a,&f,text+written,capacity-written);
    cs_fbank_free(&f);
    if(rc<0)break;
    written+=(size_t)rc;
    if(segments)(*segments)++;
  }
  free(parts);cs_wave_free(&wav_data);
  return rc<0?-1:(int)written;
}


/* ========================================================================
 * EmbeddingGemma word-space nearest neighbor, optional native INT8 table.
 * The learned reader is an external model: projected_tokens [1,750,768].
 * The fixed word table is normalized text-embedding prototypes.
 * Quantization is max-abs per row, and cosine is computed from INT8 dot
 * products, so no additional float scale is needed for ranking.
 * If the table contains 200k words, q8 storage is ~154 MB.
 * ======================================================================== */
#define CS_EG2_TEXT_DIM 768
typedef struct {
  int8_t*vec;
  float*inverse_norm;
  size_t count;
} cs_eg2_q8;
static inline void cs_eg2_q8_free(cs_eg2_q8*t){
  if(!t)return;
  free(t->vec);free(t->inverse_norm);memset(t,0,sizeof(*t));
}
static inline int cs_eg2_q8_open(cs_eg2_q8*t,const char*path){
  if(!t||!path)return -1;
  memset(t,0,sizeof(*t));FILE*f=cs_eg2_file_open(path);if(!f)return -1;
  unsigned char h[16];int rc=-1;
  if(fread(h,1,16,f)!=16||memcmp(h,"EG2Q",4)||cs_u32(h+4)!=1||
     !cs_u32(h+8)||cs_u32(h+8)>1000000u||cs_u32(h+12)!=768)goto done;
  t->count=cs_u32(h+8);size_t elems=t->count*CS_EG2_TEXT_DIM;
  if(elems>SIZE_MAX/sizeof(int8_t)||t->count>SIZE_MAX/sizeof(float))goto done;
  t->vec=(int8_t*)malloc(elems);t->inverse_norm=(float*)malloc(t->count*sizeof(float));
  if(!t->vec||!t->inverse_norm||fread(t->vec,1,elems,f)!=elems||fgetc(f)!=EOF||ferror(f))goto done;
  for(size_t k=0;k<t->count;k++){
    const int8_t*row=t->vec+k*CS_EG2_TEXT_DIM;
    double dot=0;
    for(size_t j=0;j<CS_EG2_TEXT_DIM;j++)dot+=(double)row[j]*row[j];
    if(dot<=0)goto done;
    t->inverse_norm[k]=(float)(1./sqrt(dot));
  }
  rc=0;
done:fclose(f);if(rc)cs_eg2_q8_free(t);return rc;
}
static inline int cs_eg2_dot_scalar(const int8_t*a,const int8_t*b){
  int sum=0;for(size_t k=0;k<CS_EG2_TEXT_DIM;k++)sum+=(int)a[k]*(int)b[k];
  return sum;
}
#if CATS_SIMD_X86
CATS_AVX2_FN static int cs_eg2_dot_avx2(const int8_t*a,const int8_t*b){
  __m256i acc=_mm256_setzero_si256();
  for(size_t k=0;k<CS_EG2_TEXT_DIM;k+=32){
    __m128i a0=_mm_loadu_si128((const __m128i*)(a+k));
    __m128i a1=_mm_loadu_si128((const __m128i*)(a+k+16));
    __m128i b0=_mm_loadu_si128((const __m128i*)(b+k));
    __m128i b1=_mm_loadu_si128((const __m128i*)(b+k+16));
    __m256i p0=_mm256_madd_epi16(_mm256_cvtepi8_epi16(a0),_mm256_cvtepi8_epi16(b0));
    __m256i p1=_mm256_madd_epi16(_mm256_cvtepi8_epi16(a1),_mm256_cvtepi8_epi16(b1));
    acc=_mm256_add_epi32(acc,_mm256_add_epi32(p0,p1));
  }
  int32_t sum[8];_mm256_storeu_si256((__m256i*)sum,acc);
  int total=0;for(int j=0;j<8;j++)total+=sum[j];return total;
}
/* AVX-only CPUs (no AVX2): same exact integer sum with 128-bit lanes. */
CATS_AVX_FN static int cs_eg2_dot_avx1(const int8_t*a,const int8_t*b){
  __m128i acc=_mm_setzero_si128();
  for(size_t k=0;k<CS_EG2_TEXT_DIM;k+=16){
    __m128i x=_mm_loadu_si128((const __m128i*)(a+k));
    __m128i y=_mm_loadu_si128((const __m128i*)(b+k));
    __m128i lo=_mm_madd_epi16(_mm_cvtepi8_epi16(x),_mm_cvtepi8_epi16(y));
    __m128i hi=_mm_madd_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(x,8)),
                              _mm_cvtepi8_epi16(_mm_srli_si128(y,8)));
    acc=_mm_add_epi32(acc,_mm_add_epi32(lo,hi));
  }
  int32_t sum[4];_mm_storeu_si128((__m128i*)sum,acc);
  return sum[0]+sum[1]+sum[2]+sum[3];
}
/* Four queries (already widened to int16) against one table row: the row is
 * widened once and reused, which is what limits the single-query version. */
CATS_AVX_FN static void cs_eg2_dot4_avx1(const int16_t*x,const int8_t*row,int out[4]){
  __m128i a0=_mm_setzero_si128(),a1=a0,a2=a0,a3=a0;
  for(size_t k=0;k<CS_EG2_TEXT_DIM;k+=8){
    __m128i w=_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(row+k)));
    a0=_mm_add_epi32(a0,_mm_madd_epi16(w,_mm_loadu_si128((const __m128i*)(x+k))));
    a1=_mm_add_epi32(a1,_mm_madd_epi16(w,_mm_loadu_si128((const __m128i*)(x+CS_EG2_TEXT_DIM+k))));
    a2=_mm_add_epi32(a2,_mm_madd_epi16(w,_mm_loadu_si128((const __m128i*)(x+2*CS_EG2_TEXT_DIM+k))));
    a3=_mm_add_epi32(a3,_mm_madd_epi16(w,_mm_loadu_si128((const __m128i*)(x+3*CS_EG2_TEXT_DIM+k))));
  }
  __m128i s=_mm_hadd_epi32(_mm_hadd_epi32(a0,a1),_mm_hadd_epi32(a2,a3));
  _mm_storeu_si128((__m128i*)out,s);
}
#endif
static inline int cs_eg2_dot_q8(const int8_t*a,const int8_t*b){
#if CATS_SIMD_X86
  {int level=cats_cpu_simd_level();
   if(level>=2)return cs_eg2_dot_avx2(a,b);
   if(level==1)return cs_eg2_dot_avx1(a,b);}
#endif
  return cs_eg2_dot_scalar(a,b);
}
/* Projected 768-d float vector. Query is quantized independently, but
 * L2 cosine compensates for differing scales. Learned logit_scale must be
 * supplied from the reader's training checkpoint, not guessed. */
static inline int cs_eg2_nearest(const cs_eg2_q8*t,const float*projection,
                   float logit_scale,float blank_logit,cs_eg2_score*out){
  if(!t||!t->vec||!t->inverse_norm||!t->count||!projection||!out||
     !isfinite(logit_scale)||logit_scale<=0||!isfinite(blank_logit))return -1;
  float maxabs=0;
  for(int j=0;j<CS_EG2_TEXT_DIM;j++){
    if(!isfinite(projection[j]))return -1;
    float v=fabsf(projection[j]);if(v>maxabs)maxabs=v;
  }
  int8_t x[CS_EG2_TEXT_DIM];double qnorm=0;
  if(maxabs<=1e-20f)return -1;
  for(int j=0;j<CS_EG2_TEXT_DIM;j++){
    float val=roundf(127.f*projection[j]/maxabs);
    if(val>127)val=127;
    if(val< -127)val= -127;
    x[j]=(int8_t)val;qnorm+=(double)val*val;
  }
  if(qnorm==0)return -1;
  float invq=1.f/(float)sqrt(qnorm);
  float best=-2;size_t bestid=0;
  for(size_t i=0;i<t->count;i++){
    int sum=cs_eg2_dot_q8(x,t->vec+i*CS_EG2_TEXT_DIM);
    float cos=(float)sum*(invq*t->inverse_norm[i]);
    if(cos>best){best=cos;bestid=i;}
  }
  out->word=(int32_t)bestid;out->top1=logit_scale*best;out->blank=blank_logit;
  return 0;
}
/* n projected vectors (row stride ld) against the whole table at once.
 * Gives exactly the results of n cs_eg2_nearest calls (first maximum wins),
 * but the table is read once per cache block instead of once per token, and
 * the vocabulary is split across OpenMP threads. */
#define CS_EG2_NEAREST_BLOCK 128
static inline int cs_eg2_nearest_batch(const cs_eg2_q8*t,const float*projection,
        size_t ld,float logit_scale,const float*blank,size_t n,cs_eg2_score*out){
  if(!t||!t->vec||!t->inverse_norm||!t->count||!projection||!blank||!out||!n||
     !isfinite(logit_scale)||logit_scale<=0)return -1;
  int8_t*xq=malloc(n*CS_EG2_TEXT_DIM);float*invq=malloc(n*sizeof(float));
  int nth=1,rc=-1;
#ifdef _OPENMP
  nth=omp_get_max_threads();if(nth<1)nth=1;
#endif
  float*best=malloc((size_t)nth*n*sizeof(float));size_t*bid=malloc((size_t)nth*n*sizeof(size_t));
  int16_t*x16=NULL;
  if(!xq||!invq||!best||!bid)goto done;
  for(size_t i=0;i<n;i++){
    const float*pr=projection+i*ld;float maxabs=0;double qnorm=0;
    if(!isfinite(blank[i]))goto done;
    for(int j=0;j<CS_EG2_TEXT_DIM;j++){
      if(!isfinite(pr[j]))goto done;
      float v=fabsf(pr[j]);if(v>maxabs)maxabs=v;
    }
    if(maxabs<=1e-20f)goto done;
    for(int j=0;j<CS_EG2_TEXT_DIM;j++){
      float val=roundf(127.f*pr[j]/maxabs);
      if(val>127)val=127;
      if(val< -127)val= -127;
      xq[i*CS_EG2_TEXT_DIM+j]=(int8_t)val;qnorm+=(double)val*val;
    }
    if(qnorm==0)goto done;
    invq[i]=1.f/(float)sqrt(qnorm);
  }
#if CATS_SIMD_X86
  if(cats_cpu_simd_level()==1&&n>=4){   /* AVX without AVX2: shared-row kernel */
    x16=malloc(n*CS_EG2_TEXT_DIM*sizeof(int16_t));
    if(x16)for(size_t k=0;k<n*CS_EG2_TEXT_DIM;k++)x16[k]=xq[k];
  }
#endif
  for(size_t k=0;k<(size_t)nth*n;k++){best[k]=-2;bid[k]=0;}
  const long blocks=(long)((t->count+CS_EG2_NEAREST_BLOCK-1)/CS_EG2_NEAREST_BLOCK);
#pragma omp parallel
  {
    int tid=0;
#ifdef _OPENMP
    tid=omp_get_thread_num();
#endif
    float*lb=best+(size_t)tid*n;size_t*li=bid+(size_t)tid*n;
#pragma omp for schedule(static)
    for(long blk=0;blk<blocks;blk++){
      size_t w0=(size_t)blk*CS_EG2_NEAREST_BLOCK,w1=w0+CS_EG2_NEAREST_BLOCK;
      if(w1>t->count)w1=t->count;
      size_t i=0;
#if CATS_SIMD_X86
      if(x16)for(;i+4<=n;i+=4){
        for(size_t w=w0;w<w1;w++){
          int sums[4];
          cs_eg2_dot4_avx1(x16+i*CS_EG2_TEXT_DIM,t->vec+w*CS_EG2_TEXT_DIM,sums);
          for(int q=0;q<4;q++){
            float cos=(float)sums[q]*(invq[i+q]*t->inverse_norm[w]);
            if(cos>lb[i+q]){lb[i+q]=cos;li[i+q]=w;}
          }
        }
      }
#endif
      for(;i<n;i++){
        const int8_t*x=xq+i*CS_EG2_TEXT_DIM;
        for(size_t w=w0;w<w1;w++){
          int sum=cs_eg2_dot_q8(x,t->vec+w*CS_EG2_TEXT_DIM);
          float cos=(float)sum*(invq[i]*t->inverse_norm[w]);
          if(cos>lb[i]){lb[i]=cos;li[i]=w;}
        }
      }
    }
  }
  for(size_t i=0;i<n;i++){
    float b=-2;size_t id=0;
    for(int k=0;k<nth;k++){
      float c=best[(size_t)k*n+i];size_t w=bid[(size_t)k*n+i];
      if(c>b||(c==b&&w<id)){b=c;id=w;}
    }
    out[i].word=(int32_t)id;out[i].top1=logit_scale*b;out[i].blank=blank[i];
  }
  rc=0;
done:
  free(xq);free(invq);free(best);free(bid);free(x16);return rc;
}
/* Convert precomputed EmbeddingGemma word vectors from EG2V float32 (header:
 * magic/version/count + count x 768 little-endian f32) into native EG2Q.
 * Streaming conversion holds one row at a time, no extra dependencies.
 * These are table vectors, not reader weights or audio embeddings. */
static inline int cs_eg2_quantize_table(const char*src,const char*dst){
  if(!src||!dst)return -1;
  FILE*in=fopen(src,"rb");if(!in)return -1;
  FILE*out=NULL;int rc=-1;unsigned char h[16]={0},q[768];float row[768];
  if(fread(h,1,12,in)!=12||memcmp(h,"EG2V",4)||cs_u32(h+4)!=1||
     !cs_u32(h+8)||cs_u32(h+8)>1000000u)goto done;
  unsigned count=cs_u32(h+8);
  out=fopen(dst,"wb");if(!out)goto done;
  memcpy(h,"EG2Q",4);cs_put32(h+12,768);
  if(fwrite(h,1,16,out)!=16)goto done;
  for(unsigned i=0;i<count;i++){
    if(fread(row,4,768,in)!=768)goto done;
    float maxabs=0;
    for(int j=0;j<768;j++){
      if(!isfinite(row[j]))goto done;
      float a=fabsf(row[j]);if(a>maxabs)maxabs=a;
    }
    if(maxabs<=1e-20f)goto done;
    for(int j=0;j<768;j++){
      float v=roundf(row[j]*127.f/maxabs);
      if(v>127)v=127;
      if(v< -127)v= -127;
      q[j]=(unsigned char)(int8_t)v;
    }
    if(fwrite(q,1,768,out)!=768)goto done;
  }
  if(fgetc(in)!=EOF||ferror(in))goto done;
  rc=0;
done:
  fclose(in);if(out&&fclose(out))rc=-1;
  if(rc&&out)remove(dst);
  return rc;
}

/* ========================================================================
 * Stage 28: native-C EmbeddingGemma2 Transcribe Reader-A (v1/v2) backend.
 * Binary EG2R contains ONLY the independently trained small reader weights;
 * not the 300M audio encoder or the 200k-word lookup table.
 * Architecture matches reader/train/model.py (ReaderA, arch 'a'):
 * pre-LN, RoPE 8 heads, full bidirectional attention, exact erf GELU.
 * ======================================================================== */
#define CS_EG2_RD 512
#define CS_EG2_RF 2048
#define CS_EG2_RH 8
#define CS_EG2_RHD 64
#define CS_EG2_RMAX 12

typedef struct {
  float *ln1w,*ln1b,*qkvw,*qkvb,*projw,*projb;
  float *ln2w,*ln2b,*ff1w,*ff1b,*ff2w,*ff2b;
} cs_eg2_rblock;
typedef struct {
  int layers;
  float scale;
  float *storage;
  size_t nfloat;
  float *inw,*inb,*inputw,*inputb,*outw,*outb,*outputw,*outputb;
  float *blankw,*blankb;
  cs_eg2_rblock block[CS_EG2_RMAX];
} cs_eg2_reader;
static inline void cs_eg2_reader_free(cs_eg2_reader*r){
  if(r){CS_GPU_FORGET();free(r->storage);memset(r,0,sizeof(*r));}
}
static inline size_t cs_eg2_reader_params(int layers){
  const size_t d=CS_EG2_RD,f=CS_EG2_RF;
  if(layers<1||layers>CS_EG2_RMAX)return 0;
  return (2*d+d*d+d) + (size_t)layers*(
    2*d+3*d*d+3*d+d*d+d+2*d+f*d+f+d*f+d) +
    (2*d+768*d+768)+(768+1);
}
/* The EG2R header is 24 bytes: magic, version, layers, d, ffn, nfloat.
 * The 4-byte logit scale is appended after weights (trained exp(log_scale)).
 * Converter only supports torch weights_only=True and arch='a'. */
static inline int cs_eg2_reader_load(cs_eg2_reader*r,const char*path){
  if(!r||!path)return -1;
  memset(r,0,sizeof(*r));
  FILE*fp=cs_eg2_file_open(path);if(!fp)return -1;
  unsigned char h[24];int rc=-1;
  if(fread(h,1,24,fp)!=24||memcmp(h,"EG2R",4)||cs_u32(h+4)!=1||
     cs_u32(h+12)!=CS_EG2_RD||cs_u32(h+16)!=CS_EG2_RF)goto done;
  uint32_t layers=cs_u32(h+8),ns=cs_u32(h+20);
  if(layers<1||layers>CS_EG2_RMAX||ns!=cs_eg2_reader_params((int)layers))goto done;
  if((uint64_t)ns*4u>SIZE_MAX)goto done;
  r->storage=(float*)malloc((size_t)ns*sizeof(float));if(!r->storage)goto done;
  r->nfloat=ns;r->layers=(int)layers;
  if(fread(r->storage,4,ns,fp)!=ns)goto done;
  /* Binary weights are f32 little-endian, independent of host endianness. */
  const uint16_t one=1;
  if(*(const unsigned char*)&one!=1){
    unsigned char*b=(unsigned char*)r->storage;
    for(size_t j=0;j<ns;j++){unsigned char*t=b+j*4,c=t[0];t[0]=t[3];t[3]=c;c=t[1];t[1]=t[2];t[2]=c;}
  }
  for(size_t j=0;j<ns;j++)if(!isfinite(r->storage[j]))goto done;
  unsigned char sb[4];if(fread(sb,1,4,fp)!=4||fgetc(fp)!=EOF||ferror(fp))goto done;
  uint32_t bits=cs_u32(sb);memcpy(&r->scale,&bits,4);
  if(!isfinite(r->scale)||r->scale<=0||r->scale>1e6f)goto done;
  size_t pos=0;
#define CS_EG2_TAKE(dst,count) do{size_t nn=(size_t)(count);if(nn>ns-pos)goto done;(dst)=r->storage+pos;pos+=nn;}while(0)
  CS_EG2_TAKE(r->inw,512);CS_EG2_TAKE(r->inb,512);
  CS_EG2_TAKE(r->inputw,512*512);CS_EG2_TAKE(r->inputb,512);
  for(int i=0;i<r->layers;i++){
    cs_eg2_rblock*b=&r->block[i];
    CS_EG2_TAKE(b->ln1w,512);CS_EG2_TAKE(b->ln1b,512);
    CS_EG2_TAKE(b->qkvw,1536*512);CS_EG2_TAKE(b->qkvb,1536);
    CS_EG2_TAKE(b->projw,512*512);CS_EG2_TAKE(b->projb,512);
    CS_EG2_TAKE(b->ln2w,512);CS_EG2_TAKE(b->ln2b,512);
    CS_EG2_TAKE(b->ff1w,2048*512);CS_EG2_TAKE(b->ff1b,2048);
    CS_EG2_TAKE(b->ff2w,512*2048);CS_EG2_TAKE(b->ff2b,512);
  }
  CS_EG2_TAKE(r->outw,512);CS_EG2_TAKE(r->outb,512);
  CS_EG2_TAKE(r->outputw,768*512);CS_EG2_TAKE(r->outputb,768);
  CS_EG2_TAKE(r->blankw,768);CS_EG2_TAKE(r->blankb,1);
#undef CS_EG2_TAKE
  if(pos!=ns)goto done;
  rc=0;
done:
  fclose(fp);if(rc)cs_eg2_reader_free(r);return rc;
}
static inline void cs_eg2_rnorm(const float*x,const float*gamma,const float*beta,
                                float*y,int d){
  double mean=0,v=0;
  for(int j=0;j<d;j++)mean+=x[j];mean/=d;
  for(int j=0;j<d;j++){double dx=x[j]-mean;v+=dx*dx;}
  float inv=(float)(1.0/sqrt(v/d+1e-5));
  for(int j=0;j<d;j++)y[j]=(x[j]-(float)mean)*inv*gamma[j]+beta[j];
}
/* Stage 31: dependency-free SIMD matrix-vector kernel for audio tower.
 * Runtime dispatch honors existing CATS_SIMD override. FP32 FMA may
 * differ slightly from the former scalar FP64 accumulation path.
 * No ISA-specific instructions are emitted on non-x86 targets. */
#if CATS_SIMD_X86
CATS_AVX2_FN static float cs_eg2_dot_fma(const float*x,const float*w,size_t n){
    __m256 a=_mm256_setzero_ps(),b=_mm256_setzero_ps(),
           c=_mm256_setzero_ps(),d=_mm256_setzero_ps();
    size_t i=0;
    for(;i+32<=n;i+=32){
        a=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(w+i),a);
        b=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_loadu_ps(w+i+8),b);
        c=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_loadu_ps(w+i+16),c);
        d=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_loadu_ps(w+i+24),d);
    }
    a=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
    float parts[8];_mm256_storeu_ps(parts,a);
    double sum=0;for(int j=0;j<8;j++)sum+=parts[j];
    for(;i<n;i++)sum+=(double)x[i]*w[i];
    return (float)sum;
}
#endif
static inline float cs_eg2_dot_f32(const float*x,const float*w,size_t n){
#if CATS_SIMD_X86
    if(n>=32&&cats_cpu_simd_level()>=2)return cs_eg2_dot_fma(x,w,n);
#endif
    double sum=0;for(size_t j=0;j<n;j++)sum+=(double)x[j]*w[j];
    return (float)sum;
}

static inline void cs_eg2_rlinear(float*y,const float*w,const float*b,
                                  const float*x,int rows,int cols){
  for(int i=0;i<rows;i++){
    const float*wi=w+(size_t)i*cols;
    y[i]=b[i]+cs_eg2_dot_f32(x,wi,(size_t)cols);
  }
}

/* ------------------------------------------------------------------------
 * Batched dense kernel:  C[n,out] = X[n,in] . W[out,in]^T  (+ bias[out]).
 * W is stored one output row after another, exactly as in the model files.
 * A 4-token x 2-output register tile runs over K blocks of 512 so the
 * weight panel stays in L1/L2 instead of being re-streamed from RAM for
 * every token; independent (token chunk, output block) tasks are spread
 * over OpenMP threads when built with -fopenmp (serial otherwise).
 * Dispatch follows CATS_SIMD: AVX2/FMA, plain AVX (no FMA), or scalar
 * (double accumulation, as before).
 * ------------------------------------------------------------------------ */
#define CS_EG2_GEMM_OB 16
#define CS_EG2_GEMM_TB 256
#define CS_EG2_GEMM_KC 512
typedef void (*cs_eg2_tile_fn)(float*,size_t,const float*,size_t,const float*,
                               size_t,size_t,size_t,size_t,int,const float*);
static void cs_eg2_tile_generic(float*c,size_t ldc,const float*x,size_t ldx,
        const float*w,size_t ldw,size_t klen,size_t nt,size_t no,int acc,const float*bias){
    for(size_t t=0;t<nt;t++)for(size_t o=0;o<no;o++){
        double sum=0;
        for(size_t k=0;k<klen;k++)sum+=(double)x[t*ldx+k]*w[o*ldw+k];
        float r=(float)sum;
        if(acc)r+=c[t*ldc+o];else if(bias)r+=bias[o];
        c[t*ldc+o]=r;
    }
}
#if CATS_SIMD_X86
CATS_AVX_FN static inline float cs_eg2_hsum8(__m256 v){
    __m128 s=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
    s=_mm_hadd_ps(s,s);s=_mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}
#define CS_EG2_MAC_AVX(a,b,c) _mm256_add_ps((c),_mm256_mul_ps((a),(b)))
#define CS_EG2_MAC_FMA(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#define CS_EG2_DEF_TILE(NAME,ATTR,MAC) \
ATTR static void NAME(float*c,size_t ldc,const float*x,size_t ldx,const float*w, \
        size_t ldw,size_t klen,size_t nt,size_t no,int acc,const float*bias){ \
    if(nt!=4||no!=2){cs_eg2_tile_generic(c,ldc,x,ldx,w,ldw,klen,nt,no,acc,bias);return;} \
    const float*x0=x,*x1=x+ldx,*x2=x+2*ldx,*x3=x+3*ldx,*w0=w,*w1=w+ldw; \
    __m256 a00=_mm256_setzero_ps(),a01=a00,a10=a00,a11=a00,a20=a00,a21=a00,a30=a00,a31=a00; \
    size_t k=0; \
    for(;k+8<=klen;k+=8){ \
        __m256 b0=_mm256_loadu_ps(w0+k),b1=_mm256_loadu_ps(w1+k),v; \
        v=_mm256_loadu_ps(x0+k);a00=MAC(v,b0,a00);a01=MAC(v,b1,a01); \
        v=_mm256_loadu_ps(x1+k);a10=MAC(v,b0,a10);a11=MAC(v,b1,a11); \
        v=_mm256_loadu_ps(x2+k);a20=MAC(v,b0,a20);a21=MAC(v,b1,a21); \
        v=_mm256_loadu_ps(x3+k);a30=MAC(v,b0,a30);a31=MAC(v,b1,a31); \
    } \
    float r[4][2]={{cs_eg2_hsum8(a00),cs_eg2_hsum8(a01)},{cs_eg2_hsum8(a10),cs_eg2_hsum8(a11)}, \
                   {cs_eg2_hsum8(a20),cs_eg2_hsum8(a21)},{cs_eg2_hsum8(a30),cs_eg2_hsum8(a31)}}; \
    for(;k<klen;k++){ \
        r[0][0]+=x0[k]*w0[k];r[0][1]+=x0[k]*w1[k];r[1][0]+=x1[k]*w0[k];r[1][1]+=x1[k]*w1[k]; \
        r[2][0]+=x2[k]*w0[k];r[2][1]+=x2[k]*w1[k];r[3][0]+=x3[k]*w0[k];r[3][1]+=x3[k]*w1[k]; \
    } \
    for(int t=0;t<4;t++)for(int o=0;o<2;o++){ \
        float v=r[t][o]; \
        if(acc)v+=c[(size_t)t*ldc+o];else if(bias)v+=bias[o]; \
        c[(size_t)t*ldc+o]=v; \
    } \
}
CS_EG2_DEF_TILE(cs_eg2_tile_avx,CATS_AVX_FN,CS_EG2_MAC_AVX)
CS_EG2_DEF_TILE(cs_eg2_tile_avx2,CATS_AVX2_FN,CS_EG2_MAC_FMA)
#endif
static inline void cs_eg2_gemm_nt_cpu(float*c,const float*x,const float*w,size_t n,
        size_t in,size_t out,const float*bias);
static inline void cs_eg2_gemm_nt(float*c,const float*x,const float*w,size_t n,
        size_t in,size_t out,const float*bias){
    if(!n||!in||!out)return;
#ifdef CATS_GPU
    /* Weights are cached on the GPU by pointer: whoever frees or rewrites W
     * must call cs_gpu_forget_weights() (CS_GPU_FORGET) first.  The rows are
     * shared out: the GPU takes whole 128-row tiles (about CS_GPU_SHARE percent
     * of them), the CPU computes the remaining rows while the GPU runs. */
    if(cs_gpu_ready()&&n>=32&&in%16==0&&(double)n*in*out>=2.0e7){
        size_t ng=((n*cs_gpu_share()/100+CS_GPU_TILE-1)/CS_GPU_TILE)*CS_GPU_TILE;
        if(ng>n||n-ng<16)ng=n;
        if(!cs_gpu_gemm_begin(x,w,ng,in,out,bias)){
            if(ng<n)cs_eg2_gemm_nt_cpu(c+ng*out,x+ng*in,w,n-ng,in,out,bias);
            if(!cs_gpu_gemm_end(c))return;
            cs_eg2_gemm_nt_cpu(c,x,w,ng,in,out,bias);   /* GPU failed: redo its rows */
            return;
        }
    }
#endif
    cs_eg2_gemm_nt_cpu(c,x,w,n,in,out,bias);
}
static inline void cs_eg2_gemm_nt_cpu(float*c,const float*x,const float*w,size_t n,
        size_t in,size_t out,const float*bias){
    if(!n||!in||!out)return;
    cs_eg2_tile_fn tile=cs_eg2_tile_generic;
#if CATS_SIMD_X86
    {int level=cats_cpu_simd_level();
     if(level>=2)tile=cs_eg2_tile_avx2;else if(level==1)tile=cs_eg2_tile_avx;}
#endif
    const long ntb=(long)((n+CS_EG2_GEMM_TB-1)/CS_EG2_GEMM_TB);
    const long nob=(long)((out+CS_EG2_GEMM_OB-1)/CS_EG2_GEMM_OB);
    const int big=(double)n*(double)in*(double)out>1.0e6;
    (void)big;
#pragma omp parallel for collapse(2) schedule(dynamic,1) if(big)
    for(long tb=0;tb<ntb;tb++)for(long ob=0;ob<nob;ob++){
        size_t t0=(size_t)tb*CS_EG2_GEMM_TB,t1=t0+CS_EG2_GEMM_TB;if(t1>n)t1=n;
        size_t o0=(size_t)ob*CS_EG2_GEMM_OB,o1=o0+CS_EG2_GEMM_OB;if(o1>out)o1=out;
        for(size_t k0=0;k0<in;k0+=CS_EG2_GEMM_KC){
            size_t klen=in-k0<CS_EG2_GEMM_KC?in-k0:CS_EG2_GEMM_KC;
            int acc=k0!=0;
            for(size_t t=t0;t<t1;t+=4){
                size_t nt=t1-t<4?t1-t:4;
                for(size_t o=o0;o<o1;o+=2){
                    size_t no=o1-o<2?o1-o:2;
                    tile(c+t*out+o,out,x+t*in+k0,in,w+o*in+k0,in,klen,nt,no,acc,
                         bias?bias+o:NULL);
                }
            }
        }
    }
}
/* Same as cs_eg2_rlinear for n tokens: y[n,rows] = x[n,cols] W^T + b. */
static inline void cs_eg2_rlinear_batch(float*y,const float*w,const float*b,
        const float*x,int rows,int cols,size_t n){
    cs_eg2_gemm_nt(y,x,w,n,(size_t)cols,(size_t)rows,b);
}
static inline void cs_eg2_rrope(float*q,float*k,size_t t,const float*invfreq){
  for(int head=0;head<CS_EG2_RH;head++){
    float*qh=q+head*CS_EG2_RHD,*kh=k+head*CS_EG2_RHD;
    for(int j=0;j<CS_EG2_RHD/2;j++){
      double angle=(double)t*invfreq[j];float c=(float)cos(angle),s=(float)sin(angle);
      float a=qh[j],b=qh[j+32];qh[j]=a*c-b*s;qh[j+32]=b*c+a*s;
      a=kh[j];b=kh[j+32];kh[j]=a*c-b*s;kh[j+32]=b*c+a*s;
    }
  }
}
/* soft_tokens: [tokens,512]; projection: [tokens,768] unit norm,
 * blank: [tokens]. Only valid tokens are computed; no padded query outputs.
 * Requires ~14MB scratch at 750 tokens, independent of weight size. */
static inline int cs_eg2_reader_run(const cs_eg2_reader*r,const float*soft,
                        size_t tokens,float*projection,float*blank){
  if(!r||!r->storage||!soft||!projection||!blank||!tokens||tokens>CS_EG2_TOKENS)return -1;
  const size_t T=tokens;
  if(T>SIZE_MAX/(CS_EG2_RF*sizeof(float)))return -1;
  float *x=malloc(T*512*sizeof(float)),*norm=malloc(T*512*sizeof(float));
  float *qkv=malloc(T*1536*sizeof(float)),*heads=malloc(T*512*sizeof(float));
  float *ff=malloc(T*2048*sizeof(float)),*tmp=malloc(T*512*sizeof(float));
  int rc=-1,bad=0;
  if(!x||!norm||!qkv||!heads||!ff||!tmp)goto done;
  for(size_t i=0;i<T*512;i++)if(!isfinite(soft[i]))goto done;
#pragma omp parallel for schedule(static)
  for(long t=0;t<(long)T;t++)cs_eg2_rnorm(soft+t*512,r->inw,r->inb,norm+t*512,512);
  cs_eg2_rlinear_batch(x,r->inputw,r->inputb,norm,512,512,T);
  float invfreq[32];
  for(int j=0;j<32;j++)invfreq[j]=powf(10000.0f,-(float)(2*j)/64.f);
  for(int layer=0;layer<r->layers;layer++){
    const cs_eg2_rblock*b=&r->block[layer];
#pragma omp parallel for schedule(static)
    for(long t=0;t<(long)T;t++)cs_eg2_rnorm(x+t*512,b->ln1w,b->ln1b,norm+t*512,512);
    cs_eg2_rlinear_batch(qkv,b->qkvw,b->qkvb,norm,1536,512,T);
#pragma omp parallel for schedule(static)
    for(long t=0;t<(long)T;t++)cs_eg2_rrope(qkv+t*1536,qkv+t*1536+512,(size_t)t,invfreq);
#pragma omp parallel for schedule(static)
    for(long tt=0;tt<(long)T;tt++){
      const size_t t=(size_t)tt;
      float score[CS_EG2_TOKENS];
      const float*q=qkv+t*1536;
      for(int h=0;h<CS_EG2_RH;h++){
        const float*qh=q+h*64;float hi=-INFINITY;
        for(size_t s=0;s<T;s++){
          const float*kh=qkv+s*1536+512+h*64;
          float z=0;
          for(int j=0;j<64;j++)z+=qh[j]*kh[j];
          z*=0.125f;score[s]=z;
          if(z>hi)hi=z;
        }
        float denom=0;
        for(size_t s=0;s<T;s++){score[s]=expf(score[s]-hi);denom+=score[s];}
        if(!(denom>0)||!isfinite(denom)){
#pragma omp atomic write
          bad=1;
          continue;
        }
        float *dst=heads+t*512+h*64;
        for(int j=0;j<64;j++)dst[j]=0;
        for(size_t s=0;s<T;s++){
          float coeff=score[s]/denom;
          const float*vs=qkv+s*1536+1024+h*64;
          for(int j=0;j<64;j++)dst[j]+=coeff*vs[j];
        }
      }
    }
    if(bad)goto done;
    cs_eg2_rlinear_batch(tmp,b->projw,b->projb,heads,512,512,T);
#pragma omp parallel for schedule(static)
    for(long t=0;t<(long)T;t++){
      for(int j=0;j<512;j++)x[t*512+j]+=tmp[t*512+j];
      cs_eg2_rnorm(x+t*512,b->ln2w,b->ln2b,norm+t*512,512);
    }
    cs_eg2_rlinear_batch(ff,b->ff1w,b->ff1b,norm,2048,512,T);
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)(T*2048);i++){
      float v=ff[i];ff[i]=0.5f*v*erfcf(-v*0.70710678118654752f);
    }
    cs_eg2_rlinear_batch(tmp,b->ff2w,b->ff2b,ff,512,2048,T);
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)(T*512);i++)x[i]+=tmp[i];
  }
#pragma omp parallel for schedule(static)
  for(long t=0;t<(long)T;t++)cs_eg2_rnorm(x+t*512,r->outw,r->outb,norm+t*512,512);
  cs_eg2_rlinear_batch(projection,r->outputw,r->outputb,norm,768,512,T);
#pragma omp parallel for schedule(static)
  for(long t=0;t<(long)T;t++){
    float*v=projection+t*768;
    float bl=r->blankb[0];double vnorm=0;
    for(int j=0;j<768;j++){bl+=v[j]*r->blankw[j];vnorm+=(double)v[j]*v[j];}
    if(!isfinite(bl)||vnorm<=1e-30||!isfinite(vnorm)){
#pragma omp atomic write
      bad=1;
      continue;
    }
    float inv=(float)(1.0/sqrt(vnorm));
    for(int j=0;j<768;j++)v[j]*=inv;
    blank[t]=bl;
  }
  if(bad)goto done;
  rc=0;
done:
  free(x);free(norm);free(qkv);free(heads);free(ff);free(tmp);
  return rc;
}
/* EG2T file: 12-byte header (magic/version/count), packed f32 [count,512]. */
static inline int cs_eg2_soft_read(const char*path,float**out,size_t*count){
  if(!path||!out||!count)return -1;*out=NULL;*count=0;
  FILE*fp=cs_eg2_file_open(path);if(!fp)return -1;
  unsigned char h[12];float*buf=NULL;int rc=-1;
  if(fread(h,1,12,fp)!=12||memcmp(h,"EG2T",4)||cs_u32(h+4)!=1||
     cs_u32(h+8)<1||cs_u32(h+8)>CS_EG2_TOKENS)goto done;
  size_t n=cs_u32(h+8);buf=(float*)malloc(n*512*sizeof(float));if(!buf)goto done;
  if(fread(buf,sizeof(float),n*512,fp)!=n*512||fgetc(fp)!=EOF||ferror(fp))goto done;
  const uint16_t one=1;
  if(*(const unsigned char*)&one!=1){unsigned char*b=(unsigned char*)buf;
    for(size_t j=0;j<n*512;j++){unsigned char*p=b+j*4,t=p[0];p[0]=p[3];p[3]=t;t=p[1];p[1]=p[2];p[2]=t;}}
  *out=buf;*count=n;buf=NULL;rc=0;
done:free(buf);fclose(fp);return rc;
}
static inline int cs_eg2_soft_write(const char*path,const float*data,size_t count){
  if(!path||!data||!count||count>CS_EG2_TOKENS)return -1;
  FILE*f=fopen(path,"wb");if(!f)return -1;
  unsigned char h[12];memcpy(h,"EG2T",4);cs_put32(h+4,1);cs_put32(h+8,(uint32_t)count);
  int ok=fwrite(h,1,12,f)==12;
  for(size_t i=0;ok&&i<count*512;i++){
    if(!isfinite(data[i])){ok=0;break;}
    unsigned char bytes[4];uint32_t word;memcpy(&word,&data[i],4);cs_put32(bytes,word);
    ok=fwrite(bytes,1,4,f)==4;
  }
  if(fclose(f)!=0)ok=0;
  if(!ok)remove(path);
  return ok?0:-1;
}

/* ========================================================================
 * EmbeddingGemma 2 ONNX-backed word-reader path (experimental).
 * Expected interfaces are the published independent port of the architecture:
 * encoder: features [1,3000,128], half_mask [1,1500]
 *       -> soft_tokens [1,750,512]
 * reader: soft_tokens [1,750,512], token_mask [1,750]
 *       -> top1_index, top1_logit, blank_logit [1,750].
 * This partial native ONNX engine cannot yet run the actual pretrained
 * Gemma4 conformer / full transformer reader. Reject unsupported operators;
 * NEVER substitute random weights or claim a validated ASR transcript.
 */
static inline int cx_eg2_graph_has(const cx_graph*g,const char*name,int output){
  int n=output?g->noutput:g->ninput;
  for(int i=0;i<n;i++)if(!strcmp(output?g->output[i]:g->input[i],name))return 1;
  return 0;
}
static inline int cx_eg2_open(cx_session*enc,cx_session*reader,
                              const char*encoder_file,const char*reader_file,int use_q8){
  if(!enc||!reader||!encoder_file||!reader_file)return -1;
  memset(enc,0,sizeof(*enc));memset(reader,0,sizeof(*reader));
  if(cx_session_open(enc,encoder_file)||cx_session_open(reader,reader_file))goto bad;
  if(!cx_eg2_graph_has(&enc->graph,"features",0)||
     !cx_eg2_graph_has(&enc->graph,"half_mask",0)||
     !cx_eg2_graph_has(&enc->graph,"soft_tokens",1)||
     !cx_eg2_graph_has(&reader->graph,"soft_tokens",0)||
     !cx_eg2_graph_has(&reader->graph,"token_mask",0)||
     !(use_q8?cx_eg2_graph_has(&reader->graph,"projected_tokens",1):
                (cx_eg2_graph_has(&reader->graph,"top1_index",1)&&cx_eg2_graph_has(&reader->graph,"top1_logit",1)))||
     !cx_eg2_graph_has(&reader->graph,"blank_logit",1)){
    fprintf(stderr,"eg2-asr: incompatible ONNX input/output names (see README)\n");goto bad;
  }
  if(cx_graph_preflight(&enc->graph,stderr)||cx_graph_preflight(&reader->graph,stderr)){
    fprintf(stderr,"eg2-asr: unsupported encoder/reader operators; use a compatible exported backend\n");goto bad;
  }
  enc->cache_weights=1;reader->cache_weights=1;
  return 0;
bad:cx_session_close(enc);cx_session_close(reader);return -1;
}
static inline int cx_eg2_transcribe_wav_ext_lang(const char*encoder_file,const char*reader_file,
      const char*vocab_file,const char*wav_file,char*out,size_t capacity,
      const char*save_scores,const char*q8_path,float logit_scale,
      const char*native_reader_file,int japanese){
  if(!encoder_file||(!reader_file&&!native_reader_file)||
     (reader_file&&native_reader_file)||!vocab_file||!wav_file||!out||!capacity)return -1;
  out[0]=0;
  cs_wave wave={0};cs_eg2_vocab vocab={0};
  cx_session encoder={0},reader={0};
  cs_eg2_frontend *front=NULL;cs_eg2_q8 q8={0};cs_eg2_reader native={0};
  float*feat=NULL,*audio=NULL,*projected=NULL,*native_blank=NULL;
  cs_eg2_score*scores=NULL;size_t count=0,scorecap=0;
  int ret=-1;
  if(cs_wave_read(wav_file,&wave)){
    fprintf(stderr,"eg2-asr: unable to open WAV: %s\n",wav_file);goto cleanup;
  }
  if(cs_eg2_vocab_load(&vocab,vocab_file)){
    fprintf(stderr,"eg2-asr: missing plain line-indexed word vocabulary: %s\n",vocab_file);goto cleanup;
  }
  if(q8_path&&(cs_eg2_q8_open(&q8,q8_path)||q8.count!=vocab.count)){
    fprintf(stderr,"eg2-asr: invalid q8 word table, or vocabulary count mismatch\n");goto cleanup;
  }
  if(native_reader_file){
    if(!q8_path){fprintf(stderr,"eg2-asr: -R native Reader requires -Q word table\n");goto cleanup;}
    if(cs_eg2_reader_load(&native,native_reader_file)){
      fprintf(stderr,"eg2-asr: malformed EG2R Reader weights\n");goto cleanup;
    }
    projected=malloc((size_t)CS_EG2_TOKENS*768*sizeof(float));
    native_blank=malloc((size_t)CS_EG2_TOKENS*sizeof(float));
    if(!projected||!native_blank)goto cleanup;
    logit_scale=native.scale;
    if(cx_session_open(&encoder,encoder_file)||
       !cx_eg2_graph_has(&encoder.graph,"features",0)||
       !cx_eg2_graph_has(&encoder.graph,"half_mask",0)||
       !cx_eg2_graph_has(&encoder.graph,"soft_tokens",1)||
       cx_graph_preflight(&encoder.graph,stderr)){
      fprintf(stderr,"eg2-asr: native Reader requires a compatible audio encoder ONNX\n");goto cleanup;
    }
    encoder.cache_weights=1;
  }else{
    if(q8_path&&(!isfinite(logit_scale)||logit_scale<=0))goto cleanup;
    if(cx_eg2_open(&encoder,&reader,encoder_file,reader_file,q8_path!=NULL))goto cleanup;
  }
  front=(cs_eg2_frontend*)malloc(sizeof(*front));
  feat=(float*)malloc((size_t)CS_EG2_BUCKET*CS_EG2_MEL*sizeof(*feat));
  if(!front||!feat)goto cleanup;
  cs_eg2_frontend_init(front);
  if(wave.rate!=CS_EG2_RATE){
    size_t newlen=0;
    if(cs_resample(wave.pcm,wave.count,wave.rate,CS_EG2_RATE,&audio,&newlen))goto cleanup;
    /* Frontend always consumes a 16 kHz mono float waveform. */
    cs_wave_free(&wave);wave.pcm=audio;wave.count=newlen;wave.rate=CS_EG2_RATE;
    audio=NULL;
  }
  scorecap=(wave.count+639)/640+CS_EG2_TOKENS;
  if(scorecap>SIZE_MAX/sizeof(*scores))goto cleanup;
  scores=(cs_eg2_score*)malloc(scorecap*sizeof(*scores));if(!scores)goto cleanup;
  const int64_t fd[3]={1,CS_EG2_BUCKET,CS_EG2_MEL};
  const int64_t hd[2]={1,1500},sd[3]={1,750,512},td[2]={1,750};
  float half[1500],token_mask[750];
  /* Advance by 653 tokens (~26.12 s); keep 96 tokens from each subsequent
   * window to account for causal encoder context. Drop the last token of
   * nonfinal windows (missing look-ahead) and discard overlapping labels. */
  for(size_t start=0,window=0;start<wave.count;window++){
    if(start>SIZE_MAX-(size_t)CS_EG2_STEP_TOKENS*640)goto cleanup;
    size_t remaining=wave.count-start,ns=remaining<480000?remaining:480000;
    size_t frames=0,valid=0;
    if(cs_eg2_features(front,wave.pcm+start,ns,feat,&frames))goto cleanup;
    cs_eg2_masks(frames,half,token_mask,&valid);
    if(!valid)break;
    if(cx_session_feed(&encoder,"features",fd,3,1,feat,(uint64_t)CS_EG2_BUCKET*CS_EG2_MEL)||
       cx_session_feed(&encoder,"half_mask",hd,2,1,half,1500)||
       cx_session_run(&encoder)){
      fprintf(stderr,"eg2-asr: audio encoder ONNX inference failed: %s\n",encoder.graph.error);goto cleanup;
    }
    const cx_tensor*soft=cx_session_output(&encoder,"soft_tokens");
    if(!soft||soft->n!=(uint64_t)CS_EG2_TOKENS*512||soft->type!=1){
      fprintf(stderr,"eg2-asr: encoder must return float32 [1,750,512]\n");goto cleanup;
    }
    const cx_tensor *id=NULL,*logit=NULL,*embedding=NULL,*blank=NULL;
    if(native_reader_file){
      if(cs_eg2_reader_run(&native,soft->v,valid,projected,native_blank)){
        fprintf(stderr,"eg2-asr: native Reader computation failed\n");goto cleanup;
      }
    }else{
      if(cx_session_feed(&reader,"soft_tokens",sd,3,1,soft->v,soft->n)||
         cx_session_feed(&reader,"token_mask",td,2,1,token_mask,750)||
         cx_session_run(&reader)){
        fprintf(stderr,"eg2-asr: word reader ONNX inference failed: %s\n",reader.graph.error);goto cleanup;
      }
      id=q8_path?NULL:cx_session_output(&reader,"top1_index");
      logit=q8_path?NULL:cx_session_output(&reader,"top1_logit");
      embedding=q8_path?cx_session_output(&reader,"projected_tokens"):NULL;
      blank=cx_session_output(&reader,"blank_logit");
      if(!blank||blank->n<valid||
         (q8_path?(!embedding||embedding->n<(uint64_t)valid*CS_EG2_TEXT_DIM):
                  (!id||!logit||id->n<valid||logit->n<valid))){
        fprintf(stderr,"eg2-asr: missing reader scores or projected tokens\n");goto cleanup;
      }
    }
    int last=(start+ns==wave.count);
    size_t begin=window?CS_EG2_OVERLAP_TOKENS:0;
    size_t end=last?valid:(valid?valid-1:0);
    if(begin>end)begin=end;
    for(size_t t=begin;t<end;t++){
      if(count>=scorecap)goto cleanup;
      if(q8_path){
        if(cs_eg2_nearest(&q8,native_reader_file?(projected+t*CS_EG2_TEXT_DIM):
                          (embedding->v+t*CS_EG2_TEXT_DIM),
                          logit_scale,native_reader_file?native_blank[t]:blank->v[t],&scores[count])){
          fprintf(stderr,"eg2-asr: native q8 cosine retrieval failed\n");goto cleanup;
        }
      }else{
        float raw_id=id->v[t];
        if(!isfinite(raw_id)||raw_id<0||raw_id>2147483647.0f||
           floorf(raw_id)!=raw_id||!isfinite(logit->v[t])||!isfinite(blank->v[t])||
           (size_t)raw_id>=vocab.count){
          fprintf(stderr,"eg2-asr: out-of-range or invalid vocabulary scores\n");goto cleanup;
        }
        scores[count]=(cs_eg2_score){(int32_t)raw_id,logit->v[t],blank->v[t]};
      }
      count++;
    }
    if(last)break;
    start+=(size_t)CS_EG2_STEP_TOKENS*640;
  }
  if(save_scores&&cs_eg2_write_scores(save_scores,scores,count))goto cleanup;
  ret=cs_eg2_ctc_decode_lang(scores,count,&vocab,out,capacity,japanese);
cleanup:
  cs_eg2_q8_free(&q8);cs_eg2_reader_free(&native);
  free(projected);free(native_blank);free(scores);free(feat);free(front);free(audio);
  cx_session_close(&encoder);cx_session_close(&reader);
  cs_wave_free(&wave);cs_eg2_vocab_free(&vocab);
  return ret;
}

/* Keep prior ONNX/Reader C interface source-compatible. */
static inline int cx_eg2_transcribe_wav_ext(const char*encoder_file,const char*reader_file,
      const char*vocab_file,const char*wav_file,char*out,size_t capacity,
      const char*save_scores,const char*q8_path,float logit_scale,
      const char*native_reader_file){
  return cx_eg2_transcribe_wav_ext_lang(encoder_file,reader_file,vocab_file,wav_file,
      out,capacity,save_scores,q8_path,logit_scale,native_reader_file,0);
}

/* Backward-compatible Stage27 entry point. */
static inline int cx_eg2_transcribe_wav(const char*encoder_file,const char*reader_file,
      const char*vocab_file,const char*wav_file,char*out,size_t capacity,
      const char*save_scores,const char*q8_path,float logit_scale){
  return cx_eg2_transcribe_wav_ext(encoder_file,reader_file,vocab_file,wav_file,
           out,capacity,save_scores,q8_path,logit_scale,NULL);
}

/* ========================================================================
 * Stage29: Gemma4 Audio's REAL convolutional subsampler in native C.
 * This covers both stride-2 Conv2D, channelwise LayerNorm+ReLU and
 * input_proj_linear. It outputs [ceil(T/4),1024] hidden states, NOT the
 * final [T,512] soft tokens (which require the 12-layer Conformer).
 * The weights are exported from actual safetensors using
 * convert_eg2_subsample.py (conversion-time only).
 * ======================================================================== */
#define CS_EG2_SC0 128
#define CS_EG2_SC1 32
#define CS_EG2_SF0 64
#define CS_EG2_SF1 32
#define CS_EG2_SH 1024
#define CS_EG2_SC_FLOATS ((size_t)CS_EG2_SC0*9 + CS_EG2_SC0 + \
    (size_t)CS_EG2_SC1*CS_EG2_SC0*9 + CS_EG2_SC1 + \
    (size_t)CS_EG2_SH*CS_EG2_SH)
typedef struct {
    float *weights, *w0, *norm0, *w1, *norm1, *proj;
} cs_eg2_subsample;
static inline void cs_eg2_subsample_free(cs_eg2_subsample*m){
    if(m){CS_GPU_FORGET();free(m->weights);memset(m,0,sizeof(*m));}
}
static inline int cs_eg2_subsample_load(cs_eg2_subsample*m,const char*file){
    if(!m||!file)return -1;
    memset(m,0,sizeof(*m));
    FILE*f=cs_eg2_file_open(file);if(!f)return -1;
    unsigned char h[20];int rc=-1;
    if(fread(h,1,sizeof(h),f)!=sizeof(h) || memcmp(h,"EG2C",4) ||
       cs_u32(h+4)!=1 || cs_u32(h+8)!=CS_EG2_SC0 ||
       cs_u32(h+12)!=CS_EG2_SC1 || cs_u32(h+16)!=CS_EG2_SH)goto done;
    m->weights=malloc(CS_EG2_SC_FLOATS*sizeof(float));if(!m->weights)goto done;
    if(fread(m->weights,sizeof(float),CS_EG2_SC_FLOATS,f)!=CS_EG2_SC_FLOATS ||
       fgetc(f)!=EOF || ferror(f))goto done;
    for(size_t i=0;i<CS_EG2_SC_FLOATS;i++)if(!isfinite(m->weights[i]))goto done;
    float *p=m->weights;
    m->w0=p;p+=(size_t)CS_EG2_SC0*9;
    m->norm0=p;p+=CS_EG2_SC0;
    m->w1=p;p+=(size_t)CS_EG2_SC1*CS_EG2_SC0*9;
    m->norm1=p;p+=CS_EG2_SC1;
    m->proj=p;
    rc=0;
done:
    fclose(f);
    if(rc)cs_eg2_subsample_free(m);
    return rc;
}
/* Input layout: [t,frequency,channel], as in PyTorch's permute(N,H,W,C).
 * The input frames >= valid_frames are implicitly masked to zero.
 * layer1 receives an additional validity mask of ceil(valid_frames/2).
 * Output padding locations are not meaningful unless a complete
 * 3000-frame bucket is used. */
static inline int cs_eg2_subsample_run(const cs_eg2_subsample*m,
   const float*mel,size_t nframes,size_t valid_frames,
   float*out,size_t outcap,size_t*ntokens){
    if(!m||!m->weights||!mel||!out||!ntokens||!nframes||
       nframes>CS_EG2_BUCKET||valid_frames>nframes)return -1;
    const size_t t0=(nframes+1)/2,t1=(t0+1)/2;
    if(outcap<t1*CS_EG2_SH)return -1;
    const size_t count0=t0*CS_EG2_SF0*CS_EG2_SC0;
    const size_t count1=t1*CS_EG2_SF1*CS_EG2_SC1;
    float *a=(float*)calloc(count0,sizeof(float));
    float *b=(float*)calloc(count1,sizeof(float));
    if(!a||!b){free(a);free(b);return -1;}
    /* Conv2d layer0: [1,T,128] -> [128,ceil(T/2),64]. */
#pragma omp parallel for schedule(static)
    for(long tt=0;tt<(long)t0;tt++)for(size_t f=0;f<CS_EG2_SF0;f++){
        const size_t t=(size_t)tt;
        float *y=a+(t*CS_EG2_SF0+f)*CS_EG2_SC0;
        for(size_t c=0;c<CS_EG2_SC0;c++){
            const float *w=m->w0+c*9;double sum=0;
            for(int dt=-1;dt<=1;dt++){
                int64_t st=(int64_t)t*2+dt;
                if(st<0||(uint64_t)st>=valid_frames)continue;
                for(int df=-1;df<=1;df++){
                    int sf=(int)f*2+df;
                    if(sf>=0&&sf<CS_EG2_MEL)
                        sum+=(double)mel[(size_t)st*CS_EG2_MEL+(size_t)sf]*w[(dt+1)*3+(df+1)];
                }
            }
            y[c]=(float)sum;
        }
        double mean=0,var=0;
        for(int c=0;c<CS_EG2_SC0;c++)mean+=y[c];mean/=CS_EG2_SC0;
        for(int c=0;c<CS_EG2_SC0;c++){
            double z=y[c]-mean;var+=z*z;
        }
        float inv=(float)(1.0/sqrt(var/CS_EG2_SC0+1e-6));
        for(int c=0;c<CS_EG2_SC0;c++){
            float v=(y[c]-(float)mean)*inv*m->norm0[c];y[c]=v>0?v:0;
        }
    }
    /* Conv2d layer1: [128,t0,64] -> [32,t1,32]. Mask t0 first. */
    const size_t valid0=(valid_frames+1)/2;
#pragma omp parallel for schedule(static)
    for(long tt=0;tt<(long)t1;tt++)for(size_t f=0;f<CS_EG2_SF1;f++){
        const size_t t=(size_t)tt;
        float*y=b+(t*CS_EG2_SF1+f)*CS_EG2_SC1;
        for(size_t c=0;c<CS_EG2_SC1;c++){
            const float *w=m->w1+c*CS_EG2_SC0*9;
            double sum=0;
            for(size_t ci=0;ci<CS_EG2_SC0;ci++)for(int dt=-1;dt<=1;dt++){
                int64_t st=(int64_t)t*2+dt;
                if(st<0||(uint64_t)st>=valid0)continue;
                for(int df=-1;df<=1;df++){
                    int sf=(int)f*2+df;
                    if(sf>=0&&sf<CS_EG2_SF0)
                        sum+=(double)a[((size_t)st*CS_EG2_SF0+(size_t)sf)*CS_EG2_SC0+ci]*
                        w[ci*9+(dt+1)*3+(df+1)];
                }
            }
            y[c]=(float)sum;
        }
        double mean=0,var=0;
        for(int c=0;c<CS_EG2_SC1;c++)mean+=y[c];mean/=CS_EG2_SC1;
        for(int c=0;c<CS_EG2_SC1;c++){
            double z=y[c]-mean;var+=z*z;
        }
        float inv=(float)(1.0/sqrt(var/CS_EG2_SC1+1e-6));
        for(int c=0;c<CS_EG2_SC1;c++){
            float v=(y[c]-(float)mean)*inv*m->norm1[c];y[c]=v>0?v:0;
        }
    }
    /* [t1,32,32] in frequency-major order -> Linear [1024,1024]. */
    cs_eg2_gemm_nt(out,b,m->proj,t1,CS_EG2_SH,CS_EG2_SH,NULL);
    free(a);free(b);*ntokens=t1;
    return 0;
}
/* Diagnostic hidden-state format: EG2H version 1, tokens, width 1024. */
static inline int cs_eg2_hidden_write(const char*path,const float*hidden,size_t n){
    if(!path||!hidden||!n||n>CS_EG2_TOKENS)return -1;
    FILE*f=fopen(path,"wb");if(!f)return -1;
    unsigned char h[16]={0};memcpy(h,"EG2H",4);
    cs_put32(h+4,1);cs_put32(h+8,(uint32_t)n);cs_put32(h+12,CS_EG2_SH);
    int rc=fwrite(h,1,sizeof(h),f)==sizeof(h)&&
           fwrite(hidden,sizeof(float),n*CS_EG2_SH,f)==n*CS_EG2_SH?0:-1;
    if(fclose(f))rc=-1;
    if(rc)remove(path);
    return rc;
}

/* ========================================================================
 * Stage 30: native Gemma4 / EmbeddingGemma2 audio Conformer layer.
 * EG2L v1 stores one complete layer; 12 layers may be loaded sequentially.
 * EG2L v2 is identical except that the large matrices (see
 * cs_eg2_layer_tensor_half) are IEEE half precision on disk; they are widened
 * to float32 when loaded, so inference arithmetic is unchanged.
 * Implements feedforward, 12-token blocked relative-position attention,
 * causal GLU depthwise convolution and final RMS norm without ONNX.
 * This is an unoptimized FP32 reference interpreter.  Reader integration
 * still requires audio_tower.output_proj + embed_audio.
 * ======================================================================== */
#define CS_EG2_LAYER_TENSORS 22
#define CS_EG2_LAYER_CLIPS 10

typedef struct {
    unsigned d,ff,heads,kernel;
    float *all, *p[CS_EG2_LAYER_TENSORS];
    float clip[CS_EG2_LAYER_CLIPS][4]; /* in_min,in_max,out_min,out_max */
} cs_eg2_layer;

/* Canonical EG2L tensor order; each is indexed by the constants below. */
enum { EG2_F1_PRE,EG2_F1_UP,EG2_F1_DOWN,EG2_F1_POST,
       EG2_NORM_PRE,EG2_AQ,EG2_AK,EG2_AV,EG2_AO,EG2_AR,EG2_ASCALE,
       EG2_NORM_POST,EG2_CPRE,EG2_CUP,EG2_CDW,EG2_CNORM,EG2_CDOWN,
       EG2_F2_PRE,EG2_F2_UP,EG2_F2_DOWN,EG2_F2_POST,EG2_NORM_OUT };
static inline size_t cs_eg2_layer_tensor_count(unsigned d,unsigned ff,unsigned h,unsigned k,int id){
    switch(id){
    case EG2_F1_PRE:case EG2_F1_POST:case EG2_NORM_PRE:
    case EG2_NORM_POST:case EG2_CPRE:case EG2_CNORM:
    case EG2_F2_PRE:case EG2_F2_POST:case EG2_NORM_OUT:return d;
    case EG2_F1_UP:case EG2_F2_UP:return (size_t)ff*d;
    case EG2_F1_DOWN:case EG2_F2_DOWN:return (size_t)d*ff;
    case EG2_AQ:case EG2_AK:case EG2_AV:case EG2_AO:case EG2_AR:case EG2_CDOWN:return (size_t)d*d;
    case EG2_ASCALE:return d/h;
    case EG2_CUP:return (size_t)d*2*d;
    case EG2_CDW:return (size_t)d*k;
    default:return 0;
    }
}
/* Threads to use when the user gave neither -j nor OMP_NUM_THREADS: one per
 * physical core.  Hyper-threads share the FP units, so using them adds no
 * GEMM throughput and only adds barrier cost.  Returns 0 = leave default. */
static inline int cs_eg2_physical_threads(void){
#ifdef _OPENMP
    FILE*f=fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list","r");
    if(!f)return 0;
    char line[256];int siblings=0;
    if(fgets(line,sizeof(line),f)){
        for(const char*p=line;*p&&*p!='\n';){
            char*end;long a=strtol(p,&end,10);if(end==p)break;
            long b=a;if(*end=='-'){p=end+1;b=strtol(p,&end,10);}
            siblings+=(int)(b-a+1);
            p=*end==','?end+1:end;if(*end!=','&&*end!='-')break;
        }
    }
    fclose(f);
    int procs=omp_get_num_procs();
    return siblings>1&&procs>=siblings?procs/siblings:0;
#else
    return 0;
#endif
}
/* Tensors stored as float16 in EG2L v2: every large matrix. Norm gains,
 * the depthwise kernel and the clip bounds stay float32. */
static inline int cs_eg2_layer_tensor_half(int id){
    switch(id){
    case EG2_F1_UP:case EG2_F1_DOWN:case EG2_AQ:case EG2_AK:case EG2_AV:
    case EG2_AO:case EG2_AR:case EG2_CUP:case EG2_CDOWN:
    case EG2_F2_UP:case EG2_F2_DOWN:return 1;
    default:return 0;
    }
}
static inline float cs_eg2_half_to_float(unsigned h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16,exp=(h>>10)&0x1fu,man=h&0x3ffu,bits;
    if(!exp){
        if(!man)bits=sign;
        else{int e=-1;do{e++;man<<=1;}while(!(man&0x400u));
             man&=0x3ffu;bits=sign|((uint32_t)(112-e)<<23)|(man<<13);}
    }else if(exp==31)bits=sign|0x7f800000u|(man<<13);
    else bits=sign|((exp+112u)<<23)|(man<<13);
    float f;memcpy(&f,&bits,4);return f;
}
/* Read count little-endian half values from f, widening into dst. */
static inline int cs_eg2_read_half(FILE*f,float*dst,size_t count){
    enum{CHUNK=1<<22};
    unsigned char*buf=(unsigned char*)malloc((size_t)CHUNK*2);if(!buf)return -1;
    int rc=0;
    while(count&&!rc){
        size_t chunk=count<CHUNK?count:(size_t)CHUNK;
        if(fread(buf,2,chunk,f)!=chunk){rc=-1;break;}
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)chunk;i++)dst[i]=cs_eg2_half_to_float(cs_u16(buf+2*i));
        dst+=chunk;count-=chunk;
    }
    free(buf);return rc;
}
static inline void cs_eg2_layer_free(cs_eg2_layer*m){if(m){CS_GPU_FORGET();free(m->all);memset(m,0,sizeof(*m));}}
static inline int cs_eg2_layer_load(cs_eg2_layer*m,const char*path){
    if(!m||!path)return -1;
    memset(m,0,sizeof(*m));FILE*f=cs_eg2_file_open(path);if(!f)return -1;
    unsigned char h[28];int rc=-1;
    if(fread(h,1,sizeof(h),f)!=sizeof(h)||memcmp(h,"EG2L",4)||
       (cs_u32(h+4)!=1&&cs_u32(h+4)!=2))goto done;
    const int half=cs_u32(h+4)==2;
    m->d=cs_u32(h+8);m->ff=cs_u32(h+12);m->heads=cs_u32(h+16);m->kernel=cs_u32(h+20);
    if(!m->d||m->d>1024||!m->ff||m->ff>4096||!m->heads||m->d%m->heads||
       !m->kernel||m->kernel>15||!(m->kernel&1)||cs_u32(h+24)!=CS_EG2_LAYER_CLIPS)goto done;
    size_t count=0,bytes=0;
    for(int i=0;i<CS_EG2_LAYER_TENSORS;i++){
        size_t n=cs_eg2_layer_tensor_count(m->d,m->ff,m->heads,m->kernel,i);
        count+=n;bytes+=n*(half&&cs_eg2_layer_tensor_half(i)?2:4);
    }
    /* Validate the complete byte length before allocating/reading. */
    if(fseek(f,0,SEEK_END)||ftell(f)!=(long)(sizeof(h)+bytes+CS_EG2_LAYER_CLIPS*4*sizeof(float))||
       fseek(f,sizeof(h),SEEK_SET))goto done;
    m->all=(float*)malloc(count*sizeof(float));if(!m->all)goto done;
    {float*p=m->all;
     for(int i=0;i<CS_EG2_LAYER_TENSORS;i++){
        size_t n=cs_eg2_layer_tensor_count(m->d,m->ff,m->heads,m->kernel,i);
        m->p[i]=p;
        if(half&&cs_eg2_layer_tensor_half(i)){if(cs_eg2_read_half(f,p,n))goto done;}
        else if(fread(p,sizeof(float),n,f)!=n)goto done;
        p+=n;
     }}
    if(fread(m->clip,sizeof(float),CS_EG2_LAYER_CLIPS*4,f)!=CS_EG2_LAYER_CLIPS*4)goto done;
    {int bad=0;
#pragma omp parallel for schedule(static) reduction(|:bad)
     for(long i=0;i<(long)count;i++)bad|=!isfinite(m->all[i]);
     if(bad)goto done;}
    for(int i=0;i<CS_EG2_LAYER_CLIPS;i++){
        for(int j=0;j<4;j++)if(isnan(m->clip[i][j]))goto done;
        if(m->clip[i][0]>m->clip[i][1]||m->clip[i][2]>m->clip[i][3])goto done;
    }
    rc=0;
done:
    fclose(f);if(rc)cs_eg2_layer_free(m);return rc;
}
static inline float cs_eg2_silu(float x){return x/(1.0f+expf(-x));}
static inline float cs_eg2_softplus(float x){return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x)));}
static inline float cs_eg2_clamp(float x,float low,float high){return fminf(high,fmaxf(low,x));}
static inline void cs_eg2_rms(float*dst,const float*src,const float*weight,unsigned d){
    double sum=0;for(unsigned i=0;i<d;i++)sum+=(double)src[i]*src[i];
    float scale=1.0f/sqrtf((float)(sum/d)+1.e-6f);
    for(unsigned i=0;i<d;i++)dst[i]=src[i]*scale*weight[i];
}
static inline void cs_eg2_linear(float*dst,const float*src,const float*w,
         unsigned in,unsigned out,const float clip[4]){
    /* Clamp once per input vector, not once per output row (up to 4096). */
    float bounded[4096];const float*x=src;
    if(in<=4096 && (isfinite(clip[0])||isfinite(clip[1]))){
        for(unsigned i=0;i<in;i++)bounded[i]=cs_eg2_clamp(src[i],clip[0],clip[1]);
        x=bounded;
    }
    for(unsigned o=0;o<out;o++){
        const float*row=w+(size_t)o*in;
        float acc;
        if(x==src && (isfinite(clip[0])||isfinite(clip[1]))){
            double sum=0;for(unsigned i=0;i<in;i++)
                sum+=(double)cs_eg2_clamp(src[i],clip[0],clip[1])*row[i];
            acc=(float)sum;
        }else acc=cs_eg2_dot_f32(x,row,in);
        dst[o]=cs_eg2_clamp(acc,clip[2],clip[3]);
    }
}
/* All layer stages below process every token at once so that each weight
 * matrix is read once per stage (see cs_eg2_gemm_nt), with the per-token
 * loops spread over OpenMP threads.  The arithmetic is unchanged. */
static inline int cs_eg2_linear_batch(float*dst,const float*src,const float*w,
         unsigned in,unsigned out,size_t n,const float clip[4]){
    const float*x=src;float*bounded=NULL;
    if(isfinite(clip[0])||isfinite(clip[1])){
        bounded=(float*)malloc(n*in*sizeof(float));if(!bounded)return -1;
#pragma omp parallel for schedule(static)
        for(long i=0;i<(long)(n*in);i++)bounded[i]=cs_eg2_clamp(src[i],clip[0],clip[1]);
        x=bounded;
    }
    cs_eg2_gemm_nt(dst,x,w,n,in,out,NULL);
    free(bounded);
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)(n*out);i++)dst[i]=cs_eg2_clamp(dst[i],clip[2],clip[3]);
    return 0;
}
/* In place macaron FFN for n tokens, ffno=0 or 1. */
static inline int cs_eg2_layer_ff(const cs_eg2_layer*m,float*x,size_t n,int ffno){
    const unsigned d=m->d,ff=m->ff;const int base=ffno?EG2_F2_PRE:EG2_F1_PRE,clip=ffno?8:0;
    float *norm=malloc(n*d*sizeof(float)),*mid=malloc(n*ff*sizeof(float)),
          *out=malloc(n*d*sizeof(float));
    int rc=-1;
    if(!norm||!mid||!out)goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++)cs_eg2_rms(norm+i*d,x+i*d,m->p[base],d);
    if(cs_eg2_linear_batch(mid,norm,m->p[base+1],d,ff,n,m->clip[clip]))goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)(n*ff);i++)mid[i]=cs_eg2_silu(mid[i]);
    if(cs_eg2_linear_batch(out,mid,m->p[base+2],ff,d,n,m->clip[clip+1]))goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++){
        cs_eg2_rms(norm+i*d,out+i*d,m->p[base+3],d);
        for(unsigned j=0;j<d;j++)x[i*d+j]+=.5f*norm[i*d+j];
    }
    rc=0;
done:
    free(norm);free(mid);free(out);return rc;
}
/* 25-token local context: 12-token chunk + 12 preceding positions.
 * Attention mask additionally disallows future keys and keys >11 behind.
 * Relative keys are generated once per layer (13 x d), using the official
 * sin/cos concatenation and `_rel_shift` mapping: relative[j-q]. */
static inline int cs_eg2_layer_attention(const cs_eg2_layer*m,float*x,size_t t,size_t valid){
    const unsigned d=m->d,h=m->heads,hd=d/h;
    const float log2e=1.4426950408889634f;
    float *norm=malloc(t*d*sizeof(float)),*q=malloc(t*d*sizeof(float)),
          *k=malloc(t*d*sizeof(float)),*v=malloc(t*d*sizeof(float)),
          *r=malloc(13*d*sizeof(float)),*pe=malloc(13*d*sizeof(float)),
          *att=calloc(t*d,sizeof(float));
    int rc=-1;
    if(!norm||!q||!k||!v||!r||!pe||!att)goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)t;i++)cs_eg2_rms(norm+i*d,x+i*d,m->p[EG2_NORM_PRE],d);
    if(cs_eg2_linear_batch(q,norm,m->p[EG2_AQ],d,d,t,m->clip[2])||
       cs_eg2_linear_batch(k,norm,m->p[EG2_AK],d,d,t,m->clip[3])||
       cs_eg2_linear_batch(v,norm,m->p[EG2_AV],d,d,t,m->clip[4]))goto done;
    for(unsigned j=0;j<13;j++){
        /* Position embeddings count DOWN from 12 to 0. */
        for(unsigned c=0;c<d/2;c++){
            float angle=(float)(12-j)*expf(-(float)c*logf(10000.f)/(float)((d/2)>1?(d/2-1):1));
            pe[(size_t)j*d+c]=sinf(angle);pe[(size_t)j*d+c+d/2]=cosf(angle);
        }
    }
    /* The relative-key projection is unclipped. */
#pragma omp parallel for schedule(static)
    for(long idx=0;idx<(long)(13*d);idx++){
        size_t j=(size_t)idx/d,o=(size_t)idx%d;
        double sum=0;for(unsigned n=0;n<d;n++)sum+=(double)m->p[EG2_AR][o*d+n]*pe[j*d+n];
        r[idx]=(float)sum;
    }
    float sp[1024];  /* softplus of the learned per-channel query scale */
    for(unsigned c=0;c<hd;c++)sp[c]=cs_eg2_softplus(m->p[EG2_ASCALE][c]);
#pragma omp parallel for schedule(static)
    for(long ii=0;ii<(long)t;ii++){
        const size_t i=(size_t)ii;
        for(unsigned head=0;head<h;head++){
        /* Softmax over the local context. Non-future 12-position span. */
        float logits[24],mx=-INFINITY;int pos[24];
        size_t block=i/12;int qi=(int)(i%12);
        const float qscale=1.f/sqrtf((float)hd)*log2e;
        const float kscale=log2f(1.0f+expf(1.f));
        float qsv[1024];
        for(unsigned c=0;c<hd;c++)qsv[c]=q[i*d+head*hd+c]*qscale*sp[c];
        for(int j=0;j<24;j++){
            int key=(int)(block*12)+j-12;
            pos[j]=key;
            if(key<0||(size_t)key>=valid||key>(int)i||(int)i-key>=12){logits[j]=-1e9f;continue;}
            double ac=0,bd=0;
            for(unsigned c=0;c<hd;c++)
                ac+=(double)qsv[c]*k[(size_t)key*d+head*hd+c]*kscale;
            if(j>=qi && j-qi<=12)
                for(unsigned c=0;c<hd;c++)
                    bd+=(double)qsv[c]*r[(size_t)(j-qi)*d+head*hd+c];
            float z=(float)(ac+bd);
            logits[j]=50.f*tanhf(z/50.f);
            if(logits[j]>mx)mx=logits[j];
        }
        if(!isfinite(mx))continue;
        double denominator=0;float prob[24];
        for(int j=0;j<24;j++){
            prob[j]=logits[j]<=-1e8f?0.f:expf(logits[j]-mx);
            denominator+=prob[j];
        }
        for(int j=0;j<24;j++)if(prob[j]>0){
            float a=(float)(prob[j]/denominator);
            for(unsigned c=0;c<hd;c++)att[i*d+head*hd+c]+=a*v[(size_t)pos[j]*d+head*hd+c];
        }
        }
    }
    /* Output projection into q (no longer needed), then the
     * pre-attention residual + norm_post_attn. */
    if(cs_eg2_linear_batch(q,att,m->p[EG2_AO],d,d,t,m->clip[5]))goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)t;i++){
        cs_eg2_rms(norm+i*d,q+i*d,m->p[EG2_NORM_POST],d);
        for(unsigned j=0;j<d;j++)x[i*d+j]+=norm[i*d+j];
    }
    rc=0;
done:
    free(norm);free(q);free(k);free(v);free(r);free(pe);free(att);return rc;
}
static inline int cs_eg2_layer_conv(const cs_eg2_layer*m,float*x,size_t t){
    const unsigned d=m->d;
    float *norm=malloc(t*d*sizeof(float)),*tmp=malloc(t*2*d*sizeof(float)),
          *a=malloc(t*d*sizeof(float)),*z=malloc(t*d*sizeof(float));
    int rc=-1;
    if(!norm||!tmp||!a||!z)goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)t;i++)cs_eg2_rms(norm+i*d,x+i*d,m->p[EG2_CPRE],d);
    if(cs_eg2_linear_batch(tmp,norm,m->p[EG2_CUP],d,2*d,t,m->clip[6]))goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)t;i++)for(unsigned c=0;c<d;c++)
        a[i*d+c]=tmp[(size_t)i*2*d+c]/(1.f+expf(-tmp[(size_t)i*2*d+d+c]));
#pragma omp parallel for schedule(static)
    for(long ii=0;ii<(long)t;ii++){
        const size_t i=(size_t)ii;
        for(unsigned c=0;c<d;c++){
            double sum=0;
            for(unsigned j=0;j<m->kernel;j++){
                int64_t pos=(int64_t)i-(int64_t)(m->kernel-1-j);
                if(pos>=0)sum+=(double)a[(size_t)pos*d+c]*m->p[EG2_CDW][(size_t)c*m->kernel+j];
            }
            z[i*d+c]=(float)sum;
        }
    }
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)t;i++){
        cs_eg2_rms(norm+i*d,z+i*d,m->p[EG2_CNORM],d);
        for(unsigned c=0;c<d;c++)norm[i*d+c]=cs_eg2_silu(norm[i*d+c]);
    }
    if(cs_eg2_linear_batch(a,norm,m->p[EG2_CDOWN],d,d,t,m->clip[7]))goto done;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)(t*d);i++)x[i]+=a[i];
    rc=0;
done:
    free(norm);free(tmp);free(a);free(z);return rc;
}
static inline int cs_eg2_layer_run(const cs_eg2_layer*m,float*x,size_t n,size_t valid){
    if(!m||!m->all||!x||!n||n>750||valid>n)return -1;
    const size_t d=m->d;
    if(cs_eg2_layer_ff(m,x,n,0)||cs_eg2_layer_attention(m,x,n,valid)||
       cs_eg2_layer_conv(m,x,n)||cs_eg2_layer_ff(m,x,n,1))return -1;
#pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++){
        float row[1024];
        cs_eg2_rms(row,x+i*d,m->p[EG2_NORM_OUT],(unsigned)d);
        memcpy(x+i*d,row,d*sizeof(float));
    }
    return 0;
}
static inline int cs_eg2_hidden_read(const char*path,float**hidden,size_t*count){
    if(!path||!hidden||!count)return -1;*hidden=NULL;*count=0;
    FILE*f=cs_eg2_file_open(path);if(!f)return -1;
    unsigned char h[16];int rc=-1;
    if(fread(h,1,16,f)!=16||memcmp(h,"EG2H",4)||cs_u32(h+4)!=1||
       !cs_u32(h+8)||cs_u32(h+8)>750||cs_u32(h+12)!=1024)goto done;
    size_t t=cs_u32(h+8);float *v=malloc(t*1024*sizeof(float));if(!v)goto done;
    if(fread(v,sizeof(float),t*1024,f)!=t*1024||fgetc(f)!=EOF){free(v);goto done;}
    for(size_t i=0;i<t*1024;i++)if(!isfinite(v[i])){free(v);goto done;}
    *count=t;*hidden=v;rc=0;
done:fclose(f);return rc;
}

/* Stage 30: 1024 -> 1536(+bias) -> RMSNorm(no affine) -> 512. */
#define CS_EG2_PROJECT_FLOATS ((size_t)1536*1024+1536+(size_t)512*1536)
typedef struct {float *data,*tower,*bias,*embedding;} cs_eg2_projection;
static inline void cs_eg2_projection_free(cs_eg2_projection*m){if(m){CS_GPU_FORGET();free(m->data);memset(m,0,sizeof(*m));}}
static inline int cs_eg2_projection_load(cs_eg2_projection*m,const char*path){
 if(!m||!path)return -1;memset(m,0,sizeof(*m));
 FILE*f=cs_eg2_file_open(path);if(!f)return -1;int rc=-1;
 unsigned char h[20];
 if(fread(h,1,20,f)!=20||memcmp(h,"EG2P",4)||cs_u32(h+4)!=1||
    cs_u32(h+8)!=1024||cs_u32(h+12)!=1536||cs_u32(h+16)!=512)goto done;
 if(fseek(f,0,SEEK_END)||ftell(f)!=(long)(20+4*CS_EG2_PROJECT_FLOATS)||fseek(f,20,SEEK_SET))goto done;
 m->data=malloc(CS_EG2_PROJECT_FLOATS*sizeof(float));if(!m->data)goto done;
 if(fread(m->data,sizeof(float),CS_EG2_PROJECT_FLOATS,f)!=CS_EG2_PROJECT_FLOATS)goto done;
 for(size_t i=0;i<CS_EG2_PROJECT_FLOATS;i++)if(!isfinite(m->data[i]))goto done;
 m->tower=m->data;m->bias=m->tower+1536*1024;m->embedding=m->bias+1536;rc=0;
done:fclose(f);if(rc)cs_eg2_projection_free(m);return rc;
}
static inline int cs_eg2_projection_run(const cs_eg2_projection*m,
              const float*hidden,size_t n,float*soft){
 if(!m||!m->data||!hidden||!soft||!n||n>750)return -1;
 float *mid=malloc(n*1536*sizeof(float));if(!mid)return -1;
 cs_eg2_gemm_nt(mid,hidden,m->tower,n,1024,1536,m->bias);
#pragma omp parallel for schedule(static)
 for(long t=0;t<(long)n;t++){
    float*row=mid+(size_t)t*1536;
    double sumsq=0;
    for(size_t o=0;o<1536;o++)sumsq+=(double)row[o]*row[o];
    float scale=1.0f/sqrtf((float)(sumsq/1536)+1.e-6f);
    for(size_t o=0;o<1536;o++)row[o]*=scale;
 }
 cs_eg2_gemm_nt(soft,mid,m->embedding,n,1536,512,NULL);
 free(mid);return 0;
}


/* ========================================================================
 * Stage 31: long-wave native audio tower -> Reader -> INT8 cosine -> CTC.
 * Keep fixed 30-second overlapping windows; process up to 8 windows per
 * layer load so long WAVs do not re-read a 100 MB layer for every window.
 * No model weights or runtime dependencies are bundled.
 * ======================================================================== */
#define CS_EG2_BATCH_WINDOWS 8
typedef struct {
    size_t start,samples,tokens,begin,end;
} cs_eg2_window;
static inline int cs_eg2_window_at(size_t total,size_t index,cs_eg2_window*w){
    if(!w||index>SIZE_MAX/((size_t)CS_EG2_STEP_TOKENS*640))return -1;
    if(!total)return 0;
    const size_t step=(size_t)CS_EG2_STEP_TOKENS*640;
    if(index){
        size_t previous=(index-1)*step;
        if(previous>=total||total-previous<=480000)return 0;
    }
    size_t start=index*step;
    if(start>=total)return 0;
    size_t samples=total-start;if(samples>480000)samples=480000;
    size_t frames=samples>=161?1+(samples-161)/160:0;
    size_t tokens=(frames+3)/4;
    if(!tokens)return -1;
    int last=start+samples==total;
    size_t begin=index?CS_EG2_OVERLAP_TOKENS:0;
    size_t end=last?tokens:tokens-1;
    if(begin>end)begin=end;
    *w=(cs_eg2_window){start,samples,tokens,begin,end};
    return 1;
}
static inline int cs_eg2_native_long_wav(const char*model_dir,const char*reader_file,
        const char*q8_file,const char*vocab_file,const char*wav_file,
        char*out,size_t cap,const char*score_file,int japanese){
    if(!model_dir||!reader_file||!q8_file||!vocab_file||!wav_file||!out||!cap)return -1;
    out[0]=0;
    cs_wave_stream wave={0};float *window_pcm=NULL;
    cs_eg2_frontend*front=NULL;cs_eg2_subsample sub={0};
    cs_eg2_projection proj={0};cs_eg2_reader reader={0};
    cs_eg2_q8 table={0};cs_eg2_vocab vocab={0};
    float*mel=NULL,*soft=NULL,*projected=NULL,*blank=NULL;
    cs_eg2_score *scores=NULL;size_t used=0,scorecap=0;
    float*states[CS_EG2_BATCH_WINDOWS]={0};
    cs_eg2_window windows[CS_EG2_BATCH_WINDOWS];
    char path[4096];int rc=-1;
    if(cs_wave_stream_open(&wave,wav_file))goto done;
    size_t total=cs_wave_stream_length16(&wave);
    if(total<320||total>SIZE_MAX-480000)goto done;
    if(cs_eg2_vocab_load(&vocab,vocab_file)||cs_eg2_q8_open(&table,q8_file)||
       table.count!=vocab.count||cs_eg2_reader_load(&reader,reader_file))goto done;
    if(snprintf(path,sizeof(path),"%s/subsampler.eg2c",model_dir)>=(int)sizeof(path)||
       cs_eg2_subsample_load(&sub,path))goto done;
    if(snprintf(path,sizeof(path),"%s/projection.eg2p",model_dir)>=(int)sizeof(path)||
       cs_eg2_projection_load(&proj,path))goto done;
    front=malloc(sizeof(*front));
    mel=malloc((size_t)CS_EG2_BUCKET*CS_EG2_MEL*sizeof(float));
    soft=malloc((size_t)CS_EG2_TOKENS*512*sizeof(float));
    projected=malloc((size_t)CS_EG2_TOKENS*768*sizeof(float));
    blank=malloc((size_t)CS_EG2_TOKENS*sizeof(float));
    scorecap=(total+639)/640+CS_EG2_TOKENS;
    if(scorecap>SIZE_MAX/sizeof(*scores))goto done;
    scores=malloc(scorecap*sizeof(*scores));
    window_pcm=malloc(480000*sizeof(float));
    if(!front||!mel||!soft||!projected||!blank||!scores||!window_pcm)goto done;
    cs_eg2_frontend_init(front);
    if(snprintf(path,sizeof(path),"%s/mel_filters.bin",model_dir)>=(int)sizeof(path))goto done;
    {FILE *mf=cs_eg2_file_open(path);
     if(mf){fclose(mf);if(cs_eg2_frontend_load_mel(front,path))goto done;}
     else fprintf(stderr,"eg2-native-asr: warning: missing mel_filters.bin; using computed fallback (reference parity unverified)\n");}

    for(size_t first=0;;first+=CS_EG2_BATCH_WINDOWS){
        size_t batch=0;
        for(;batch<CS_EG2_BATCH_WINDOWS;batch++){
            int active=cs_eg2_window_at(total,first+batch,&windows[batch]);
            if(active<0)goto done;
            if(active==0)break;
            size_t nframes=0,tokens=0;
            states[batch]=malloc((size_t)CS_EG2_TOKENS*CS_EG2_SH*sizeof(float));
            if(!states[batch]||cs_wave_stream_read16(&wave,windows[batch].start,
                windows[batch].samples,window_pcm)||
               cs_eg2_features(front,window_pcm,windows[batch].samples,mel,&nframes)||
               cs_eg2_subsample_run(&sub,mel,nframes,nframes,states[batch],
                    (size_t)CS_EG2_TOKENS*CS_EG2_SH,&tokens)||tokens!=windows[batch].tokens)
                goto done;
        }
        if(!batch)break;
        /* Only one full-size weight layer is resident at a time. */
        for(int layer_no=0;layer_no<12;layer_no++){
            if(snprintf(path,sizeof(path),"%s/layer_%02d.eg2l",model_dir,layer_no)>=(int)sizeof(path))goto done;
            cs_eg2_layer layer={0};
            int good=!cs_eg2_layer_load(&layer,path)&&
                     layer.d==1024&&layer.ff==4096&&layer.heads==8&&layer.kernel==5;
            for(size_t j=0;good&&j<batch;j++)
                good=!cs_eg2_layer_run(&layer,states[j],windows[j].tokens,windows[j].tokens);
            cs_eg2_layer_free(&layer);
            if(!good)goto done;
        }
        for(size_t j=0;j<batch;j++){
            size_t t=windows[j].tokens;
            if(cs_eg2_projection_run(&proj,states[j],t,soft)||
               cs_eg2_reader_run(&reader,soft,t,projected,blank))goto done;
            if(windows[j].end>windows[j].begin){
                size_t count=windows[j].end-windows[j].begin;
                if(count>scorecap-used||cs_eg2_nearest_batch(&table,
                        projected+windows[j].begin*768,768,reader.scale,
                        blank+windows[j].begin,count,scores+used))goto done;
                used+=count;
            }
            free(states[j]);states[j]=NULL;
        }
    }
    if(!used||cs_eg2_ctc_decode_lang(scores,used,&vocab,out,cap,japanese)<0||
       (score_file&&cs_eg2_write_scores(score_file,scores,used)))goto done;
    rc=0;
done:
    for(size_t i=0;i<CS_EG2_BATCH_WINDOWS;i++)free(states[i]);
    cs_wave_stream_close(&wave);free(window_pcm);free(front);free(mel);free(soft);
    free(projected);free(blank);free(scores);
    cs_eg2_subsample_free(&sub);cs_eg2_projection_free(&proj);
    cs_eg2_reader_free(&reader);cs_eg2_q8_free(&table);cs_eg2_vocab_free(&vocab);
    if(rc)out[0]=0;
    return rc;
}

#endif /* CATS_RNNT_H */

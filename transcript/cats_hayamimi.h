//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

/*
 * hayamimi minimal C integration (Linux/POSIX).
 * No Python, sherpa headers, or ONNX headers needed at build time.
 * Uses the *existing* sherpa-onnx C runtime at execution via dlopen().
 * ABI definitions below are pinned to sherpa-onnx 1.13.6 (Apache 2.0).
 * Do NOT bypass the ABI version check when using a newer runtime.
 * Supported: offline ASR + energy-gated raw PCM streaming; not the entire
 * hayamimi multilingual inference/routing/post-processing pipeline.
 */
#ifndef CATS_HAYAMIMI_H
#define CATS_HAYAMIMI_H
#if !defined(__linux__) && !defined(__APPLE__)
#error cats_hayamimi.h currently supports Linux/macOS only
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <glob.h>
#include <limits.h>

/* Public config ABI, faithfully mirrored from sherpa-onnx v1.13.6. */
typedef struct { int32_t sample_rate, feature_dim; } hm_feature;
typedef struct { const char *encoder, *decoder, *joiner; } hm_triple;
typedef struct { const char *model; } hm_single;
typedef struct { const char *encoder,*decoder,*language,*task; int32_t tail_paddings,enable_token_timestamps,enable_segment_timestamps; } hm_whisper;
typedef struct { const char *encoder,*decoder,*src_lang,*tgt_lang; int32_t use_pnc; } hm_canary;
typedef struct { const char *encoder,*decoder,*language; int32_t use_punct,use_itn; } hm_cohere;
typedef struct { const char *preprocessor,*encoder,*uncached_decoder,*cached_decoder,*merged_decoder; } hm_moonshine;
typedef struct { const char *model; float scale; } hm_lm;
typedef struct { const char *model,*language; int32_t use_itn; } hm_sensevoice;
typedef struct { const char *encoder_adaptor,*llm,*embedding,*tokenizer,*system_prompt,*user_prompt; int32_t max_new_tokens; float temperature,top_p; int32_t seed; const char *language; int32_t itn; const char *hotwords; } hm_funasr;
typedef struct { const char *conv_frontend,*encoder,*decoder,*tokenizer; int32_t max_total_len,max_new_tokens; float temperature,top_p; int32_t seed; const char *hotwords; } hm_qwen;
/* Keep all ABI fields in exact order; no shared/deduced type shortcuts. */
typedef struct {
  hm_triple transducer;
  hm_single paraformer,nemo_ctc;
  hm_whisper whisper;
  hm_single tdnn;
  const char *tokens;
  int32_t num_threads,debug;
  const char *provider,*model_type,*modeling_unit,*bpe_vocab,*telespeech_ctc;
  hm_sensevoice sense_voice;
  hm_moonshine moonshine;
  struct { const char *encoder,*decoder; } fire_red_asr;
  hm_single dolphin,zipformer_ctc;
  hm_canary canary;
  hm_single wenet_ctc,omnilingual,medasr;
  hm_funasr funasr_nano;
  hm_single fire_red_asr_ctc;
  hm_qwen qwen3_asr;
  hm_cohere cohere_transcribe;
} hm_model;
typedef struct { const char *dict_dir,*lexicon,*rule_fsts; } hm_homophone;
typedef struct {
  hm_feature feat_config;
  hm_model model_config;
  hm_lm lm_config;
  const char *decoding_method;
  int32_t max_active_paths;
  const char *hotwords_file;
  float hotwords_score;
  const char *rule_fsts,*rule_fars;
  float blank_penalty;
  hm_homophone hr;
} hm_config;
typedef struct { const char *text; } hm_result;
typedef struct { const float *samples; int32_t sample_rate,num_samples; } hm_wave;

typedef struct {
  void *handle;
  const char *(*version)(void);
  const void *(*create_recognizer)(const hm_config *);
  void (*destroy_recognizer)(const void *);
  const void *(*create_stream)(const void *);
  void (*destroy_stream)(const void *);
  void (*accept)(const void *,int32_t,const float *,int32_t);
  void (*decode)(const void *,const void *);
  const hm_result *(*get_result)(const void *);
  void (*free_result)(const hm_result *);
  const hm_wave *(*read_wave)(const char *);
  void (*free_wave)(const hm_wave *);
  const void *recognizer;
} hm_api;

/* 0 means success; no optional feature silently selected. */
static int hm_open(hm_api *a, const char *lib) {
  memset(a,0,sizeof(*a));
  a->handle=dlopen(lib && *lib ? lib : "libsherpa-onnx-c-api.so",RTLD_NOW|RTLD_LOCAL);
  if(!a->handle){fprintf(stderr,"hayamimi: cannot open sherpa-onnx C runtime: %s\n",dlerror());return -1;}
#define HM_SYM(field, name) do { void *s=dlsym(a->handle,"SherpaOnnx" name); if(!s){fprintf(stderr,"hayamimi: missing symbol SherpaOnnx%s\n",name);goto fail;} memcpy(&a->field,&s,sizeof(s)); }while(0)
  HM_SYM(version,"GetVersionStr");
  HM_SYM(create_recognizer,"CreateOfflineRecognizer");
  HM_SYM(destroy_recognizer,"DestroyOfflineRecognizer");
  HM_SYM(create_stream,"CreateOfflineStream");
  HM_SYM(destroy_stream,"DestroyOfflineStream");
  HM_SYM(accept,"AcceptWaveformOffline");
  HM_SYM(decode,"DecodeOfflineStream");
  HM_SYM(get_result,"GetOfflineStreamResult");
  HM_SYM(free_result,"DestroyOfflineRecognizerResult");
  HM_SYM(read_wave,"ReadWave");
  HM_SYM(free_wave,"FreeWave");
#undef HM_SYM
  { const char *v=a->version();
    if(!v || (strncmp(v,"1.13.6",6)!=0 || (v[6] && v[6]!='+' && v[6]!='-'))) {
      fprintf(stderr,"hayamimi: incompatible sherpa-onnx ABI (%s); require 1.13.6\n",v?v:"unknown");
      goto fail;
    }
  }
  return 0;
fail:
  dlclose(a->handle);memset(a,0,sizeof(*a));return -1;
}
static void hm_close(hm_api *a){
  if(a->recognizer) a->destroy_recognizer(a->recognizer);
  if(a->handle) dlclose(a->handle);
  memset(a,0,sizeof(*a));
}

static int hm_pattern(char *out,size_t cap,const char *dir,const char *pattern) {
  char path[4096];glob_t g;memset(&g,0,sizeof(g));
  if(snprintf(path,sizeof(path),"%s/%s",dir,pattern)>=(int)sizeof(path))return -1;
  int rc=glob(path,0,NULL,&g);
  if(rc || g.gl_pathc==0 || strlen(g.gl_pathv[0])>=cap){globfree(&g);return -1;}
  /* Match the same first-sorted-glob convention as hayamimi _find(). */
  strcpy(out,g.gl_pathv[0]);globfree(&g);return 0;
}

/* Only one recognizer is resident: eliminates routing-cache + dependencies. */
static int hm_configure(hm_api *a,const char *root,const char *lang,int threads){
  hm_config c;char files[4][4096];const char *sub=NULL,*patterns[4]={NULL};
  int mode=0;
  memset(&c,0,sizeof(c));
  c.feat_config.sample_rate=16000;
  c.feat_config.feature_dim=80;
  c.model_config.provider="cpu";
  c.model_config.num_threads=threads>0 ? threads:4;
  c.decoding_method="greedy_search";
  if(!strcmp(lang,"ja")) {
    mode=1;sub="sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17";
    patterns[0]="encoder-*.int8.onnx";patterns[1]="decoder-*.int8.onnx";
    patterns[2]="joiner-*.int8.onnx";patterns[3]="tokens.txt";
    c.model_config.model_type="zipformer";
    c.model_config.modeling_unit="cjkchar";
    c.decoding_method="modified_beam_search";
  }else if(!strcmp(lang,"zh")) {
    mode=2;sub="sherpa-onnx-paraformer-zh-int8-2025-10-07";
    patterns[0]="model*.onnx";patterns[3]="tokens.txt";
  }else if(!strcmp(lang,"ko")||!strcmp(lang,"yue")||!strcmp(lang,"auto")) {
    mode=3;sub="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";
    patterns[0]="model*.onnx";patterns[3]="tokens.txt";
  }else if(!strcmp(lang,"en")||!strcmp(lang,"fr")||!strcmp(lang,"de")||!strcmp(lang,"es")||!strcmp(lang,"it")||!strcmp(lang,"pt")) {
    mode=4;sub="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
    patterns[0]="encoder*.onnx";patterns[1]="decoder*.onnx";
    patterns[2]="joiner*.onnx";patterns[3]="tokens.txt";
    c.model_config.model_type="nemo_transducer";
  }else{fprintf(stderr,"hayamimi: language '%s' unsupported in minimal C route\n",lang);return -1;}
  char dir[4096];
  if(snprintf(dir,sizeof(dir),"%s/%s",root,sub)>=(int)sizeof(dir))return -1;
  for(int i=0;i<4;i++)if(patterns[i] && hm_pattern(files[i],sizeof(files[i]),dir,patterns[i])){
    fprintf(stderr,"hayamimi: missing model file %s/%s\n",dir,patterns[i]);return -1;
  }
  c.model_config.tokens=files[3];
  switch(mode){
  case 1:case 4:
    c.model_config.transducer.encoder=files[0];c.model_config.transducer.decoder=files[1];c.model_config.transducer.joiner=files[2];break;
  case 2:c.model_config.paraformer.model=files[0];break;
  case 3:c.model_config.sense_voice.model=files[0];c.model_config.sense_voice.language="";
         c.model_config.sense_voice.use_itn=1;break;
  }
  a->recognizer=a->create_recognizer(&c);
  if(!a->recognizer){fprintf(stderr,"hayamimi: SherpaOnnxCreateOfflineRecognizer failed\n");return -1;}
  return 0;
}

static int hm_infer(hm_api *a,const float *samples,int32_t n,int rate,int partial) {
  if(n<=0 || n>16000*180 || !samples){fprintf(stderr,"hayamimi: invalid utterance length\n");return -1;}
  const void *stream=a->create_stream(a->recognizer);
  if(!stream) return -1;
  a->accept(stream,rate,samples,n);
  a->decode(a->recognizer,stream);
  const hm_result *r=a->get_result(stream);
  int ok=(r!=NULL);
  if(r){if(r->text){if(partial)fprintf(stderr,"[partial] %s\n",r->text);else puts(r->text);}a->free_result(r);}
  a->destroy_stream(stream);
  return ok?0:-1;
}

/* stdin: S16_LE mono @16000 Hz. Energy VAD is a deliberate dependency-free
 * replacement for Silero, not accuracy/latency parity with hayamimi. */
static int hm_run_stdin(hm_api *a,int partial){
  enum {SR=16000, CHUNK=1600, PRE=8000, MAX=SR*12,SIL=SR*35/100};
  int16_t input[CHUNK];float samples[MAX+PRE],history[PRE];
  int hpos=0,hcount=0,n=0,quiet=0,active=0;
  for(;;){
    size_t got=fread(input,sizeof(int16_t),CHUNK,stdin);
    if(!got)break;
    float power=0;
    for(size_t i=0;i<got;i++){float f=input[i]/32768.0f;power+=f*f;}
    float rms=sqrtf(power/(float)got);
    int voiced=rms>=0.016f; /* empirical threshold: tunable in later versions */
    if(!active && voiced){
      active=1;n=0;quiet=0;
      int start=(hpos+PRE-hcount)%PRE;
      for(int k=0;k<hcount;k++)samples[n++]=history[(start+k)%PRE];
    }
    for(size_t i=0;i<got;i++){
      float f=input[i]/32768.0f;
      if(active&&n<MAX+PRE)samples[n++]=f;
      history[hpos]=f;hpos=(hpos+1)%PRE;if(hcount<PRE)hcount++;
    }
    if(active){
      quiet=voiced?0:quiet+(int)got;
      if(quiet>=SIL||n>=MAX){
        int trimmed=n-quiet+SR/10;
        if(trimmed>n)trimmed=n;
        if(trimmed>SR/5&&hm_infer(a,samples,trimmed,SR,0))return -1;
        active=0;n=0;quiet=0;
      } else if(partial && n>=SR && (n%(SR/2)<CHUNK)){
        if(hm_infer(a,samples,n,SR,1))return -1;
      }
    }
  }
  if(active&&n>SR/5)return hm_infer(a,samples,n,SR,0);
  return 0;
}

static int cats_hayamimi_run(const char *root,const char *lib,const char *wav,
                             const char *lang,int threads,int raw,int partial){
  hm_api api;int rc=1;
  if(!raw&&!wav){fprintf(stderr,"hayamimi: set -w audio.wav or -r (raw stdin)\n");return 1;}
  if(hm_open(&api,lib))return 1;
  if(hm_configure(&api,root?root:"models",lang?lang:"ja",threads))goto done;
  if(raw){rc=hm_run_stdin(&api,partial)?1:0;}
  else{
    const hm_wave *wave=api.read_wave(wav);
    if(!wave){fprintf(stderr,"hayamimi: cannot read WAV %s (16-bit PCM mono expected)\n",wav);goto done;}
    rc=hm_infer(&api,wave->samples,wave->num_samples,wave->sample_rate,0)?1:0;
    api.free_wave(wave);
  }
done:hm_close(&api);return rc;
}
#endif

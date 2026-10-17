//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

/* cats_speech.h - unified native C audio front-end, WAV/fbank/VAD.
 * No external signal-processing libraries; energy-based VAD only.
 */
#ifndef CATS_SPEECH_H
#define CATS_SPEECH_H

/* ========================================================================
 * WAV, mel filterbank, native speech helpers
 * ======================================================================== */

/*
 * cats_speech.h - minimal native-C speech primitives for Cat's eye.
 * Copyright 2026. MPL-2.0 (Cat's eye integration).
 * No Python, sherpa-onnx, ONNX Runtime, or third-party C dependencies.
 *
 * This implements a subset of Sherpa-ONNX's *preprocessing / decoding*.
 * It is NOT a Zipformer neural-network inference implementation.
 *
 * File API and memory: caller owns all result allocations; use cs_wave_free,
 * cs_fbank_free and free() as documented. Not tied to Cat's eye's malloc macro.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
/* EG2PKG1 bundled model streaming. Keep all component parsing in cats_speech.h
 * to avoid new runtime dependencies or extra application headers. */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

static inline uint64_t cs_eg2_u64(const unsigned char *b){
  uint64_t x=0;for(int k=0;k<8;k++)x|=(uint64_t)b[k]<<(8*k);return x;
}
typedef struct {int fd;uint64_t base,size,cursor;} cs_eg2_slice;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
static ssize_t cs_eg2_slice_read(void*cookie,char*buf,size_t n){
  cs_eg2_slice*c=(cs_eg2_slice*)cookie;
  if(c->cursor>=c->size)return 0;
  if((uint64_t)n>c->size-c->cursor)n=(size_t)(c->size-c->cursor);
  ssize_t got=pread(c->fd,buf,n,(off_t)(c->base+c->cursor));
  if(got>0)c->cursor+=(uint64_t)got;
  return got;
}
static int cs_eg2_slice_seek(void*cookie,off64_t*pos,int whence){
  cs_eg2_slice*c=(cs_eg2_slice*)cookie;
  int64_t origin=whence==SEEK_SET?0:whence==SEEK_CUR?(int64_t)c->cursor:whence==SEEK_END?(int64_t)c->size:-1;
  if(origin<0||(*pos<0 && *pos < -origin) ||
     (*pos>0 && origin>INT64_MAX-*pos))return -1;
  int64_t target=origin+*pos;
  if(target<0||(uint64_t)target>c->size)return -1;
  c->cursor=(uint64_t)target;*pos=target;return 0;
}
static int cs_eg2_slice_close(void*cookie){
  cs_eg2_slice*c=(cs_eg2_slice*)cookie;int rc=close(c->fd);free(c);return rc;
}
#endif
/* Virtual path: /path/encoder.eg2m/layer_00.eg2l, or /path/ja.eg2m/reader.eg2r.
 * Unlike temporary extraction, one Conformer layer is streamed on demand. */
static inline FILE* cs_eg2_file_open(const char*path){
  if(!path)return NULL;
  const char*ext=strstr(path,".eg2m/");
  if(!ext)return fopen(path,"rb");
  size_t n=(size_t)(ext-path)+5;
  if(n>=4096)return NULL;
  char package[4096];memcpy(package,path,n);package[n]=0;
  const char*item=ext+6;
  size_t itemlen=strlen(item);
  if(itemlen<1||itemlen>63||strchr(item,'/')||strchr(item,'\\'))return NULL;
  int fd=open(package,O_RDONLY);if(fd<0)return NULL;
  FILE*fp=NULL;unsigned char h[16];
  struct stat st;
  if(fstat(fd,&st)||st.st_size<16||pread(fd,h,16,0)!=16||
     memcmp(h,"EG2PKG1\0",8))goto done;
  uint32_t count=((uint32_t)h[8]|(uint32_t)h[9]<<8|(uint32_t)h[10]<<16|(uint32_t)h[11]<<24),flags=((uint32_t)h[12]|(uint32_t)h[13]<<8|(uint32_t)h[14]<<16|(uint32_t)h[15]<<24);
  if(!count||count>64||flags||16u+(uint64_t)count*80>(uint64_t)st.st_size)goto done;
  uint64_t start=0,len=0;
  for(uint32_t i=0;i<count;i++){
    unsigned char entry[80];
    if(pread(fd,entry,80,(off_t)(16u+(uint64_t)i*80))!=80)goto done;
    if(!memchr(entry,0,64))goto done;
    uint64_t pos=cs_eg2_u64(entry+64),sz=cs_eg2_u64(entry+72);
    if(pos<16u+(uint64_t)count*80||pos>(uint64_t)st.st_size||
       sz>(uint64_t)st.st_size-pos||!sz)goto done;
    if(strlen((char*)entry)==itemlen&&!memcmp(entry,item,itemlen)){
      if(len)goto done;
      start=pos;len=sz;
    }
  }
  if(!len||start>INT64_MAX||len>INT64_MAX-start)goto done;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  cs_eg2_slice*c=(cs_eg2_slice*)calloc(1,sizeof(*c));if(!c)goto done;
  *c=(cs_eg2_slice){fd,start,len,0};
  cookie_io_functions_t io={0};
  io.read=cs_eg2_slice_read;io.seek=cs_eg2_slice_seek;io.close=cs_eg2_slice_close;
  fp=fopencookie(c,"rb",io);
  if(fp)return fp;
  free(c);
#else
  /* Portable fallback: preserve existing loader API with an anonymous temp file.
   * Linux/glibc uses a zero-extraction seekable stream instead. */
  fp=tmpfile();if(!fp)goto done;
  char block[65536];uint64_t off=0;
  while(off<len){
    size_t want=len-off>sizeof(block)?sizeof(block):(size_t)(len-off);
    ssize_t got=pread(fd,block,want,(off_t)(start+off));
    if(got<=0||fwrite(block,1,(size_t)got,fp)!=(size_t)got){fclose(fp);fp=NULL;goto done;}
    off+=(uint64_t)got;
  }
  rewind(fp);
#endif
 done:if(fd>=0)close(fd);return fp;
}



#ifndef CS_PI
#define CS_PI 3.14159265358979323846
#endif

typedef struct { float *pcm; size_t count; int rate; } cs_wave;
typedef struct { float *data; size_t frames; int dim, rate; } cs_features;

static inline unsigned cs_u16(const unsigned char *p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
static inline uint32_t cs_u32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static inline void cs_put32(unsigned char *p, uint32_t v) { for(int i=0;i<4;i++)p[i]=(unsigned char)(v>>(i*8)); }
static inline void cs_wave_free(cs_wave *w) { if(w){ free(w->pcm); memset(w,0,sizeof(*w)); } }
static inline void cs_fbank_free(cs_features *f) { if(f){ free(f->data); memset(f,0,sizeof(*f)); } }

/* RIFF/WAV (PCM 8/16/24/32, IEEE float32), downmix N channels to mono.
   Uses a 100 MB safety cap and checks malformed chunk sizes. */
static inline int cs_wave_read(const char *path, cs_wave *out) {
  FILE *fp=NULL; unsigned char hdr[12],chunk[8],fmt[40];
  unsigned fmt_tag=0,channels=0,bits=0,rate=0,block=0;
  uint32_t bytes=0; long data_at=-1;
  int ok=-1;
  if(!path||!out)return -1;
  memset(out,0,sizeof(*out));
  fp=fopen(path,"rb");if(!fp)return -1;
  if(fread(hdr,1,12,fp)!=12||memcmp(hdr,"RIFF",4)||memcmp(hdr+8,"WAVE",4))goto done;
  /* The RIFF size itself can be wrong; chunk scans stop at EOF. */
  for(int i=0;i<100000 && fread(chunk,1,8,fp)==8;i++) {
    uint32_t sz=cs_u32(chunk+4); long start=ftell(fp);
    if(start<0||sz>100000000U)goto done;
    if(!memcmp(chunk,"fmt ",4)) {
      if(sz<16||sz>sizeof(fmt)||fread(fmt,1,sz,fp)!=sz)goto done;
      fmt_tag=cs_u16(fmt);channels=cs_u16(fmt+2);rate=cs_u32(fmt+4);
      block=cs_u16(fmt+12);bits=cs_u16(fmt+14);
      /* WAVE_FORMAT_EXTENSIBLE: use the first 16 bytes of GUID subtype. */
      if(fmt_tag==0xfffe && sz>=40)fmt_tag=cs_u16(fmt+24);
    }else if(!memcmp(chunk,"data",4)) {
      if(data_at<0){data_at=start;bytes=sz;}
    }
    if((unsigned long)start+(unsigned long)sz+(sz&1)>(unsigned long)LONG_MAX)goto done;
    if(fseek(fp,start+(long)sz+(long)(sz&1),SEEK_SET))goto done;
  }
  if(data_at<0||channels<1||channels>8||rate<8000||rate>192000||bits%8||bits<8||bits>32)goto done;
  if(!((fmt_tag==1&&(bits==8||bits==16||bits==24||bits==32))||(fmt_tag==3&&bits==32)))goto done;
  if(block!=channels*(bits/8)||bytes%block||bytes==0||bytes>100000000U)goto done;
  size_t n=bytes/block;
  if(n>SIZE_MAX/sizeof(float))goto done;
  out->pcm=(float*)malloc(n*sizeof(float));if(!out->pcm)goto done;
  if(fseek(fp,data_at,SEEK_SET))goto done;
  unsigned char sample[4];
  for(size_t i=0;i<n;i++) {
    double s=0;
    for(unsigned c=0;c<channels;c++) {
      unsigned b=bits/8;
      if(fread(sample,1,b,fp)!=b)goto done;
      float v=0;
      if(fmt_tag==3){ uint32_t word=cs_u32(sample);memcpy(&v,&word,4); if(!isfinite(v))v=0; }
      else if(bits==8)v=((int)sample[0]-128)/128.f;
      else if(bits==16)v=(float)(int16_t)cs_u16(sample)/32768.f;
      else if(bits==24){int32_t x=(int32_t)((uint32_t)sample[0]|((uint32_t)sample[1]<<8)|((uint32_t)sample[2]<<16));if(x&0x800000)x|=(int32_t)0xff000000;v=x/8388608.f;}
      else v=(float)(int32_t)cs_u32(sample)/2147483648.f;
      s+=v;
    }
    out->pcm[i]=(float)(s/(double)channels);
  }
  out->count=n;out->rate=(int)rate;ok=0;
done:
  fclose(fp);if(ok)cs_wave_free(out);return ok;
}

/* Stage 32: random-access RIFF reader for long native-ASR recordings.
 * Bounds memory to one ~30-second resampled window and avoids the legacy
 * 100 MB full-file cap. Uses the SAME sinc math as cs_resample, but with
 * global absolute sample positions to avoid discontinuities between windows.
 * Caller owns cs_wave_stream and must close it (even on failure). */
#include <sys/types.h>
typedef struct {
  FILE *fp;
  uint64_t data_at;
  size_t count;
  unsigned fmt_tag,channels,bits,block;
  int rate;
} cs_wave_stream;
static inline void cs_wave_stream_close(cs_wave_stream *s){
  if(s){if(s->fp)fclose(s->fp);memset(s,0,sizeof(*s));}
}
static inline int cs_wave_stream_open(cs_wave_stream *s,const char *path){
  if(!s||!path)return -1;
  memset(s,0,sizeof(*s));
  FILE *f=fopen(path,"rb");if(!f)return -1;
  unsigned char h[12],chunk[8],fmt[40];int rc=-1;
  uint64_t data_at=0,bytes=0;int found=0,hasfmt=0;
  if(fread(h,1,12,f)!=12||memcmp(h,"RIFF",4)||memcmp(h+8,"WAVE",4))goto done;
  if(fseeko(f,0,SEEK_END))goto done;
  off_t end=ftello(f);if(end<12||fseeko(f,12,SEEK_SET))goto done;
  for(int k=0;k<100000;k++){
    off_t pos=ftello(f);if(pos<0||(uint64_t)pos+8>(uint64_t)end)break;
    if(fread(chunk,1,8,f)!=8)goto done;
    uint64_t start=(uint64_t)pos+8, n=cs_u32(chunk+4);
    if(start+n>(uint64_t)end)goto done;
    if(!memcmp(chunk,"fmt ",4)){
      if(n<16||n>sizeof(fmt)||fread(fmt,1,(size_t)n,f)!=(size_t)n)goto done;
      s->fmt_tag=cs_u16(fmt);s->channels=cs_u16(fmt+2);s->rate=(int)cs_u32(fmt+4);
      s->block=cs_u16(fmt+12);s->bits=cs_u16(fmt+14);
      if(s->fmt_tag==0xfffe && n>=40)s->fmt_tag=cs_u16(fmt+24);
      hasfmt=1;
    }else if(!memcmp(chunk,"data",4)&&!found){data_at=start;bytes=n;found=1;}
    uint64_t next=start+n+(n&1);
    if(next>(uint64_t)end) {if(next!=(uint64_t)end+1)goto done;next=(uint64_t)end;}
    if(fseeko(f,(off_t)next,SEEK_SET))goto done;
  }
  if(!found||!hasfmt||s->channels<1||s->channels>8||s->rate<8000||s->rate>192000||
     s->bits<8||s->bits>32||s->bits%8||
     !((s->fmt_tag==1&&(s->bits==8||s->bits==16||s->bits==24||s->bits==32))||
       (s->fmt_tag==3&&s->bits==32))||
     s->block!=s->channels*(s->bits/8)||!bytes||bytes%s->block)goto done;
  s->count=(size_t)(bytes/s->block);
  if((uint64_t)s->count!=bytes/s->block||s->count>SIZE_MAX/16000u)goto done;
  s->data_at=data_at;s->fp=f;rc=0;
done:
  if(rc){fclose(f);memset(s,0,sizeof(*s));}
  return rc;
}
static inline size_t cs_wave_stream_length16(const cs_wave_stream *s){
  if(!s||!s->fp)return 0;
  return (size_t)((double)s->count*16000/s->rate+0.5);
}
static inline int cs_wave_stream_samples(cs_wave_stream *s,size_t first,size_t n,float *dst){
  if(!s||!s->fp||!dst||first>s->count||n>s->count-first||
     first>(UINT64_MAX-s->data_at)/s->block)return -1;
  uint64_t offset=s->data_at+(uint64_t)first*s->block;
  if(offset>(uint64_t)INT64_MAX||fseeko(s->fp,(off_t)offset,SEEK_SET))return -1;
  unsigned char b[4];const unsigned sz=s->bits/8;
  for(size_t i=0;i<n;i++){
    double sum=0;
    for(unsigned c=0;c<s->channels;c++){
      if(fread(b,1,sz,s->fp)!=sz)return -1;
      float v=0;
      if(s->fmt_tag==3){uint32_t a=cs_u32(b);memcpy(&v,&a,4);if(!isfinite(v))v=0;}
      else if(s->bits==8)v=((int)b[0]-128)/128.f;
      else if(s->bits==16)v=(float)(int16_t)cs_u16(b)/32768.f;
      else if(s->bits==24){int32_t x=(int32_t)((uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16));if(x&0x800000)x|=(int32_t)0xff000000;v=x/8388608.f;}
      else v=(float)(int32_t)cs_u32(b)/2147483648.f;
      sum+=v;
    }
    dst[i]=(float)(sum/s->channels);
  }
  return 0;
}
/* start and n refer to the 16-kHz output timeline, not WAV samples. */
static inline int cs_wave_stream_read16(cs_wave_stream*s,size_t start,size_t n,float *out){
  if(!s||!out||!n||n>480000)return -1;
  size_t total=cs_wave_stream_length16(s);
  if(start>total||n>total-start)return -1;
  if(s->rate==16000)return cs_wave_stream_samples(s,start,n,out);
  double cutoff=s->rate>16000?16000.0/s->rate:1.0;
  double support=8.0/cutoff;
  double pos0=(double)start*s->rate/16000;
  double pos1=(double)(start+n-1)*s->rate/16000;
  size_t first=pos0>support?(size_t)ceil(pos0-support):0;
  double lastd=floor(pos1+support);
  size_t last=lastd>=(double)s->count?s->count-1:(size_t)lastd;
  if(first>last||last-first+1>SIZE_MAX/sizeof(float))return -1;
  size_t extent=last-first+1;
  float *src=(float*)malloc(extent*sizeof(float));if(!src)return -1;
  int rc=cs_wave_stream_samples(s,first,extent,src);
  if(!rc){
#pragma omp parallel for schedule(static)
  for(long ii=0;ii<(long)n;ii++){
    const size_t i=(size_t)ii;
    double pos=(double)(start+i)*s->rate/16000;
    ptrdiff_t a=(ptrdiff_t)ceil(pos-support),b=(ptrdiff_t)floor(pos+support);
    if(a<0)a=0;
    if(b>=(ptrdiff_t)s->count)b=(ptrdiff_t)s->count-1;
    double val=0,wsum=0;
    for(ptrdiff_t j=a;j<=b;j++){
      double t=((double)j-pos)*cutoff;
      double sinc=fabs(t)<1e-9?1.0:sin(CS_PI*t)/(CS_PI*t);
      double win=fabs(t)<8.0?(0.5+0.5*cos(CS_PI*t/8.0)):0.0;
      double w=sinc*win;val+=(double)src[j-(ptrdiff_t)first]*w;wsum+=w;
    }
    out[i]=(float)(wsum?val/wsum:0.0);
  }
  }
  free(src);return rc;
}

/* 16-tap/windowed-sinc lowpass with scale adjusted for downsampling.
   Accurate enough as a lightweight replacement; not bit-identical to sherpa. */
static inline int cs_resample(const float *in,size_t n,int from,int to,float **dest,size_t *len) {
  if(!in||!dest||!len||!n||from<1||to<1||n>SIZE_MAX/(size_t)to)return -1;
  *dest=NULL;*len=0;
  size_t count=(size_t)((double)n*to/from+0.5);
  if(!count||count>SIZE_MAX/sizeof(float))return -1;
  float *out=(float*)malloc(count*sizeof(float));if(!out)return -1;
  if(from==to){size_t m=count<n?count:n;memcpy(out,in,m*sizeof(float));}
  else {
    double cutoff=from>to?(double)to/from:1.0;
    double support=8.0/cutoff;
    for(size_t i=0;i<count;i++) {
      double pos=(double)i*from/to, v=0,wsum=0;
      ptrdiff_t a=(ptrdiff_t)ceil(pos-support),b=(ptrdiff_t)floor(pos+support);
      if(a<0)a=0;
      if(b>=(ptrdiff_t)n)b=(ptrdiff_t)n-1;
      for(ptrdiff_t j=a;j<=b;j++) {
        double t=((double)j-pos)*cutoff;
        double sinc=(fabs(t)<1e-9)?1.0:sin(CS_PI*t)/(CS_PI*t);
        double win=(fabs(t)<8.0)?(0.5+0.5*cos(CS_PI*t/8.0)):0.0;
        double w=sinc*win;v+=in[j]*w;wsum+=w;
      }
      out[i]=(float)(wsum? v/wsum : 0.0);
    }
  }
  *dest=out;*len=count;return 0;
}

/* In-place 512-point complex FFT. All twiddles prepared once per call. */
static inline void cs_fft512(float *re,float *im,const float *twc,const float *tws) {
  for(unsigned i=1,j=0;i<512;i++){
    unsigned b=256;for(;j&b;b>>=1)j^=b;j^=b;
    if(i<j){float t=re[i];re[i]=re[j];re[j]=t;t=im[i];im[i]=im[j];im[j]=t;}
  }
  for(unsigned len=2;len<=512;len*=2){
    unsigned half=len/2,step=512/len;
    for(unsigned i=0;i<512;i+=len)for(unsigned j=0;j<half;j++){
      unsigned k=j*step;
      float vr=re[i+j+half]*twc[k]-im[i+j+half]*tws[k];
      float vi=re[i+j+half]*tws[k]+im[i+j+half]*twc[k];
      float ur=re[i+j],ui=im[i+j];
      re[i+j]=ur+vr; im[i+j]=ui+vi;
      re[i+j+half]=ur-vr; im[i+j+half]=ui-vi;
    }
  }
}

/* Kaldi-style 80-bin log-mel frontend: 16kHz, 25ms Povey window,
 * 10ms stride, 512-point FFT, 20..7600Hz, snip_edges=false, no dither.
 * Input PCM normalized to [-1,1]. Differences from kaldi-native-fbank are
 * possible; compare against upstream numerical reference before production.
 */
static inline int cs_fbank80(const float *pcm,size_t count,int in_rate,cs_features *out) {
  enum {SR=16000,SHIFT=160,WINDOW=400,NFFT=512,BINS=80,SPECTRUM=257};
  if(!out||!pcm||!count||count>16000u*60u*30u)return -1;
  memset(out,0,sizeof(*out));
  float *resampled=NULL;size_t n=count;
  if(in_rate!=SR){if(cs_resample(pcm,count,in_rate,SR,&resampled,&n))return -1;pcm=resampled;}
  size_t frames=(n+SHIFT/2)/SHIFT;
  if(!frames||frames>SIZE_MAX/(BINS*sizeof(float))){free(resampled);return -1;}
  float *features=(float*)malloc(frames*BINS*sizeof(float));
  if(!features){free(resampled);return -1;}
  float *filter=(float*)malloc(BINS*SPECTRUM*sizeof(float));
  if(!filter){free(features);free(resampled);return -1;}
  memset(filter,0,BINS*SPECTRUM*sizeof(float));
  float win[WINDOW],twc[256],tws[256];
  for(int i=0;i<WINDOW;i++)win[i]=powf(0.5f-0.5f*cosf((float)(2*CS_PI*i/(WINDOW-1))),0.85f);
  for(int i=0;i<256;i++){twc[i]=cosf((float)(2*CS_PI*i/512));tws[i]=-sinf((float)(2*CS_PI*i/512));}
  const double low=1127.0*log(1.0+20.0/700),high=1127.0*log(1.0+7600.0/700);
  for(int b=0;b<BINS;b++) {
    double l=low+(high-low)*b/(BINS+1),c=low+(high-low)*(b+1)/(BINS+1),h=low+(high-low)*(b+2)/(BINS+1);
    for(int k=0;k<SPECTRUM;k++){
      double hz=k*(double)SR/NFFT;
      double mel=1127.0*log(1.0+hz/700.0);
      if(mel>l&&mel<h)filter[b*SPECTRUM+k]=(float)(mel<c?(mel-l)/(c-l):(h-mel)/(h-c));
    }
  }
  for(size_t f=0;f<frames;f++) {
    float re[NFFT]={0},im[NFFT]={0};
    ptrdiff_t start=(ptrdiff_t)(f*SHIFT)+(SHIFT/2)-(WINDOW/2);
    double sum=0;
    for(int i=0;i<WINDOW;i++) {
      ptrdiff_t idx=start+i;
      /* kaldi snip_edges=false: reflection at beginning/end. */
      while(idx<0||idx>=(ptrdiff_t)n){if(idx<0)idx=-idx-1;else idx=2*(ptrdiff_t)n-1-idx;}
      re[i]=pcm[idx];sum+=re[i];
    }
    float mean=(float)(sum/WINDOW);for(int i=0;i<WINDOW;i++)re[i]-=mean;
    /* Reverse-order preemphasis, followed by Povey window. */
    for(int i=WINDOW-1;i>0;i--)re[i]=(re[i]-0.97f*re[i-1])*win[i];
    re[0]=(re[0]-0.97f*re[0])*win[0];
    cs_fft512(re,im,twc,tws);
    float power[SPECTRUM];
    for(int k=0;k<SPECTRUM;k++)power[k]=re[k]*re[k]+im[k]*im[k];
    for(int b=0;b<BINS;b++){
      const float *weights=filter+b*SPECTRUM;
      double e=0;for(int k=0;k<SPECTRUM;k++)e+=(double)weights[k]*power[k];
      features[f*BINS+b]=logf((float)(e>FLT_EPSILON?e:FLT_EPSILON));
    }
  }
  free(filter);free(resampled);
  out->data=features;out->frames=frames;out->dim=BINS;out->rate=SR;
  return 0;
}

/* Raw feature file: magic CFB1 + 3 LE uint32 (sample rate, frames, bins)
 * followed by little-endian IEEE754 float32 row-major [frames,bins]. */
static inline int cs_fbank_write(const char *path,const cs_features *f) {
  if(!path||!f||!f->data||f->frames>UINT32_MAX||f->dim<=0)return -1;
  FILE *fp=fopen(path,"wb");if(!fp)return -1;
  unsigned char h[16]={'C','F','B','1'};
  cs_put32(h+4,(uint32_t)f->rate);cs_put32(h+8,(uint32_t)f->frames);cs_put32(h+12,(uint32_t)f->dim);
  int ok=fwrite(h,1,sizeof(h),fp)==sizeof(h);
  /* Explicit LE float encoding, including on big-endian systems. */
  for(size_t i=0;ok&&i<f->frames*(size_t)f->dim;i++){
    uint32_t bits;unsigned char raw[4];memcpy(&bits,&f->data[i],4);cs_put32(raw,bits);
    ok=fwrite(raw,1,4,fp)==4;
  }
  if(fclose(fp)!=0)ok=0;
  return ok?0:-1;
}

/* CTC greedy is useful for CTC models (not Zipformer transducer!).
 * logits shape: [frames, vocabulary]; returns number of emitted token IDs. */
static inline int cs_ctc_greedy(const float *logits,size_t frames,int vocab,int blank,
                         int *tokens,size_t capacity) {
  if(!logits||!tokens||vocab<2||blank<0||blank>=vocab)return -1;
  int prev=blank;size_t used=0;
  for(size_t i=0;i<frames;i++) {
    const float *row=logits+i*(size_t)vocab;int best=0;
    for(int j=1;j<vocab;j++)if(row[j]>row[best])best=j;
    if(best!=blank&&best!=prev){if(used>=capacity)return -1;tokens[used++]=best;}
    prev=best;
  }
  return used>INT_MAX?-1:(int)used;
}

/* Dependency-free F32 and per-row symmetric INT8 matrix-vector kernels.
 * w: [rows,cols], x: [cols], output [rows], scales: [rows] for INT8. */
static inline void cs_matvec_f32(float *y,const float *w,const float *x,int rows,int cols) {
  for(int i=0;i<rows;i++){const float *r=w+(size_t)i*cols;float s=0;
    for(int j=0;j<cols;j++)s+=r[j]*x[j];
    y[i]=s;}
}
static inline void cs_matvec_q8(float *y,const int8_t *w,const float *scale,const float *x,int rows,int cols) {
  for(int i=0;i<rows;i++){const int8_t *r=w+(size_t)i*cols;float s=0;
    for(int j=0;j<cols;j++)s+=(float)r[j]*x[j];
    y[i]=s*scale[i];}
}

/* Minimal model-backend interface for RNN-Transducer greedy decoding.
 * A future native Zipformer backend must implement predictor() and joiner().
 * The caller supplies *already encoded* frames. These callbacks are intentionally
 * not implemented as the model requires its real trained weights and operator graph.
 */
typedef struct {
  int vocabulary,blank,context_size,predictor_dim,max_symbols_per_frame;
  void *user;
  int (*predictor)(void *user,const int *context,int context_len,float *out);
  int (*joiner)(void *user,const float *frame,const float *prediction,float *logits);
} cs_rnnt;
static inline int cs_rnnt_greedy(const cs_rnnt *m,const float *encoded,size_t frames,int encoder_dim,
                          int initial_token,int *output,size_t capacity) {
  if(!m||!encoded||!output||!m->predictor||!m->joiner||
     m->vocabulary<2||m->blank<0||m->blank>=m->vocabulary||m->context_size<1||
     m->context_size>32||m->predictor_dim<1||m->predictor_dim>65536||
     m->max_symbols_per_frame<1||m->max_symbols_per_frame>32||encoder_dim<1)return -1;
  int context[32];for(int i=0;i<m->context_size;i++)context[i]=initial_token;
  float *p=(float*)malloc((size_t)m->predictor_dim*sizeof(float));
  float *logits=(float*)malloc((size_t)m->vocabulary*sizeof(float));
  if(!p||!logits){free(p);free(logits);return -1;}
  int used=0,ok=0;
  for(size_t t=0;t<frames;t++) {
    const float *frame=encoded+t*(size_t)encoder_dim;
    for(int s=0;s<m->max_symbols_per_frame;s++) {
      if(m->predictor(m->user,context,m->context_size,p)||m->joiner(m->user,frame,p,logits)){ok=-1;goto done;}
      int best=0;
      for(int v=1;v<m->vocabulary;v++)if(logits[v]>logits[best])best=v;
      if(best==m->blank)break;
      if((size_t)used>=capacity){ok=-1;goto done;}
      output[used++]=best;
      memmove(context,context+1,(size_t)(m->context_size-1)*sizeof(int));
      context[m->context_size-1]=best;
    }
  }
  ok=used;
done:free(p);free(logits);return ok;
}


/* ========================================================================
 * Simple energy VAD and audio segmenting
 * ======================================================================== */

/* Minimal dependency-free WAV segmentation for experimental offline ASR.
 * Pure signal-energy VAD -- NOT the Silero neural VAD in Hayamimi.
 * Includes REAL audio pre-roll to reduce clipped word onsets.
 * MPL-2.0 2026.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct { size_t begin,end; } cs_segment;
/* Default: 20ms RMS every 10ms, 50ms onset, 400ms quiet releases,
   1sec real-audio history, max 15sec utterance. Rate 8..192k Hz. */
static inline int cs_vad_split(const float *pcm,size_t n,int rate,
                             cs_segment **out,size_t *out_count){
  if(!pcm||!out||!out_count||rate<8000||rate>192000||!n)return -1;
  *out=NULL;*out_count=0;
  const size_t hop=(size_t)rate/100u,win=(size_t)rate/50u;
  if(!hop||!win)return -1;
  /* More than 3h of audio is intentionally unsupported by this tiny path. */
  size_t nwin=(n+hop-1)/hop;
  if(nwin>1080000u)return -1;
  cs_segment *segments=NULL;
  size_t used=0,cap=0;
  size_t start=0,prev_end=0;
  int active=0,onset=0,silence=0,voiced=0;
  /* Convert threshold to normalized WAV amplitude.  Two thresholds provide
     hysteresis, so breaths are less likely to split a phrase. */
  const double on_level=0.012,off_level=0.007;
  for(size_t t=0;t<nwin;t++){
    size_t at=t*hop,end=at+win;
    if(end>n)end=n;
    double sum=0;
    for(size_t s=at;s<end;s++){
      double x=pcm[s];if(!isfinite(x))x=0;
      sum+=x*x;
    }
    double rms=sqrt(sum/(end-at));
    if(!active){
      onset=(rms>=on_level)?onset+1:0;
      if(onset>=5){
        active=1;silence=0;voiced=onset;
        size_t observed_onset=at>=4*hop?at-4*hop:0;
        size_t preroll=(size_t)rate;  /* up to 1sec real audio */
        start=observed_onset>preroll?observed_onset-preroll:0;
        if(start<prev_end)start=prev_end;
      }
    }else{
      if(rms<off_level)silence++;else {silence=0;voiced++;}
      size_t duration=end-start;
      if(silence>=40 || duration>=(size_t)rate*15u || t==nwin-1){
        size_t stop=end;
        if(stop>n)stop=n;
        /* Ignore very short energy spikes.  Voiced windows, not padded
           segment duration, determine if there was real detected speech. */
        if(stop>start && voiced>=6){
          if(used==cap){
            size_t nc=cap?cap*2u:8u;
            if(nc>2048u){free(segments);return -1;}
            cs_segment*p=realloc(segments,nc*sizeof(*segments));
            if(!p){free(segments);return -1;}
            segments=p;cap=nc;
          }
          segments[used++]=(cs_segment){start,stop};
          prev_end=stop;
        }
        active=0;silence=0;onset=0;voiced=0;
      }
    }
  }
  /* A still-active speech segment at EOF must be returned too. */
  if(active && voiced>=6 && n>start){
    if(used==cap){
      size_t nc=cap?cap*2u:8u;
      if(nc>2048u){free(segments);return -1;}
      cs_segment*p=realloc(segments,nc*sizeof(*segments));
      if(!p){free(segments);return -1;}
      segments=p;cap=nc;
    }
    segments[used++]=(cs_segment){start,n};
  }
  *out=segments;*out_count=used;
  return 0;
}


/* ========================================================================
 * EmbeddingGemma 2 audio feature extractor and CTC labels (stage 27)
 * ========================================================================
 * Uses the published Gemma4 audio front end, NOT the 80-bin Kaldi fbank.
 * 16 kHz mono; periodic Hann, 320-sample window, 512-point magnitude FFT,
 * 128 HTK mel bands, log(mel+0.001), 160-sample hop, 160 zero left pad.
 * The optional learned reader and large encoder weights are NOT bundled.
 */
#define CS_EG2_RATE 16000
#define CS_EG2_MEL 128
#define CS_EG2_BUCKET 3000
#define CS_EG2_TOKENS 750
#define CS_EG2_STEP_TOKENS 653
#define CS_EG2_OVERLAP_TOKENS 96

typedef struct {int32_t word;float top1,blank;} cs_eg2_score;
typedef struct {char **word;size_t count;} cs_eg2_vocab;

static inline void cs_eg2_vocab_free(cs_eg2_vocab*v){
  if(!v)return;
  for(size_t i=0;i<v->count;i++)free(v->word[i]);
  free(v->word);memset(v,0,sizeof(*v));
}
/* Plain newline-separated word list: line index is the model class index.
 * Deliberately different from the RNNT token-id and byte-BPE formats. */
static inline int cs_eg2_vocab_load(cs_eg2_vocab*v,const char*path){
  if(!v||!path)return -1;
  memset(v,0,sizeof(*v));FILE*f=cs_eg2_file_open(path);if(!f)return -1;
  size_t cap=0;int rc=-1;char line[4096];
  while(fgets(line,sizeof(line),f)){
    size_t n=strlen(line);
    if(!n||(line[n-1]!='\n'&&!feof(f)))goto done;
    while(n&&(line[n-1]=='\n'||line[n-1]=='\r'))line[--n]=0;
    if(!n||v->count>=1000000)goto done;
    if(v->count==cap){
      size_t next=cap?cap*2:512;
      char**p=(char**)realloc(v->word,next*sizeof(*p));
      if(!p)goto done;
      v->word=p;cap=next;
    }
    char*s=(char*)malloc(n+1);if(!s)goto done;
    memcpy(s,line,n+1);v->word[v->count++]=s;
  }
  if(ferror(f)||!v->count)goto done;
  rc=0;
done:fclose(f);if(rc)cs_eg2_vocab_free(v);return rc;
}

/* Persistent precomputed FFT roots and sparse mel triangle endpoints.
 * No global mutable state; safe to use separate state per thread. */
typedef struct {
  double wr[256],wi[256],hann[320],mel[CS_EG2_MEL][257];
  int low[CS_EG2_MEL],high[CS_EG2_MEL];
} cs_eg2_frontend;
static inline double cs_eg2_hz_to_mel(double hz){return 2595.0*log10(1.0+hz/700.0);}
static inline double cs_eg2_mel_to_hz(double mel){return 700.0*(pow(10.0,mel/2595.0)-1.0);}
static inline void cs_eg2_frontend_init(cs_eg2_frontend*s){
  memset(s,0,sizeof(*s));
  for(int k=0;k<256;k++){
    double t=-2.0*CS_PI*k/512.0;s->wr[k]=cos(t);s->wi[k]=sin(t);
  }
  /* Precompute the unchanged periodic Hann window for every frame. */
  for(int k=0;k<320;k++)s->hann[k]=0.5-0.5*cos(2.0*CS_PI*k/320.0);
  const double max_mel=cs_eg2_hz_to_mel(8000.0);
  for(int j=0;j<CS_EG2_MEL;j++){
    const double left=cs_eg2_mel_to_hz(max_mel*j/129.0);
    const double middle=cs_eg2_mel_to_hz(max_mel*(j+1)/129.0);
    const double right=cs_eg2_mel_to_hz(max_mel*(j+2)/129.0);
    s->low[j]=257;s->high[j]=0;
    for(int k=0;k<=256;k++){
      double hz=(double)k*16000.0/512.0;
      double w=hz<middle?(hz-left)/(middle-left):(right-hz)/(right-middle);
      if(w>0){s->mel[j][k]=w;if(k<s->low[j])s->low[j]=k;if(k>s->high[j])s->high[j]=k;}
    }
  }
}
/* Stage 32: load exact reference HTK mel bank, shape [257,128] float64 LE.
 * The repo-specific mel_filters.bin avoids small but systematic frontend
 * parity differences caused by recomputing coefficient triangles. */
static inline int cs_eg2_frontend_load_mel(cs_eg2_frontend*s,const char*path){
  if(!s||!path)return -1;
  FILE *f=cs_eg2_file_open(path);if(!f)return -1;
  int rc=-1;
  if(fseeko(f,0,SEEK_END)||ftello(f)!=(off_t)(257*CS_EG2_MEL*8)||
     fseeko(f,0,SEEK_SET))goto done;
  for(int k=0;k<=256;k++)for(int j=0;j<CS_EG2_MEL;j++){
    unsigned char b[8];uint64_t bits=0;
    if(fread(b,1,8,f)!=8)goto done;
    for(int h=0;h<8;h++)bits|=(uint64_t)b[h]<<(8*h);
    double v;memcpy(&v,&bits,8);
    if(!isfinite(v)||v<0||v>100)goto done;
    s->mel[j][k]=v;
  }
  for(int j=0;j<CS_EG2_MEL;j++){
    s->low[j]=257;s->high[j]=0;
    for(int k=0;k<=256;k++)if(s->mel[j][k]>0){
      if(k<s->low[j])s->low[j]=k;
      if(k>s->high[j])s->high[j]=k;
    }
    /* Low HTK bins may legitimately be all zero on a 512-point FFT. */
  }
  rc=0;
done:fclose(f);return rc;
}
/* Complex in-place FFT (double): low error in near-silent log-mel bins. */
static inline void cs_eg2_fft(const cs_eg2_frontend*s,double*r,double*i){
  for(unsigned a=1,b=0;a<512;a++){
    unsigned m=256;for(;b&m;m>>=1)b^=m;b^=m;
    if(a<b){double z=r[a];r[a]=r[b];r[b]=z;z=i[a];i[a]=i[b];i[b]=z;}
  }
  for(unsigned step=2;step<=512;step*=2){
    unsigned half=step/2,delta=512/step;
    for(unsigned b=0;b<512;b+=step)for(unsigned j=0;j<half;j++){
      unsigned k=j*delta;
      double ar=r[b+j+half]*s->wr[k]-i[b+j+half]*s->wi[k];
      double ai=r[b+j+half]*s->wi[k]+i[b+j+half]*s->wr[k];
      double br=r[b+j],bi=i[b+j];
      r[b+j]=br+ar;i[b+j]=bi+ai;
      r[b+j+half]=br-ar;i[b+j+half]=bi-ai;
    }
  }
}
/* Extract a window at most 30s long, with zero padded output [3000,128].
 * Audio starts at a local window boundary, so long-clip context requires
 * overlapping windows with the decoder policy below. */
static inline int cs_eg2_features(const cs_eg2_frontend*s,
    const float*pcm,size_t n,float*out,size_t*valid_frames){
  if(!s||!pcm||!out||!valid_frames||n>480000)return -1;
  size_t frames=n>=161?1+(n-161)/160:0;
  if(frames>CS_EG2_BUCKET)return -1;
  memset(out,0,(size_t)CS_EG2_BUCKET*CS_EG2_MEL*sizeof(float));
  for(size_t f=0;f<frames;f++){
    double re[512]={0},im[512]={0},mag[257];
    for(int j=0;j<320;j++){
      ptrdiff_t at=(ptrdiff_t)(f*160)+(ptrdiff_t)j-160;
      double v=at<0||at>=(ptrdiff_t)n?0:(double)pcm[at];
      re[j]=v*s->hann[j];
    }
    cs_eg2_fft(s,re,im);
    for(int k=0;k<=256;k++)mag[k]=hypot(re[k],im[k]);
    for(int j=0;j<CS_EG2_MEL;j++){
      double energy=0;
      for(int k=s->low[j];k<=s->high[j];k++)energy+=mag[k]*s->mel[j][k];
      out[f*CS_EG2_MEL+j]=(float)log(energy+0.001);
    }
  }
  *valid_frames=frames;return 0;
}
/* Gemma4 second strided conv needs half-resolution padding mask. */
static inline void cs_eg2_masks(size_t frames,float *half,float*token,size_t *count){
  const size_t nh=(frames+1)/2,nt=(nh+1)/2;
  if(half)for(size_t j=0;j<1500;j++)half[j]=(float)(j<nh);
  if(token)for(size_t j=0;j<750;j++)token[j]=(float)(j<nt);
  if(count)*count=nt;
}

/* Small, versioned binary interchange used to call an independent backend
 * (for example an exported Core AI graph) without linking it into C.
 * EG2F: magic, version, frame_count, then 3000x128 little-endian f32.
 * EG2S: magic, version, score_count, then (int32,f32,f32) per token.
 * These files are exchange formats, NOT pretrained models. */
static inline int cs_eg2_io_header(FILE*f,const char*magic,uint32_t n,int write){
  unsigned char b[12];
  if(write){memcpy(b,magic,4);cs_put32(b+4,1);cs_put32(b+8,n);return fwrite(b,1,12,f)==12?0:-1;}
  if(fread(b,1,12,f)!=12||memcmp(b,magic,4)||cs_u32(b+4)!=1||cs_u32(b+8)!=n)return -1;
  return 0;
}
static inline int cs_eg2_write_features(const char*path,const float*f,size_t valid){
  if(!path||!f||valid>CS_EG2_BUCKET)return -1;
  FILE*fp=fopen(path,"wb");if(!fp)return -1;
  int rc=cs_eg2_io_header(fp,"EG2F",(uint32_t)valid,1);
  if(!rc&&fwrite(f,sizeof(float),(size_t)CS_EG2_BUCKET*CS_EG2_MEL,fp)!=(size_t)CS_EG2_BUCKET*CS_EG2_MEL)rc=-1;
  if(fclose(fp))rc=-1;
  return rc;
}
static inline int cs_eg2_write_scores(const char*path,const cs_eg2_score*s,size_t n){
  if(!path||(!s&&n)||n>100000000U)return -1;
  FILE*f=fopen(path,"wb");if(!f)return -1;
  int rc=cs_eg2_io_header(f,"EG2S",(uint32_t)n,1);
  for(size_t k=0;!rc&&k<n;k++){
    unsigned char b[12];uint32_t x=0,y=0;
    if(!isfinite(s[k].top1)||!isfinite(s[k].blank)||s[k].word<0){rc=-1;break;}
    cs_put32(b,(uint32_t)s[k].word);
    memcpy(&x,&s[k].top1,4);memcpy(&y,&s[k].blank,4);
    cs_put32(b+4,x);cs_put32(b+8,y);
    if(fwrite(b,1,12,f)!=12)rc=-1;
  }
  if(fclose(f))rc=-1;
  return rc;
}
static inline int cs_eg2_read_scores(const char*path,cs_eg2_score**out,size_t*n){
  if(!path||!out||!n)return -1;
  *out=NULL;*n=0;
  FILE*f=fopen(path,"rb");if(!f)return -1;
  unsigned char h[12];int rc=-1;cs_eg2_score*s=NULL;
  if(fread(h,1,12,f)!=12||memcmp(h,"EG2S",4)||cs_u32(h+4)!=1)goto done;
  size_t cnt=cs_u32(h+8);
  if(cnt>100000000U||cnt>SIZE_MAX/sizeof(*s))goto done;
  s=(cs_eg2_score*)malloc((cnt?cnt:1)*sizeof(*s));if(!s)goto done;
  for(size_t i=0;i<cnt;i++){
    unsigned char b[12];uint32_t a=0,c=0;
    if(fread(b,1,12,f)!=12)goto done;
    s[i].word=(int32_t)cs_u32(b);a=cs_u32(b+4);c=cs_u32(b+8);
    memcpy(&s[i].top1,&a,4);memcpy(&s[i].blank,&c,4);
    if(s[i].word<0||!isfinite(s[i].top1)||!isfinite(s[i].blank))goto done;
  }
  if(fgetc(f)!=EOF||ferror(f))goto done;
  *out=s;*n=cnt;s=NULL;rc=0;
done:free(s);fclose(f);return rc;
}
/* Strict CTC: blank if blank_logit > best_word_logit; equal -> word.
 * A blank resets the repetition state. A repeated nonblank is emitted once.
 * Space-separated words, no punctuation or casing hallucinated. */
/* Language-aware CTC. Japanese word pieces are joined without inserting ASCII spaces.
 * The public English ABI remains identical. Explicit U+2581 word-boundary
 * pieces (SentencePiece convention) become a single real ASCII space.
 * Blank always resets repeat collapse; whitespace is never injected for ja.
 */
static inline int cs_eg2_ctc_decode_lang(const cs_eg2_score*s,size_t n,
    const cs_eg2_vocab*v,char*out,size_t cap,int japanese){
  if((n&&!s)||!v||!out||!cap)return -1;
  size_t used=0;int32_t prev=-1;out[0]=0;
  for(size_t i=0;i<n;i++){
    if(!isfinite(s[i].top1)||!isfinite(s[i].blank))return -1;
    if(s[i].blank>s[i].top1){prev=-1;continue;}
    int32_t word=s[i].word;
    if(word<0||(size_t)word>=v->count)return -1;
    if(word==prev)continue;
    prev=word;const char*t=v->word[word];size_t len=strlen(t);
    int starts_space=japanese&&len>=3&&(unsigned char)t[0]==0xe2&&
        (unsigned char)t[1]==0x96&&(unsigned char)t[2]==0x81;
    if(starts_space){t+=3;len-=3;}
    size_t sep=japanese?(starts_space&&used?1u:0u):(used?1u:0u);
    if(used>=cap||len>cap-1-used||sep>cap-1-used-len)return -1;
    if(sep)out[used++]=' ';
    memcpy(out+used,t,len);used+=len;out[used]=0;
  }
  return used>(size_t)INT_MAX?-1:(int)used;
}
static inline int cs_eg2_ctc_decode(const cs_eg2_score*s,size_t n,
    const cs_eg2_vocab*v,char*out,size_t cap){
  return cs_eg2_ctc_decode_lang(s,n,v,out,cap,0);
}

#endif /* CATS_SPEECH_H */

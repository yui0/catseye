//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

/* Optional GPU GEMM for the EmbeddingGemma 2 audio tower (build with
 * -DCATS_GPU, link -lEGL -lGLESv2).
 *
 *   C[n,out] = X[n,in] . W[out,in]^T (+ bias[out])      float32
 *
 * Runs as an OpenGL ES 3.1 compute shader on a headless EGL device (no
 * window, no GLFW), so any Mesa/vendor driver that exposes GLES 3.1 works.
 * Software rasterizers (llvmpipe, softpipe, swrast) are skipped: the CPU
 * kernels are faster than those.  Weights are cached on the GPU by pointer
 * until cs_gpu_forget_weights(); every failure disables the backend and the
 * caller falls back to the CPU kernels, so results are always produced.
 *
 *   CATS_GPU=off|auto|N     (auto is the default; N = EGL device index)
 */
#ifndef CATS_GPU_H
#define CATS_GPU_H
#ifdef CATS_GPU

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CS_GPU_TILE 128             /* output tile per work group (both axes) */
#define CS_GPU_BK 16                /* K slab staged in shared memory */
#define CS_GPU_CACHE_SLOTS 48
#define CS_GPU_CACHE_BYTES ((size_t)1200u<<20)

#ifdef CATS_GPU_PROFILE
#include <time.h>
static double cs_gpu_now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,&t);
	return t.tv_sec+t.tv_nsec*1e-9;
}
static double cs_gpu_t[4];
static double cs_gpu_flop;
#define CS_GPU_TIC(i) double tic_##i=cs_gpu_now()
#define CS_GPU_TOC(i,k) (cs_gpu_t[k]+=cs_gpu_now()-tic_##i)
#else
#define CS_GPU_TIC(i)
#define CS_GPU_TOC(i,k)
#endif
typedef struct {
	const void*ptr;
	size_t bytes;
	GLuint buf,aux;                 /* aux: per-block scales of the int8 layout */
	int raw;                        /* 1: row-major copy for the GEMV kernel, 0: tile-packed for the GEMM kernel */
} cs_gpu_slot;
typedef struct {
	int state;                      /* 0 untried, 1 ready, -1 unavailable */
	EGLDisplay dpy;
	EGLContext ctx;
	GLuint prog,bx,bc,bbias;
	GLuint vprog[3];                /* GEMV (n=1) kernels: float32, float16, int8 weights */
	GLint v_in[3],v_out[3];
	size_t cap_x,cap_c;
	GLint u_n,u_in,u_out,u_bias;
	cs_gpu_slot slot[CS_GPU_CACHE_SLOTS];
	size_t cached;
	char name[160];
} cs_gpu_state;
static cs_gpu_state cs_gpu_g;

/* 128x128 output tile per work group, 16x16 threads, 8x8 outputs per thread
 * (rows ty*4+i and 64+ty*4+i, columns likewise, so each thread reads four
 * consecutive shared floats twice: conflict-free and vectorizable).  K is staged 16 at a time.  The 64
 * accumulators are separate scalars (generated below): indexing an array
 * makes drivers spill it to scratch memory. */
/* Operands are repacked on upload into [tile of 128 rows][K slab of 16][row][16]
 * so every slab a work group stages is one contiguous 8 KB block.  Reading
 * rows of a [rows][in] matrix instead (64 bytes per row at a power-of-two
 * stride) serialized on a few memory channels: ~2.5x slower on Tahiti. */
#define CS_GPU_SRC_BYTES 24576
static inline int cs_gpu_build_source(char*buf,size_t cap)
{
	size_t n=0;
#define A(...) do{int w_=snprintf(buf+n,cap-n,__VA_ARGS__);if(w_<0||(size_t)w_>=cap-n)return -1;n+=(size_t)w_;}while(0)
	A("#version 310 es\nprecision highp float;\nlayout(local_size_x=16, local_size_y=16) in;\n"
	  "layout(std430, binding=0) readonly buffer XB { vec4 x4[]; };\n"
	  "layout(std430, binding=1) readonly buffer WB { vec4 w4[]; };\n"
	  "layout(std430, binding=2) writeonly buffer CB { float c[]; };\n"
	  "layout(std430, binding=3) readonly buffer BB { float bias[]; };\n"
	  "uniform uint n_, in_, out_, use_bias;\n"
	  "shared float xs[16*128];\nshared float ws[16*128];\n"
	  "uint tx, ty, rowBase, colBase;\n"
	  "void st(uint i, uint j, float v){\n"
	  "  uint r=rowBase+(i<4u?ty*4u+i:64u+ty*4u+i-4u); uint col=colBase+(j<4u?tx*4u+j:64u+tx*4u+j-4u);\n"
	  "  if(r<n_ && col<out_){ if(use_bias!=0u) v+=bias[col]; c[r*out_+col]=v; }\n}\n"
	  "void main(){\n"
	  "  tx=gl_LocalInvocationID.x; ty=gl_LocalInvocationID.y;\n"
	  "  uint tid=ty*16u+tx;\n"
	  "  rowBase=gl_WorkGroupID.y*128u; colBase=gl_WorkGroupID.x*128u;\n");
	for (int i=0; i<8; i++) {
		A("  float ");
		for (int j=0; j<8; j++) {
			A("c%d_%d=0.0%s",i,j,j==7?";\n":", ");
		}
	}
	A("  uint r0=tid>>2, r1=(tid+256u)>>2, kq=(tid&3u)*4u;\n"
	  "  uint nslab=in_>>4;\n"
	  "  uint xt=gl_WorkGroupID.y*nslab*512u, wt=gl_WorkGroupID.x*nslab*512u;\n"
	  "  vec4 xa=x4[xt+tid], xb=x4[xt+tid+256u], wa=w4[wt+tid], wb=w4[wt+tid+256u];\n"
	  "  for(uint s=0u;s<nslab;s++){\n"
	  "    xs[(kq   )*128u+r0]=xa.x; xs[(kq+1u)*128u+r0]=xa.y; xs[(kq+2u)*128u+r0]=xa.z; xs[(kq+3u)*128u+r0]=xa.w;\n"
	  "    xs[(kq   )*128u+r1]=xb.x; xs[(kq+1u)*128u+r1]=xb.y; xs[(kq+2u)*128u+r1]=xb.z; xs[(kq+3u)*128u+r1]=xb.w;\n"
	  "    ws[(kq   )*128u+r0]=wa.x; ws[(kq+1u)*128u+r0]=wa.y; ws[(kq+2u)*128u+r0]=wa.z; ws[(kq+3u)*128u+r0]=wa.w;\n"
	  "    ws[(kq   )*128u+r1]=wb.x; ws[(kq+1u)*128u+r1]=wb.y; ws[(kq+2u)*128u+r1]=wb.z; ws[(kq+3u)*128u+r1]=wb.w;\n"
	  "    barrier();\n"
	  "    if(s+1u<nslab){\n"            /* prefetch the next slab while this one is computed */
	  "      uint o=(s+1u)*512u+tid;\n"
	  "      xa=x4[xt+o]; xb=x4[xt+o+256u]; wa=w4[wt+o]; wb=w4[wt+o+256u];\n"
	  "    }\n"
	  "    for(uint kk=0u;kk<16u;kk++){\n");
	for (int i=0; i<8; i++) {
		A("      float a%d=xs[kk*128u+ty*4u+%du];\n",i,i<4?i:64+i-4);
	}
	for (int j=0; j<8; j++) {
		A("      float b%d=ws[kk*128u+tx*4u+%du];\n",j,j<4?j:64+j-4);
	}
	for (int i=0; i<8; i++)for (int j=0; j<8; j++) {
			A("      c%d_%d+=a%d*b%d;\n",i,j,i,j);
		}
	A("    }\n    barrier();\n  }\n");
	for (int i=0; i<8; i++)for (int j=0; j<8; j++) {
			A("  st(%du,%du,c%d_%d);\n",i,j,i,j);
		}
	A("}\n");
#undef A
	return 0;
}

/* Percentage of the rows given to the GPU when the CPU shares the work
 * ($CATS_GPU_SHARE, default 100 = GPU only; lower values give the CPU a share, which was slower here). */
static inline size_t cs_gpu_share(void)
{
	static int cached=-1;
	if (cached<0) {
		const char*e=getenv("CATS_GPU_SHARE");
		int v=e?atoi(e):100;
		cached=v<1?1:v>100?100:v;
	}
	return (size_t)cached;
}
static inline int cs_gpu_ready(void)
{
	return cs_gpu_g.state==1;
}
static inline const char*cs_gpu_name(void)
{
	return cs_gpu_g.state==1?cs_gpu_g.name:"(none)";
}

static inline void cs_gpu_forget_weights(void)
{
	if (cs_gpu_g.state!=1) {
		return;
	}
	for (int i=0; i<CS_GPU_CACHE_SLOTS; i++)
		if (cs_gpu_g.slot[i].buf) {
			glDeleteBuffers(1,&cs_gpu_g.slot[i].buf);
			if (cs_gpu_g.slot[i].aux) {
				glDeleteBuffers(1,&cs_gpu_g.slot[i].aux);
			}
			cs_gpu_g.slot[i]=(cs_gpu_slot) {
				0
			};
		}
	cs_gpu_g.cached=0;
}
static inline void cs_gpu_term(void)
{
	cs_gpu_state*g=&cs_gpu_g;
#ifdef CATS_GPU_PROFILE
	if (g->state==1) {
		fprintf(stderr,"cats-gpu: pack+upload %.2fs, compute %.2fs (%.0f GFLOPS), readback %.2fs\n",cs_gpu_t[0],cs_gpu_t[1],cs_gpu_t[1]>0?cs_gpu_flop/cs_gpu_t[1]/1e9:0.0,cs_gpu_t[2]);
	}
#endif
	if (g->state==1) {
		cs_gpu_forget_weights();
		if (g->bx) {
			glDeleteBuffers(1,&g->bx);
		}
		if (g->bc) {
			glDeleteBuffers(1,&g->bc);
		}
		if (g->bbias) {
			glDeleteBuffers(1,&g->bbias);
		}
		if (g->prog) {
			glDeleteProgram(g->prog);
		}
		for (int i=0; i<3; i++)
			if (g->vprog[i]) {
				glDeleteProgram(g->vprog[i]);
			}
		eglMakeCurrent(g->dpy,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
		eglDestroyContext(g->dpy,g->ctx);
		eglTerminate(g->dpy);
	}
	memset(g,0,sizeof(*g));
	g->state=-1;
}
static inline int cs_gpu_compile(cs_gpu_state*g)
{
	GLuint sh=glCreateShader(GL_COMPUTE_SHADER);
	static char source[CS_GPU_SRC_BYTES];
	if (cs_gpu_build_source(source,sizeof(source))) {
		return -1;
	}
	const char*src=source;
	glShaderSource(sh,1,&src,NULL);
	glCompileShader(sh);
	GLint ok=0;
	glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
	if (!ok) {
		char log[1024];
		glGetShaderInfoLog(sh,sizeof(log),NULL,log);
		fprintf(stderr,"cats-gpu: shader: %s\n",log);
		glDeleteShader(sh);
		return -1;
	}
	g->prog=glCreateProgram();
	glAttachShader(g->prog,sh);
	glLinkProgram(g->prog);
	glDeleteShader(sh);
	glGetProgramiv(g->prog,GL_LINK_STATUS,&ok);
	if (!ok) {
		fprintf(stderr,"cats-gpu: link failed\n");
		return -1;
	}
	g->u_n=glGetUniformLocation(g->prog,"n_");
	g->u_in=glGetUniformLocation(g->prog,"in_");
	g->u_out=glGetUniformLocation(g->prog,"out_");
	g->u_bias=glGetUniformLocation(g->prog,"use_bias");
	/* GEMV (one input row): a work group of 64 threads (one wave) per output row;
	 * the threads stride over the row with wide loads, then reduce in shared memory.
	 * Weights stay row-major, so no repacking and no 128-row tile padding.  The
	 * decode step is bound by weight bandwidth, so the float16 variant (weights
	 * rounded to half precision on upload) moves half the bytes. */
	static const char vhead[]=
	    "#version 310 es\nprecision highp float;\nlayout(local_size_x=64) in;\n"
	    "layout(std430, binding=0) readonly buffer XB { vec4 x4[]; };\n"
	    "layout(std430, binding=2) writeonly buffer CB { float c[]; };\n"
	    "uniform uint in_, out_;\nshared float red[64];\n";
	static const char vmain[]=
	    "  uint t=gl_LocalInvocationID.x;\n"
	    "  uint row=gl_WorkGroupID.y*32768u+gl_WorkGroupID.x;\n"
	    "  if(row>=out_) return;\n"
	    "  vec4 a=vec4(0.0);\n"
	    "  LOOP\n"
	    "  red[t]=a.x+a.y+a.z+a.w; barrier();\n"
	    "  for(uint s=32u;s>0u;s>>=1){ if(t<s) red[t]+=red[t+s]; barrier(); }\n"
	    "  if(t==0u) c[row]=red[0];\n}\n";
	static const char*const vbody[3]= {
		"layout(std430, binding=1) readonly buffer WB { vec4 w4[]; };\nvoid main(){\n",
		"layout(std430, binding=1) readonly buffer WB { uvec4 w8[]; };\nvoid main(){\n",
		/* int8: 16 weights per uvec4 with one float scale each (like Q8_0, 20 bytes per 16 weights) */
		"layout(std430, binding=1) readonly buffer WB { uvec4 w16[]; };\n"
		"layout(std430, binding=3) readonly buffer SB { float sc[]; };\n"
		"vec4 q4(uint u){ int i=int(u); return vec4(float(bitfieldExtract(i,0,8)),float(bitfieldExtract(i,8,8)),float(bitfieldExtract(i,16,8)),float(bitfieldExtract(i,24,8))); }\n"
		"void main(){\n"
	};
	static const char*const vloop[3]= {
		"uint n4=in_>>2u; for(uint k=t; k<n4; k+=64u) a+=w4[row*n4+k]*x4[k];\n",
		"uint n8=in_>>3u; for(uint k=t; k<n8; k+=64u){ uvec4 u=w8[row*n8+k];\n"
		"    a+=vec4(unpackHalf2x16(u.x),unpackHalf2x16(u.y))*x4[2u*k]+vec4(unpackHalf2x16(u.z),unpackHalf2x16(u.w))*x4[2u*k+1u]; }\n",
		"uint n16=in_>>4u; for(uint k=t; k<n16; k+=64u){ uvec4 u=w16[row*n16+k];\n"
		"    a+=sc[row*n16+k]*(q4(u.x)*x4[4u*k]+q4(u.y)*x4[4u*k+1u]+q4(u.z)*x4[4u*k+2u]+q4(u.w)*x4[4u*k+3u]); }\n"
	};
	for (int v=0; v<3; v++) {
		static char vs_src[2048];
		const char*loop=strstr(vmain,"LOOP");
		snprintf(vs_src,sizeof(vs_src),"%s%s%.*s%s%s",vhead,vbody[v],(int)(loop-vmain),vmain,vloop[v],loop+4);
		GLuint vs=glCreateShader(GL_COMPUTE_SHADER);
		const char*vp=vs_src;
		glShaderSource(vs,1,&vp,NULL);
		glCompileShader(vs);
		glGetShaderiv(vs,GL_COMPILE_STATUS,&ok);
		if (ok) {
			g->vprog[v]=glCreateProgram();
			glAttachShader(g->vprog[v],vs);
			glLinkProgram(g->vprog[v]);
			glGetProgramiv(g->vprog[v],GL_LINK_STATUS,&ok);
			if (ok) {
				g->v_in[v]=glGetUniformLocation(g->vprog[v],"in_");
				g->v_out[v]=glGetUniformLocation(g->vprog[v],"out_");
			} else {
				glDeleteProgram(g->vprog[v]);
				g->vprog[v]=0;
			}
		}
		glDeleteShader(vs);
	}
	return 0;
}
/* want: -1 auto (first hardware device), >=0 explicit EGL device index. */
static inline int cs_gpu_open(int want)
{
	cs_gpu_state*g=&cs_gpu_g;
	PFNEGLQUERYDEVICESEXTPROC qd=(PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
	PFNEGLGETPLATFORMDISPLAYEXTPROC gp=(PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
	EGLDeviceEXT dev[16];
	EGLint nd=0;
	if (!qd||!gp||!qd(16,dev,&nd)||nd<1) {
		return -1;
	}
	for (int i=0; i<nd; i++) {
		if (want>=0&&i!=want) {
			continue;
		}
		EGLDisplay d=gp(EGL_PLATFORM_DEVICE_EXT,dev[i],NULL);
		EGLint maj,min;
		if (d==EGL_NO_DISPLAY||!eglInitialize(d,&maj,&min)) {
			continue;
		}
		eglBindAPI(EGL_OPENGL_ES_API);
		EGLConfig cfg;
		EGLint nc=0;
		EGLint ca[]= {EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT,EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_NONE};
		eglChooseConfig(d,ca,&cfg,1,&nc);
		EGLint cx[]= {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
		EGLContext c=eglCreateContext(d,nc?cfg:EGL_NO_CONFIG_KHR,EGL_NO_CONTEXT,cx);
		if (c==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,c)) {
			if (c!=EGL_NO_CONTEXT) {
				eglDestroyContext(d,c);
			}
			eglTerminate(d);
			continue;
		}
		const char*rend=(const char*)glGetString(GL_RENDERER);
		if (want<0&&rend&&(strstr(rend,"llvmpipe")||strstr(rend,"softpipe")||strstr(rend,"swrast"))) {
			eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
			eglDestroyContext(d,c);
			eglTerminate(d);
			continue;
		}
		GLint shared=0;
		glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE,&shared);
		if (shared<2*CS_GPU_BK*CS_GPU_TILE*4) {
			eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
			eglDestroyContext(d,c);
			eglTerminate(d);
			continue;
		}
		g->dpy=d;
		g->ctx=c;
		snprintf(g->name,sizeof(g->name),"%s",rend?rend:"GPU");
		if (cs_gpu_compile(g)) {
			eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
			eglDestroyContext(d,c);
			eglTerminate(d);
			memset(g,0,sizeof(*g));
			continue;
		}
		glGenBuffers(1,&g->bx);
		glGenBuffers(1,&g->bc);
		glGenBuffers(1,&g->bbias);
		g->state=1;
		return 0;
	}
	return -1;
}
/* spec: NULL/"auto" (default; honors $CATS_GPU), "off", or an EGL device index. */
static inline int cs_gpu_init(const char*spec)
{
	if (cs_gpu_g.state!=0) {
		return cs_gpu_g.state==1?0:-1;
	}
	if (!spec) {
		spec=getenv("CATS_GPU");
	}
	if (spec&&(!strcmp(spec,"off")||!strcmp(spec,"0")||!strcmp(spec,"cpu"))) {
		cs_gpu_g.state=-1;
		return -1;
	}
	int want=(spec&&*spec&&strcmp(spec,"auto"))?atoi(spec):-1;
	if (cs_gpu_open(want)) {
		memset(&cs_gpu_g,0,sizeof(cs_gpu_g));
		cs_gpu_g.state=-1;
		return -1;
	}
	return 0;
}
static inline void cs_gpu_disable(const char*why)
{
	fprintf(stderr,"cats-gpu: %s; falling back to the CPU\n",why);
	cs_gpu_term();
}
static size_t cs_gpu_packed_floats(size_t rows,size_t in)
{
	return ((rows+CS_GPU_TILE-1)/CS_GPU_TILE)*CS_GPU_TILE*in;
}
/* dst[tile][slab][r][16] <- src[row][in]; rows past the end are zero. */
static void cs_gpu_pack(float*dst,const float*src,size_t rows,size_t in)
{
	const size_t tiles=(rows+CS_GPU_TILE-1)/CS_GPU_TILE,slabs=in/CS_GPU_BK;
	#pragma omp parallel for schedule(static)
	for (long t=0; t<(long)(tiles*slabs); t++) {
		size_t tile=(size_t)t/slabs,slab=(size_t)t%slabs;
		float*d=dst+(size_t)t*CS_GPU_TILE*CS_GPU_BK;
		for (size_t r=0; r<CS_GPU_TILE; r++) {
			size_t row=tile*CS_GPU_TILE+r;
			if (row<rows) {
				memcpy(d+r*CS_GPU_BK,src+row*in+slab*CS_GPU_BK,CS_GPU_BK*sizeof(float));
			} else {
				memset(d+r*CS_GPU_BK,0,CS_GPU_BK*sizeof(float));
			}
		}
	}
}
/* float32 -> IEEE half, round to nearest even (no F16C needed) */
static inline uint16_t cs_gpu_half(float f)
{
	union {
		float f;
		uint32_t u;
	} v= {f};
	uint32_t x=v.u,sign=(x>>16)&0x8000u;
	int e=(int)((x>>23)&0xff)-127+15;
	uint32_t m=x&0x7fffffu;
	if (((x>>23)&0xff)==0xff) {
		return (uint16_t)(sign|0x7c00u|(m?0x200u:0));
	}
	if (e>=31) {
		return (uint16_t)(sign|0x7c00u);
	}
	if (e<=0) {
		if (e<-10) {
			return (uint16_t)sign;
		}
		m|=0x800000u;
		uint32_t sh=(uint32_t)(14-e),r=(m>>sh)+(((m>>(sh-1))&1u)&&((m&((1u<<(sh-1))-1))||((m>>sh)&1u)));
		return (uint16_t)(sign|r);
	}
	uint32_t r=((uint32_t)e<<10)|(m>>13);
	if ((m&0x1000u)&&((m&0xfffu)||(r&1u))) {
		r++;
	}
	return (uint16_t)(sign|r);
}
static void cs_gpu_to_half(uint16_t*dst,const float*src,size_t n)
{
	#pragma omp parallel for schedule(static)
	for (long i=0; i<(long)n; i++) {
		dst[i]=cs_gpu_half(src[i]);
	}
}
/* raw: 0 tile-packed float32 (GEMM), 1 row-major float32, 2 row-major float16, 3 row-major int8 + scales (GEMV) */
static inline GLuint cs_gpu_weights(const float*ptr,size_t rows,size_t in,int raw,GLuint*aux)
{
	cs_gpu_state*g=&cs_gpu_g;
	const size_t bytes=rows*in*sizeof(float);   /* cache key: the unpacked size */
	int freeslot=-1;
	for (int i=0; i<CS_GPU_CACHE_SLOTS; i++) {
		if (g->slot[i].buf&&g->slot[i].ptr==ptr&&g->slot[i].bytes==bytes&&g->slot[i].raw==raw) {
			if (aux) {
				*aux=g->slot[i].aux;
			}
			return g->slot[i].buf;
		}
		if (!g->slot[i].buf&&freeslot<0) {
			freeslot=i;
		}
	}
	const size_t pbytes=raw==3?bytes/4:raw==2?bytes/2:raw?bytes:cs_gpu_packed_floats(rows,in)*sizeof(float);
	if (freeslot<0||g->cached+pbytes>CS_GPU_CACHE_BYTES) {
		cs_gpu_forget_weights();
		freeslot=0;
	}
	void*tmp=raw==1?NULL:malloc(pbytes);
	if (raw!=1&&!tmp) {
		return 0;
	}
	GLuint ab=0;
	if (raw==3) {
		/* int8 with one scale per 16 weights: q=round(w/s), s=max|w|/127 */
		const size_t nb=rows*in/16;
		float*sc=(float*)malloc(nb*sizeof(float));
		if (!sc) {
			free(tmp);
			return 0;
		}
		#pragma omp parallel for schedule(static)
		for (long j=0; j<(long)nb; j++) {
			const float*p=ptr+(size_t)j*16;
			float m=0;
			for (int i=0; i<16; i++) {
				float a=p[i]<0?-p[i]:p[i];
				m=a>m?a:m;
			}
			float s=m/127.0f,r=m>0?127.0f/m:0;
			sc[j]=s;
			for (int i=0; i<16; i++) {
				float q=p[i]*r;
				((int8_t*)tmp)[(size_t)j*16+i]=(int8_t)(q<0?q-0.5f:q+0.5f);
			}
		}
		glGenBuffers(1,&ab);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER,ab);
		glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)(nb*sizeof(float)),sc,GL_STATIC_DRAW);
		free(sc);
		g->cached+=nb*sizeof(float);
	} else if (raw==2) {
		cs_gpu_to_half((uint16_t*)tmp,ptr,rows*in);
	} else if (tmp) {
		cs_gpu_pack((float*)tmp,ptr,rows,in);
	}
	GLuint b;
	glGenBuffers(1,&b);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,b);
	glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)pbytes,raw==1?(const void*)ptr:tmp,GL_STATIC_DRAW);
	free(tmp);
	g->slot[freeslot]=(cs_gpu_slot) {
		ptr,bytes,b,ab,raw
	};
	g->cached+=pbytes;
	if (aux) {
		*aux=ab;
	}
	return b;
}
/* usage matters on radeonsi: *_DRAW with STATIC keeps the buffer in VRAM (the
 * kernel re-reads X many times), STREAM_READ puts the result in host-visible
 * memory for the read-back. */
static inline void cs_gpu_grow(GLuint buf,size_t*cap,size_t bytes,GLenum usage)
{
	if (*cap>=bytes) {
		return;
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,buf);
	glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)bytes,NULL,usage);
	*cap=bytes;
}
/* The dispatch is asynchronous: cs_gpu_gemm_begin uploads and launches the
 * kernel and returns at once, so the caller can compute other rows on the CPU
 * meanwhile; cs_gpu_gemm_end then waits and copies the result into c.
 * Both return 0 on success; any non-zero result means "use the CPU kernel"
 * (after a failed _begin nothing is pending). */
static size_t cs_gpu_pending_bytes;
static inline int cs_gpu_gemm_begin(const float*x,const float*w,size_t n,
                                    size_t in,size_t out,const float*bias)
{
	cs_gpu_state*g=&cs_gpu_g;
	if (g->state!=1||!n||!in||!out||in%CS_GPU_BK||n>0x7fffffffu/(in?in:1)||
	                out>0x7fffffffu/(in?in:1)||n*out>0x7fffffffu) {
		return -1;
	}
	if (eglGetCurrentContext()!=g->ctx&&!eglMakeCurrent(g->dpy,EGL_NO_SURFACE,EGL_NO_SURFACE,g->ctx)) {
		return -1;
	}
	CS_GPU_TIC(0);
	const size_t xb=cs_gpu_packed_floats(n,in)*sizeof(float),cb=n*out*sizeof(float);
	GLuint bw=cs_gpu_weights(w,out,in,0,NULL);
	float*xp=(float*)malloc(xb);
	if (!bw||!xp) {
		free(xp);
		return -1;
	}
	cs_gpu_pack(xp,x,n,in);
	cs_gpu_grow(g->bx,&g->cap_x,xb,GL_STATIC_DRAW);
	cs_gpu_grow(g->bc,&g->cap_c,cb,GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,g->bx);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,(GLsizeiptr)xb,xp);
	free(xp);
	if (bias) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER,g->bbias);
		glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)(out*sizeof(float)),bias,GL_DYNAMIC_DRAW);
	} else {
		static const float zero[4]= {0};
		glBindBuffer(GL_SHADER_STORAGE_BUFFER,g->bbias);
		glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof(zero),zero,GL_DYNAMIC_DRAW);
	}
	CS_GPU_TOC(0,0);
	glUseProgram(g->prog);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,g->bx);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,bw);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,g->bc);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,g->bbias);
	glUniform1ui(g->u_n,(GLuint)n);
	glUniform1ui(g->u_in,(GLuint)in);
	glUniform1ui(g->u_out,(GLuint)out);
	glUniform1ui(g->u_bias,bias?1u:0u);
	glDispatchCompute((GLuint)((out+CS_GPU_TILE-1)/CS_GPU_TILE),
	                  (GLuint)((n+CS_GPU_TILE-1)/CS_GPU_TILE),1);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);
	glFlush();                      /* start now; the CPU overlaps from here */
	cs_gpu_pending_bytes=cb;
#ifdef CATS_GPU_PROFILE
	cs_gpu_flop+=2.0*n*in*out;
#endif
	return 0;
}
static inline int cs_gpu_gemm_end(float*c)
{
	cs_gpu_state*g=&cs_gpu_g;
	CS_GPU_TIC(2);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,g->bc);
	const void*m=glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,(GLsizeiptr)cs_gpu_pending_bytes,GL_MAP_READ_BIT);
	if (!m||glGetError()!=GL_NO_ERROR) {
		if (m) {
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		cs_gpu_disable("compute dispatch failed");
		return -1;
	}
	memcpy(c,m,cs_gpu_pending_bytes);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	CS_GPU_TOC(2,2);
	return 0;
}
/* c[out] = W[out,in] . x[in] for a single input row (the LLM decode step).
 * Returns 0 on success; non-zero means "use the CPU kernel". */
static inline int cs_gpu_gemv(float*c,const float*x,const float*w,size_t in,size_t out)
{
	cs_gpu_state*g=&cs_gpu_g;
	/* $CATS_GPU_WEIGHTS = f32 | f16 (default) | i8: how the weights are stored on the GPU.
	 * The decode step is bound by weight bandwidth, so the smaller formats are faster. */
	static int fmt=-1;
	if (fmt<0) {
		const char*e=getenv("CATS_GPU_WEIGHTS");
		fmt=e&&!strcmp(e,"f32")?0:e&&(!strcmp(e,"i8")||!strcmp(e,"int8"))?2:1;
	}
	int v=fmt;
	if (v==2&&(in%16||!g->vprog[2])) {
		v=1;
	}
	if (v==1&&(in%8||!g->vprog[1])) {
		v=0;
	}
	if (g->state!=1||!g->vprog[v]||!in||!out||in%4||out*in>0x7fffffffu||out>32768u*65535u) {
		return -1;
	}
	if (eglGetCurrentContext()!=g->ctx&&!eglMakeCurrent(g->dpy,EGL_NO_SURFACE,EGL_NO_SURFACE,g->ctx)) {
		return -1;
	}
	GLuint aux=0,bw=cs_gpu_weights(w,out,in,v+1,&aux);
	if (!bw) {
		return -1;
	}
	const size_t cb=out*sizeof(float);
	cs_gpu_grow(g->bc,&g->cap_c,cb,GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER,g->bx);
	if (g->cap_x<in*sizeof(float)) {
		glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)(in*sizeof(float)),x,GL_STATIC_DRAW);
		g->cap_x=in*sizeof(float);
	} else {
		glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,(GLsizeiptr)(in*sizeof(float)),x);
	}
	glUseProgram(g->vprog[v]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,g->bx);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,bw);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,g->bc);
	if (v==2) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,aux);
	}
	glUniform1ui(g->v_in[v],(GLuint)in);
	glUniform1ui(g->v_out[v],(GLuint)out);
	glDispatchCompute((GLuint)(out<32768u?out:32768u),(GLuint)((out+32767u)/32768u),1);
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);
	cs_gpu_pending_bytes=cb;
	return cs_gpu_gemm_end(c);
}
/* Whole product on the GPU (begin + end). */
static inline int cs_gpu_gemm_nt(float*c,const float*x,const float*w,size_t n,
                                 size_t in,size_t out,const float*bias)
{
	return cs_gpu_gemm_begin(x,w,n,in,out,bias)||cs_gpu_gemm_end(c);
}
#endif /* CATS_GPU */
#endif /* CATS_GPU_H */

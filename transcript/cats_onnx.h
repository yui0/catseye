//---------------------------------------------------------
//	Cat's eye
//
//		©2026 Yuichiro Nakada
//---------------------------------------------------------

#ifndef CATS_ONNX_H
#define CATS_ONNX_H

/* ========================================================================
 * ONNX protobuf metadata inspection
 * ======================================================================== */

/*
 * ONNX inspector section - dependency-free ONNX graph/operator inspector for Cat's eye.
 * This parses protobuf wire framing; NO protobuf/ONNX/ONNXRuntime required.
 * This does NOT execute ONNX inference or decompress external_data weights.
 * Only ModelProto.graph(7), GraphProto.node(1)/initializer(5), and
 * NodeProto.op_type(4) are inspected.
 */
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

#define COI_MAX_OPS 128
typedef struct {
	char name[96];
	unsigned count;
} coi_op;
typedef struct {
	coi_op ops[COI_MAX_OPS];
	unsigned types, nodes, initializers;
} coi_stats;

typedef struct {
	FILE *fp;
	long end;
	int error;
} coi_reader;

static inline int coi_varint(coi_reader *p,uint64_t *value)
{
	uint64_t v=0;
	for (unsigned i=0; i<10; i++) {
		int c=fgetc(p->fp);
		if (c==EOF||ftell(p->fp)>p->end) {
			return -1;
		}
		if (i==9 && (c&0xfe)) {
			return -1;
		}
		v|=(uint64_t)(c&127)<<(i*7);
		if (!(c&128)) {
			*value=v;
			return 0;
		}
	}
	return -1;
}
static inline int coi_next(coi_reader *p,unsigned *field,unsigned *wire,long *limit)
{
	long pos=ftell(p->fp);
	if (pos<0||pos>p->end) {
		return -1;
	}
	if (pos==p->end) {
		return 0;
	}
	uint64_t key=0,len=0;
	if (coi_varint(p,&key)||!(key>>3)||key>>3>0x1fffffff) {
		return -1;
	}
	*field=(unsigned)(key>>3);
	*wire=(unsigned)(key&7);
	pos=ftell(p->fp);
	if (pos<0) {
		return -1;
	}
	if (*wire==2) {
		if (coi_varint(p,&len)) {
			return -1;
		}
		pos=ftell(p->fp);
		if (pos<0||len>(uint64_t)(p->end-pos)) {
			return -1;
		}
		*limit=pos+(long)len;
	} else if (*wire==0) {
		if (coi_varint(p,&len)) {
			return -1;
		}
		*limit=ftell(p->fp);
	} else if (*wire==1||*wire==5) {
		len=*wire==1?8:4;
		if (len>(uint64_t)(p->end-pos)) {
			return -1;
		}
		*limit=pos+(long)len;
	} else {
		return -1;
	}
	return 1;
}
static inline int coi_skip(coi_reader *p,long end)
{
	return (end>=0 && end<=p->end && fseek(p->fp,end,SEEK_SET)==0)?0:-1;
}
static inline int coi_node(coi_reader *p,long end,coi_stats *stats)
{
	long parent=p->end;
	p->end=end;
	char op[96]= {0};
	int status=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int r=coi_next(p,&field,&wire,&stop);
		if (r==0) {
			status=0;
			break;
		}
		if (r<0) {
			break;
		}
		if (field==4&&wire==2) {
			long here=ftell(p->fp);
			size_t size=(size_t)(stop-here);
			if (size>=sizeof(op)) {
				size=sizeof(op)-1;
			}
			if (fread(op,1,size,p->fp)!=size) {
				break;
			}
			op[size]=0;
		}
		if (coi_skip(p,stop)) {
			break;
		}
	}
	p->end=parent;
	if (status) {
		return -1;
	}
	stats->nodes++;
	if (!op[0]) {
		strcpy(op,"(unnamed)");
	}
	unsigned i=0;
	for (; i<stats->types; i++)if (!strcmp(stats->ops[i].name,op)) {
			break;
		}
	if (i==stats->types) {
		if (stats->types==COI_MAX_OPS) {
			return -1;
		}
		memcpy(stats->ops[i].name,op,strlen(op)+1);
		stats->types++;
	}
	stats->ops[i].count++;
	return 0;
}
static inline int coi_graph(coi_reader *p,long end,coi_stats *stats)
{
	long parent=p->end;
	p->end=end;
	int status=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int r=coi_next(p,&field,&wire,&stop);
		if (r==0) {
			status=0;
			break;
		}
		if (r<0) {
			break;
		}
		if (field==1&&wire==2) {
			if (coi_node(p,stop,stats)) {
				break;
			}
		}
		if (field==5&&wire==2) {
			stats->initializers++;
		}
		if (coi_skip(p,stop)) {
			break;
		}
	}
	p->end=parent;
	return status;
}
static inline int coi_inspect(const char *path,coi_stats *result)
{
	if (!path||!result) {
		return -1;
	}
	memset(result,0,sizeof(*result));
	FILE *fp=fopen(path,"rb");
	if (!fp) {
		return -1;
	}
	int ok=-1,found=0;
	if (fseek(fp,0,SEEK_END)) {
		goto done;
	}
	long end=ftell(fp);
	if (end<=0||fseek(fp,0,SEEK_SET)) {
		goto done;
	}
	coi_reader p= {fp,end,0};
	for (;;) {
		unsigned field,wire;
		long stop;
		int r=coi_next(&p,&field,&wire,&stop);
		if (r==0) {
			ok=found?0:-1;
			break;
		}
		if (r<0) {
			break;
		}
		if (field==7&&wire==2) {
			if (coi_graph(&p,stop,result)) {
				break;
			}
			found=1;
		}
		if (coi_skip(&p,stop)) {
			break;
		}
	}
done:fclose(fp);
	return ok;
}
static inline void coi_print(const coi_stats *s,FILE *out)
{
	fprintf(out,"ONNX graph: %u nodes, %u initializers, %u operator types\n",
	        s->nodes,s->initializers,s->types);
	for (unsigned i=0; i<s->types; i++) {
		fprintf(out,"  %-32s %u\n",s->ops[i].name,s->ops[i].count);
	}
	fprintf(out,"This is metadata inspection only; no ONNX operators are executed.\n");
}


/* ========================================================================
 * Runtime CPU dispatch and AVX/AVX2/AVX-512/VNNI kernels
 * ======================================================================== */

/* CPU SIMD section — runtime-dispatched CPU kernels for Cat's eye.
 * C11, libc only; GCC/Clang x86 builds AVX2/FMA and AVX-512 variants with
 * function target attributes, WITHOUT -mavx2/-mavx512f. Other targets use C.
 * MPL-2.0.  Avoid fast-math: ONNX NaN/Inf behavior must not silently change.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <math.h>
#include <limits.h>
/* CPU feature detection (CATS_SIMD_X86, CATS_AVX*_FN, cats_cpu_simd_level) and the shared
 * dot/GEMM kernels live in the generic header. Inference-only mode skips the training code. */
#define CATS_INFER_ONLY
#define CATS_NO_ALLOC_MACROS
#include "../catseye.h"
#undef CATS_INFER_ONLY
#undef CATS_NO_ALLOC_MACROS

/* A[M,K], B[K,N] row-major -> C[M,N], overwrites C. */
static inline void cats_sgemm_scalar(const float*a,const float*b,float*c,
                                     int64_t m,int64_t n,int64_t k)
{
	for (int64_t i=0; i<m; i++) {
		float *o=c+i*n;
		for (int64_t j=0; j<n; j++) {
			o[j]=0;
		}
		for (int64_t p=0; p<k; p++) {
			const float av=a[i*k+p],*r=b+p*n;
			for (int64_t j=0; j<n; j++) {
				o[j]+=av*r[j];
			}
		}
	}
}
#if CATS_SIMD_X86
CATS_AVX_FN static void cats_sgemm_avx(const float*a,const float*b,float*c,
                                       int64_t m,int64_t n,int64_t k)
{
	for (int64_t i=0; i<m; i+=4) {
		int rows=(int)(m-i<4?m-i:4);
		int64_t j=0;
		for (; j+8<=n; j+=8) {
			__m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
			__m256 s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
			for (int64_t p=0; p<k; p++) {
				__m256 w=_mm256_loadu_ps(b+p*n+j);
				s0=_mm256_add_ps(s0,_mm256_mul_ps(_mm256_set1_ps(a[(i+0)*k+p]),w));
				if (rows>1) {
					s1=_mm256_add_ps(s1,_mm256_mul_ps(_mm256_set1_ps(a[(i+1)*k+p]),w));
				}
				if (rows>2) {
					s2=_mm256_add_ps(s2,_mm256_mul_ps(_mm256_set1_ps(a[(i+2)*k+p]),w));
				}
				if (rows>3) {
					s3=_mm256_add_ps(s3,_mm256_mul_ps(_mm256_set1_ps(a[(i+3)*k+p]),w));
				}
			}
			_mm256_storeu_ps(c+(i+0)*n+j,s0);
			if (rows>1) {
				_mm256_storeu_ps(c+(i+1)*n+j,s1);
			}
			if (rows>2) {
				_mm256_storeu_ps(c+(i+2)*n+j,s2);
			}
			if (rows>3) {
				_mm256_storeu_ps(c+(i+3)*n+j,s3);
			}
		}
		for (int q=0; q<rows; q++)for (int64_t jj=j; jj<n; jj++) {
				float sum=0;
				for (int64_t p=0; p<k; p++) {
					sum+=a[(i+q)*k+p]*b[p*n+jj];
				}
				c[(i+q)*n+jj]=sum;
			}
	}
}
CATS_AVX2_FN static void cats_sgemm_avx2(const float*a,const float*b,float*c,
                int64_t m,int64_t n,int64_t k)
{
	for (int64_t i=0; i<m; i+=4) {
		int rows=(int)(m-i<4?m-i:4);
		int64_t j=0;
		for (; j+8<=n; j+=8) {
			__m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
			__m256 s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
			for (int64_t p=0; p<k; p++) {
				__m256 w=_mm256_loadu_ps(b+p*n+j);
				s0=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+0)*k+p]),w,s0);
				if (rows>1) {
					s1=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+1)*k+p]),w,s1);
				}
				if (rows>2) {
					s2=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+2)*k+p]),w,s2);
				}
				if (rows>3) {
					s3=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+3)*k+p]),w,s3);
				}
			}
			_mm256_storeu_ps(c+(i+0)*n+j,s0);
			if (rows>1) {
				_mm256_storeu_ps(c+(i+1)*n+j,s1);
			}
			if (rows>2) {
				_mm256_storeu_ps(c+(i+2)*n+j,s2);
			}
			if (rows>3) {
				_mm256_storeu_ps(c+(i+3)*n+j,s3);
			}
		}
		for (int q=0; q<rows; q++)for (int64_t jj=j; jj<n; jj++) {
				float sum=0;
				for (int64_t p=0; p<k; p++) {
					sum+=a[(i+q)*k+p]*b[p*n+jj];
				}
				c[(i+q)*n+jj]=sum;
			}
	}
}
CATS_AVX512_FN static void cats_sgemm_avx512(const float*a,const float*b,float*c,
                int64_t m,int64_t n,int64_t k)
{
	for (int64_t i=0; i<m; i+=4) {
		int rows=(int)(m-i<4?m-i:4);
		int64_t j=0;
		for (; j+16<=n; j+=16) {
			__m512 s0=_mm512_setzero_ps(),s1=_mm512_setzero_ps();
			__m512 s2=_mm512_setzero_ps(),s3=_mm512_setzero_ps();
			for (int64_t p=0; p<k; p++) {
				__m512 w=_mm512_loadu_ps(b+p*n+j);
				s0=_mm512_fmadd_ps(_mm512_set1_ps(a[(i+0)*k+p]),w,s0);
				if (rows>1) {
					s1=_mm512_fmadd_ps(_mm512_set1_ps(a[(i+1)*k+p]),w,s1);
				}
				if (rows>2) {
					s2=_mm512_fmadd_ps(_mm512_set1_ps(a[(i+2)*k+p]),w,s2);
				}
				if (rows>3) {
					s3=_mm512_fmadd_ps(_mm512_set1_ps(a[(i+3)*k+p]),w,s3);
				}
			}
			_mm512_storeu_ps(c+(i+0)*n+j,s0);
			if (rows>1) {
				_mm512_storeu_ps(c+(i+1)*n+j,s1);
			}
			if (rows>2) {
				_mm512_storeu_ps(c+(i+2)*n+j,s2);
			}
			if (rows>3) {
				_mm512_storeu_ps(c+(i+3)*n+j,s3);
			}
		}
		/* AVX2 handles remainder in 8-column units, with scalar final tail. */
		for (; j+8<=n; j+=8) {
			__m256 s0=_mm256_setzero_ps(),s1=_mm256_setzero_ps();
			__m256 s2=_mm256_setzero_ps(),s3=_mm256_setzero_ps();
			for (int64_t p=0; p<k; p++) {
				__m256 w=_mm256_loadu_ps(b+p*n+j);
				s0=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+0)*k+p]),w,s0);
				if (rows>1) {
					s1=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+1)*k+p]),w,s1);
				}
				if (rows>2) {
					s2=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+2)*k+p]),w,s2);
				}
				if (rows>3) {
					s3=_mm256_fmadd_ps(_mm256_set1_ps(a[(i+3)*k+p]),w,s3);
				}
			}
			_mm256_storeu_ps(c+(i+0)*n+j,s0);
			if (rows>1) {
				_mm256_storeu_ps(c+(i+1)*n+j,s1);
			}
			if (rows>2) {
				_mm256_storeu_ps(c+(i+2)*n+j,s2);
			}
			if (rows>3) {
				_mm256_storeu_ps(c+(i+3)*n+j,s3);
			}
		}
		for (int q=0; q<rows; q++)for (int64_t jj=j; jj<n; jj++) {
				float sum=0;
				for (int64_t p=0; p<k; p++) {
					sum+=a[(i+q)*k+p]*b[p*n+jj];
				}
				c[(i+q)*n+jj]=sum;
			}
	}
}
#endif
static inline void cats_sgemm_f32(const float*a,const float*b,float*c,
                                  int64_t m,int64_t n,int64_t k)
{
#if CATS_SIMD_X86
	int level=cats_cpu_simd_level();
	if (level==3 && n>=16) {
		cats_sgemm_avx512(a,b,c,m,n,k);
		return;
	}
	if (level>=2 && n>=8) {
		cats_sgemm_avx2(a,b,c,m,n,k);
		return;
	}
	if (level>=1 && n>=8) {
		cats_sgemm_avx(a,b,c,m,n,k);
		return;
	}
#endif
	cats_sgemm_scalar(a,b,c,m,n,k);
}

#if CATS_SIMD_X86
static void cats_simd_elementwise_private(float*,const float*,const float*,uint64_t,int);
#endif
/* A and B in the same row-major packed format as above. */
static inline void cats_f32_elementwise(float*dst,const float*a,const float*b,
                                        uint64_t count,int op)
{
	/* op: 0=add,1=sub,2=mul,3=div. Division stays scalar because NaN/Inf
	   and rounding are easiest to preserve without approximate reciprocals. */
#if CATS_SIMD_X86
	if (op<=2 && cats_cpu_simd_level()>=1) {
		/* Implemented below, dispatch after forward declaration. */
		cats_simd_elementwise_private(dst,a,b,count,op);
		return;
	}
#endif
	for (uint64_t i=0; i<count; i++) {
		dst[i]=op==0?a[i]+b[i]:op==1?a[i]-b[i]:op==2?a[i]*b[i]:a[i]/b[i];
	}
}
#if CATS_SIMD_X86
CATS_AVX_FN static void cats_elementwise_avx(float*d,const float*a,const float*b,uint64_t n,int op)
{
	uint64_t i=0;
	for (; i+8<=n; i+=8) {
		__m256 x=_mm256_loadu_ps(a+i),y=_mm256_loadu_ps(b+i);
		__m256 z=op==0?_mm256_add_ps(x,y):op==1?_mm256_sub_ps(x,y):_mm256_mul_ps(x,y);
		_mm256_storeu_ps(d+i,z);
	}
	for (; i<n; i++) {
		d[i]=op==0?a[i]+b[i]:op==1?a[i]-b[i]:a[i]*b[i];
	}
}
/* no ABI trick: always call the target-attributed kernel explicitly */
static void cats_simd_elementwise_private(float*d,const float*a,const float*b,uint64_t n,int op)
{
	cats_elementwise_avx(d,a,b,n,op);
}
#endif


/* Eight output columns of (A-z_a) @ (B-z_b), B is packed INT8/UINT8.
   K<=32768 and legal quantized zero-points bound the int32 accumulator:
   abs(sum) <= 32768*255*255 < INT32_MAX. No saturation arithmetic.
   Called only after higher-level shape/type/range checks. */
#if CATS_SIMD_X86
CATS_AVX2_FN static void cats_q8_row8_avx2(const float*arow,const uint8_t*b,
                int b_signed,int64_t ksize,int64_t ncols,int64_t col,
                int32_t za,int32_t zb,int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	__m256i vz=_mm256_set1_epi32(zb);
	for (int64_t k=0; k<ksize; k++) {
		__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(b+k*ncols+col));
		__m256i wt=b_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
		wt=_mm256_sub_epi32(wt,vz);
		__m256i av=_mm256_set1_epi32((int32_t)arow[k]-za);
		acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,wt));
	}
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
}
#endif
/* ONNX Relu uses the historical interpreter rule x>0 ? x : 0; this also
   maps NaN and negative zero to +0, matching _mm*_max_ps(x, zero). */
#if CATS_SIMD_X86
CATS_AVX_FN static void cats_relu_avx(float*dst,const float*src,uint64_t n)
{
	uint64_t i=0;
	__m256 zero=_mm256_setzero_ps();
	for (; i+8<=n; i+=8) {
		_mm256_storeu_ps(dst+i,_mm256_max_ps(_mm256_loadu_ps(src+i),zero));
	}
	for (; i<n; i++) {
		dst[i]=src[i]>0?src[i]:0;
	}
}
CATS_AVX512_FN static void cats_relu_avx512(float*dst,const float*src,uint64_t n)
{
	uint64_t i=0;
	__m512 zero=_mm512_setzero_ps();
	for (; i+16<=n; i+=16) {
		_mm512_storeu_ps(dst+i,_mm512_max_ps(_mm512_loadu_ps(src+i),zero));
	}
	for (; i<n; i++) {
		dst[i]=src[i]>0?src[i]:0;
	}
}
#endif
static inline void cats_f32_relu(float*dst,const float*src,uint64_t n)
{
#if CATS_SIMD_X86
	int mode=cats_cpu_simd_level();
	if (mode==3 && n>=16) {
		cats_relu_avx512(dst,src,n);
		return;
	}
	if (mode>=1 && n>=8) {
		cats_relu_avx(dst,src,n);
		return;
	}
#endif
	for (uint64_t i=0; i<n; i++) {
		dst[i]=src[i]>0?src[i]:0;
	}
}

/* 1D depthwise convolution: one channel per filter, stride=1/dilation=1.
   Only interior positions are vectorized, boundaries use zero-padded scalar
   reference arithmetic. Returns 1 if SIMD was used, 0 for scalar fallback.
   Input/output each point to a single independent [width] channel row. */
#if CATS_SIMD_X86
CATS_AVX_FN static void cats_dwconv1d_avx(float*y,const float*x,const float*w,
                int64_t iw,int64_t ow,int64_t kw,int64_t pad,float bias)
{
	int64_t t=0;
	int64_t begin=pad>0?pad:0;
	int64_t stop=iw-kw+pad+1;
	if (stop>ow) {
		stop=ow;
	}
	if (begin>ow) {
		begin=ow;
	}
	if (stop<begin) {
		stop=begin;
	}
	for (; t<begin; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
	for (; t+8<=stop; t+=8) {
		__m256 v=_mm256_set1_ps(bias);
		for (int64_t k=0; k<kw; k++)v=_mm256_add_ps(v,_mm256_mul_ps(
			                                      _mm256_loadu_ps(x+t-pad+k),_mm256_set1_ps(w[k])));
		_mm256_storeu_ps(y+t,v);
	}
	for (; t<ow; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
}
CATS_AVX2_FN static void cats_dwconv1d_avx2(float*y,const float*x,const float*w,
                int64_t iw,int64_t ow,int64_t kw,int64_t pad,float bias)
{
	int64_t t=0;
	int64_t begin=pad>0?pad:0;
	int64_t stop=iw-kw+pad+1;
	if (stop>ow) {
		stop=ow;
	}
	if (begin>ow) {
		begin=ow;
	}
	if (stop<begin) {
		stop=begin;
	}
	for (; t<begin; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
	for (; t+8<=stop; t+=8) {
		__m256 v=_mm256_set1_ps(bias);
		for (int64_t k=0; k<kw; k++)v=_mm256_fmadd_ps(
			                                      _mm256_loadu_ps(x+t-pad+k),_mm256_set1_ps(w[k]),v);
		_mm256_storeu_ps(y+t,v);
	}
	for (; t<ow; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
}
CATS_AVX512_FN static void cats_dwconv1d_avx512(float*y,const float*x,const float*w,
                int64_t iw,int64_t ow,int64_t kw,int64_t pad,float bias)
{
	int64_t t=0;
	int64_t begin=pad>0?pad:0;
	int64_t stop=iw-kw+pad+1;
	if (stop>ow) {
		stop=ow;
	}
	if (begin>ow) {
		begin=ow;
	}
	if (stop<begin) {
		stop=begin;
	}
	for (; t<begin; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
	for (; t+16<=stop; t+=16) {
		__m512 v=_mm512_set1_ps(bias);
		for (int64_t k=0; k<kw; k++)v=_mm512_fmadd_ps(
			                                      _mm512_loadu_ps(x+t-pad+k),_mm512_set1_ps(w[k]),v);
		_mm512_storeu_ps(y+t,v);
	}
	for (; t<ow; t++) {
		double z=bias;
		for (int64_t k=0; k<kw; k++) {
			int64_t p=t-pad+k;
			if (p>=0&&p<iw) {
				z+=(double)x[p]*w[k];
			}
		}
		y[t]=(float)z;
	}
}
#endif
static inline int cats_dwconv1d_f32(float*y,const float*x,const float*w,
                                    int64_t iw,int64_t ow,int64_t kw,int64_t pad,float bias)
{
#if CATS_SIMD_X86
	int mode=cats_cpu_simd_level();
	if (mode==3 && iw>=16 && ow>=16 && kw>=1) {
		cats_dwconv1d_avx512(y,x,w,iw,ow,kw,pad,bias);
		return 1;
	}
	if (mode>=2 && iw>=8 && ow>=8 && kw>=1) {
		cats_dwconv1d_avx2(y,x,w,iw,ow,kw,pad,bias);
		return 1;
	}
	if (mode>=1 && iw>=8 && ow>=8 && kw>=1) {
		cats_dwconv1d_avx(y,x,w,iw,ow,kw,pad,bias);
		return 1;
	}
#endif
	(void)y;
	(void)x;
	(void)w;
	(void)iw;
	(void)ow;
	(void)kw;
	(void)pad;
	(void)bias;
	return 0;
}

/* Stage 17: Packed 8-bit matrix multiplication.
   AVX-512 VNNI processes 16 output columns and 4 reduction samples per
   dpbusd instruction.  Signed A is offset by 128 to make it unsigned;
   unsigned B is offset by -128 to make it signed. Corrections restore the
   original ONNX zero-point arithmetic EXACTLY (using 64-bit scalars).
   Returns 1 only when the complete result was produced; 0 requests the
   existing AVX2/scalar fallback. Inputs are not mutated. */
#if CATS_SIMD_X86
#define CATS_VNNI_FN __attribute__((target("avx512f,avx512vnni")))
CATS_VNNI_FN static int cats_q8_matmul_vnni_impl(const float *a,
                const uint8_t *b,int64_t m,int64_t k,int64_t n,
                int a_signed,int b_signed,int32_t za,int32_t zb,int32_t *out)
{
	if (k<=0||k>32768||n<16||(n&15)||m<4||k>INT64_MAX-3) {
		return 0;
	}
	const int64_t blocks=(k+3)/4,kk=blocks*4;
	if ((uint64_t)kk>SIZE_MAX/16 ||
	                (uint64_t)m>SIZE_MAX/(uint64_t)blocks/sizeof(uint32_t) ||
	                (uint64_t)m>SIZE_MAX/sizeof(int64_t)) {
		return 0;
	}
	uint8_t *packed=(uint8_t*)malloc((size_t)kk*16);
	uint32_t *apack=(uint32_t*)malloc((size_t)m*(size_t)blocks*sizeof(*apack));
	int64_t *suma=(int64_t*)calloc((size_t)m,sizeof(*suma));
	if (!packed||!apack||!suma) {
		free(packed);
		free(apack);
		free(suma);
		return 0;
	}
	const int64_t za_u=(int64_t)za+(a_signed?128:0);
	const int64_t zb_s=(int64_t)zb-(b_signed?0:128);
	/* Convert A once, not once per output-column tile. */
	for (int64_t i=0; i<m; i++)for (int64_t p=0; p<k; p++) {
			float q=a[i*k+p];
			if (!isfinite(q)||q<(a_signed?-128.f:0.f)||q>(a_signed?127.f:255.f) ||
			                q!=(float)(int32_t)q) {
				free(suma);
				free(apack);
				free(packed);
				return 0;
			}
			int32_t av=(int32_t)q+(a_signed?128:0);
			if (!(p&3)) {
				apack[i*blocks+(p>>2)]=0;
			}
			apack[i*blocks+(p>>2)]|=((uint32_t)av&255u)<<(8*(p&3));
			suma[i]+=av;
		}
	for (int64_t j=0; j<n; j+=16) {
		int64_t sum_b[16]= {0};
		memset(packed,0,(size_t)kk*16);
		for (int64_t p=0; p<k; p++)for (int col=0; col<16; col++) {
				uint8_t v=b[p*n+j+col];
				int32_t sb=b_signed?(int32_t)(int8_t)v:(int32_t)v-128;
				sum_b[col]+=sb;
				packed[(p>>2)*64+col*4+(p&3)]=(uint8_t)(int8_t)sb;
			}
		for (int64_t i=0; i<m; i++) {
			__m512i acc=_mm512_setzero_si512();
			const uint32_t *row=apack+i*blocks;
			for (int64_t block=0; block<blocks; block++) {
				__m512i av=_mm512_set1_epi32((int32_t)row[block]);
				__m512i bv=_mm512_loadu_si512((const void*)(packed+block*64));
				acc=_mm512_dpbusd_epi32(acc,av,bv);
			}
			/* Explicit 64-byte alignment also keeps GCC+ASan target-function
			   stack instrumentation compatible with wide stores. */
			__attribute__((aligned(64))) int32_t raw[16];
			_mm512_store_si512((void*)raw,acc);
			for (int col=0; col<16; col++) {
				int64_t v=(int64_t)raw[col]-zb_s*suma[i]-za_u*sum_b[col]+k*za_u*zb_s;
				if (v<INT32_MIN||v>INT32_MAX) {
					free(suma);
					free(apack);
					free(packed);
					return 0;
				}
				out[i*n+j+col]=(int32_t)v;
			}
		}
	}
	free(suma);
	free(apack);
	free(packed);
	return 1;
}
CATS_AVX2_FN static void cats_q8_row8_avx2_zcols(const float *arow,
                const uint8_t *b,int b_signed,int64_t ksize,int64_t ncols,int64_t col,
                int32_t za,const int32_t *zb,int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	__m256i vz=_mm256_loadu_si256((const __m256i*)(const void*)zb);
	for (int64_t k=0; k<ksize; k++) {
		__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(b+k*ncols+col));
		__m256i wt=b_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
		wt=_mm256_sub_epi32(wt,vz);
		__m256i av=_mm256_set1_epi32((int32_t)arow[k]-za);
		acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,wt));
	}
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
}
/* 1-D depthwise convolution, contiguous interior 8-output tiles.
   Quantized inputs can be unpacked floats or original packed bytes. The
   kernel performs integer accumulation only; requantization remains in the
   ONNX operator so rounding/bias semantics are shared with scalar code. */
CATS_AVX2_FN static void cats_q8_dwconv8_avx2(const float *xf,
                const uint8_t *xq,const uint8_t *wq,int x_signed,int w_signed,
                int64_t wi,int64_t kw,int64_t start,int64_t pad,
                int32_t zx,int32_t zw,int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	for (int64_t k=0; k<kw; k++) {
		int64_t off=start+k-pad;
		__m256i av;
		if (xq) {
			__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(xq+off));
			av=x_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
		} else {
			__m256 f=_mm256_loadu_ps(xf+off);
			av=_mm256_cvttps_epi32(f);
		}
		av=_mm256_sub_epi32(av,_mm256_set1_epi32(zx));
		int32_t w=w_signed?(int32_t)(int8_t)wq[k]:(int32_t)wq[k];
		acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,_mm256_set1_epi32(w-zw)));
	}
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
	(void)wi;
}
/* Stage 20: 2-D depthwise 8-output tile. The caller guarantees that the
 * complete KH x KW receptive field for all eight output pixels is inside
 * the input. Supports packed INT8/UINT8 activations and unpacked FP32
 * activation storage. Uses 64-bit scalar checks outside this hot path.
 * KH*KW<=32768 keeps every partial int32 accumulator in range.
 */
CATS_AVX2_FN static void cats_q8_dwconv2d8_avx2(const float *xf,
                const uint8_t *xq,const uint8_t *wq,int x_signed,int w_signed,
                int64_t wi,int64_t kh,int64_t kw,int64_t in_h,int64_t in_w,
                int32_t zx,int32_t zw,int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	const __m256i vzx=_mm256_set1_epi32(zx);
	for (int64_t ky=0; ky<kh; ky++)for (int64_t kx=0; kx<kw; kx++) {
			int64_t off=(in_h+ky)*wi+in_w+kx;
			__m256i av;
			if (xq) {
				__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(xq+off));
				av=x_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
			} else {
				__m256 f=_mm256_loadu_ps(xf+off);
				av=_mm256_cvttps_epi32(f);
			}
			av=_mm256_sub_epi32(av,vzx);
			int32_t w=w_signed?(int32_t)(int8_t)wq[ky*kw+kx]:(int32_t)wq[ky*kw+kx];
			acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,_mm256_set1_epi32(w-zw)));
		}
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
}
#endif


/* Stage 21: grouped 2-D convolution with >=2 input channels per group.
 * Eight horizontally adjacent output pixels are computed together. The
 * caller checks that the full receptive field is interior, strides and
 * dilations are 1, and CG*KH*KW <= 32768 (no signed int32 overflow).
 * xf/xq points to the first input channel of this batch/group; wq to the
 * first weight of this output channel. NCHW and OIHW layout. */
#if CATS_SIMD_X86
CATS_AVX2_FN static void cats_q8_groupconv2d8_avx2(const float *xf,
                const uint8_t *xq,const uint8_t *wq,int x_signed,int w_signed,
                int64_t cg,int64_t hi,int64_t wi,int64_t kh,int64_t kw,
                int64_t in_h,int64_t in_w,int32_t zx,int32_t zw,int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	const __m256i vzx=_mm256_set1_epi32(zx);
	const int64_t plane=hi*wi,filter=kh*kw;
	for (int64_t ic=0; ic<cg; ic++) {
		const int64_t xb=ic*plane,wb=ic*filter;
		for (int64_t ky=0; ky<kh; ky++)for (int64_t kx=0; kx<kw; kx++) {
				int64_t off=xb+(in_h+ky)*wi+in_w+kx;
				__m256i av;
				if (xq) {
					__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(xq+off));
					av=x_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
				} else {
					__m256 v=_mm256_loadu_ps(xf+off);
					av=_mm256_cvttps_epi32(v);
				}
				av=_mm256_sub_epi32(av,vzx);
				int32_t wt=w_signed?(int32_t)(int8_t)wq[wb+ky*kw+kx]:
				           (int32_t)wq[wb+ky*kw+kx];
				acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,
				                     _mm256_set1_epi32(wt-zw)));
			}
	}
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
}
#endif

/* Stage 22: prepare one output channel's 2D grouped convolution weights.
 * Centering the weights once and correcting the activation zero point only
 * after accumulation avoids two subtractions in every inner MAC iteration.
 * The caller bounds CG*KH*KW <= 32768; q8 coefficients fit int16 exactly.
 * Returns the sum of centered coefficients for activation zero correction.
 */
static inline int32_t cats_q8_groupconv_pack_weights(
        const uint8_t *w,int w_signed,int32_t zw,int64_t count,int16_t *packed)
{
	int32_t sum=0;
	for (int64_t i=0; i<count; i++) {
		int32_t v=(w_signed?(int32_t)(int8_t)w[i]:(int32_t)w[i])-zw;
		packed[i]=(int16_t)v;
		sum+=v;
	}
	return sum;
}
#if CATS_SIMD_X86
CATS_AVX2_FN static void cats_q8_groupconv2d8_packed_avx2(
        const float *xf,const uint8_t *xq,const int16_t *wc,int x_signed,
        int64_t cg,int64_t hi,int64_t wi,int64_t kh,int64_t kw,
        int64_t in_h,int64_t in_w,int32_t zx,int32_t coeff_sum,
        int32_t out[8])
{
	__m256i acc=_mm256_setzero_si256();
	const int64_t plane=hi*wi,filter=kh*kw;
	for (int64_t ic=0; ic<cg; ic++) {
		int64_t xb=ic*plane,wb=ic*filter;
		for (int64_t ky=0; ky<kh; ky++)for (int64_t kx=0; kx<kw; kx++) {
				int64_t off=xb+(in_h+ky)*wi+in_w+kx;
				__m256i av;
				if (xq) {
					__m128i bytes=_mm_loadl_epi64((const __m128i*)(const void*)(xq+off));
					av=x_signed?_mm256_cvtepi8_epi32(bytes):_mm256_cvtepu8_epi32(bytes);
				} else {
					av=_mm256_cvttps_epi32(_mm256_loadu_ps(xf+off));
				}
				acc=_mm256_add_epi32(acc,_mm256_mullo_epi32(av,
				                     _mm256_set1_epi32(wc[wb+ky*kw+kx])));
			}
	}
	/* Worst-case |zx * coeff_sum| and |result| are <= 255*255*32768,
	   strictly within signed int32. */
	acc=_mm256_sub_epi32(acc,_mm256_set1_epi32(zx*coeff_sum));
	_mm256_storeu_si256((__m256i*)(void*)out,acc);
}
#endif

/* Stage 24: AVX-512 VNNI grouped 2-D convolution microkernel.
 * A is packed four unsigned activation bytes per uint32; B is packed as
 * [ceil(K/4), 16 output channels, 4 signed weight bytes]. Padding channels
 * have zero weights. The caller applies affine zero point corrections with
 * 64-bit arithmetic; raw dpbusd accumulators fit signed 32-bit for K<=32768.
 * Only call after runtime VNNI feature detection. */
#if CATS_SIMD_X86
CATS_VNNI_FN static void cats_q8_conv2d_vnni_dot(
        const uint32_t *a,const uint8_t *b,int64_t blocks,int32_t raw[16])
{
	__m512i acc=_mm512_setzero_si512();
	for (int64_t p=0; p<blocks; p++) {
		__m512i x=_mm512_set1_epi32((int32_t)a[p]);
		__m512i w=_mm512_loadu_si512((const void*)(b+(size_t)p*64));
		acc=_mm512_dpbusd_epi32(acc,x,w);
	}
	_mm512_storeu_si512((void*)raw,acc);
}
#endif

/* Stage 18: optional session-owned immutable VNNI B matrix cache.
 * The ONNX initializer remains in its native 8-bit form. Each [K,N]
 * slice is packed once as [N/16,ceil(K/4),16,4] for dpbusd, with
 * precomputed signed column sums for exact zero-point correction.
 * Never attach this cache to mutable activations. Lifetime: cx_tensor.
 * No process-global cache: separate sessions cannot share stale pointers.
 */
typedef struct cats_q8_vnni_weights {
	uint8_t *tiles;
	int64_t *column_sums;
	int64_t matrices,k,n,blocks;
	int b_signed;
} cats_q8_vnni_weights;
static inline void cats_q8_vnni_weights_free(void *ptr)
{
	cats_q8_vnni_weights *w=(cats_q8_vnni_weights*)ptr;
	if (w) {
		free(w->tiles);
		free(w->column_sums);
		free(w);
	}
}
static inline cats_q8_vnni_weights *cats_q8_vnni_weights_create(
        const uint8_t *b,int64_t matrices,int64_t k,int64_t n,int b_signed)
{
	if (!b||matrices<1||k<1||k>32768||n<16||(n&15)||
	                matrices>INT64_MAX/n||k>INT64_MAX-3) {
		return NULL;
	}
	int64_t blocks=(k+3)/4;
	uint64_t nm=(uint64_t)matrices*(uint64_t)n;
	if (nm>(uint64_t)SIZE_MAX/(uint64_t)blocks/4 ||
	                nm>(uint64_t)SIZE_MAX/sizeof(int64_t) ||
	                nm>(uint64_t)SIZE_MAX/(uint64_t)k) {
		return NULL;
	}
	size_t nbytes=(size_t)nm*(size_t)blocks*4;
	/* Prevent an accidentally huge second copy of multi-GB model weights.
	   Falls back transparently to the normal uncached kernel. */
#ifndef CATS_VNNI_CACHE_BYTES_MAX
#define CATS_VNNI_CACHE_BYTES_MAX (64u*1024u*1024u)
#endif
	if (nbytes>CATS_VNNI_CACHE_BYTES_MAX) {
		return NULL;
	}
	cats_q8_vnni_weights *w=(cats_q8_vnni_weights*)calloc(1,sizeof(*w));
	if (!w) {
		return NULL;
	}
	w->tiles=(uint8_t*)calloc(nbytes,1);
	w->column_sums=(int64_t*)calloc((size_t)nm,sizeof(int64_t));
	if (!w->tiles||!w->column_sums) {
		cats_q8_vnni_weights_free(w);
		return NULL;
	}
	w->matrices=matrices;
	w->k=k;
	w->n=n;
	w->blocks=blocks;
	w->b_signed=b_signed;
	const size_t matrix_bytes=(size_t)n*(size_t)blocks*4;
	for (int64_t batch=0; batch<matrices; batch++) {
		const uint8_t *src=b+(size_t)batch*(size_t)k*(size_t)n;
		uint8_t *dst=w->tiles+(size_t)batch*matrix_bytes;
		int64_t *sum=w->column_sums+(size_t)batch*(size_t)n;
		for (int64_t j=0; j<n; j++)for (int64_t t=0; t<k; t++) {
				uint8_t q=src[t*n+j];
				int32_t v=b_signed?(int32_t)(int8_t)q:(int32_t)q-128;
				sum[j]+=v;
				size_t at=((size_t)(j/16)*(size_t)blocks+(size_t)(t/4))*64+
				          (size_t)(j&15)*4+(size_t)(t&3);
				dst[at]=(uint8_t)(int8_t)v;
			}
	}
	return w;
}
#if CATS_SIMD_X86
CATS_VNNI_FN static int cats_q8_matmul_vnni_cached_impl(
        const float *a,const cats_q8_vnni_weights *w,int64_t batch,int64_t m,
        int a_signed,int32_t za,int32_t zb,int32_t *out)
{
	if (!a||!w||!out||batch<0||batch>=w->matrices||m<1||
	                w->k<1||w->k>32768||w->n<16||(w->n&15)||
	                (uint64_t)m>SIZE_MAX/(uint64_t)w->blocks/sizeof(uint32_t)||
	                (uint64_t)m>SIZE_MAX/sizeof(int64_t)) {
		return 0;
	}
	const int64_t k=w->k,n=w->n,blocks=w->blocks;
	uint32_t *apack=(uint32_t*)calloc((size_t)m*(size_t)blocks,sizeof(*apack));
	int64_t *suma=(int64_t*)calloc((size_t)m,sizeof(*suma));
	if (!apack||!suma) {
		free(apack);
		free(suma);
		return 0;
	}
	int64_t za_u=(int64_t)za+(a_signed?128:0);
	int64_t zb_s=(int64_t)zb-(w->b_signed?0:128);
	/* Validate ALL activations before producing output; fallback remains safe. */
	for (int64_t i=0; i<m; i++)for (int64_t p=0; p<k; p++) {
			float q=a[i*k+p];
			if (!isfinite(q)||q<(a_signed?-128.f:0.f)||q>(a_signed?127.f:255.f)||
			                q!=(float)(int32_t)q) {
				free(apack);
				free(suma);
				return 0;
			}
			uint32_t av=(uint32_t)((int32_t)q+(a_signed?128:0));
			apack[i*blocks+(p>>2)]|=av<<(8*(p&3));
			suma[i]+=av;
		}
	size_t matrix_bytes=(size_t)n*(size_t)blocks*4;
	const uint8_t *packed=w->tiles+(size_t)batch*matrix_bytes;
	const int64_t *sumb=w->column_sums+(size_t)batch*(size_t)n;
	/* Four-row microkernel: B is loaded ONCE for four output rows.
	   Reduces memory traffic and loop overhead compared with the original
	   one-row-at-a-time VNNI kernel. */
	for (int64_t col0=0; col0<n; col0+=16) {
		const uint8_t *tile=packed+(size_t)(col0/16)*(size_t)blocks*64;
		for (int64_t i=0; i<m; i+=4) {
			int rows=(int)(m-i<4?m-i:4);
			__m512i acc0=_mm512_setzero_si512(),acc1=acc0,acc2=acc0,acc3=acc0;
			const uint32_t *r0=apack+i*blocks;
			for (int64_t z=0; z<blocks; z++) {
				__m512i bv=_mm512_loadu_si512((const void*)(tile+(size_t)z*64));
				acc0=_mm512_dpbusd_epi32(acc0,_mm512_set1_epi32((int32_t)r0[z]),bv);
				if (rows>1)acc1=_mm512_dpbusd_epi32(acc1,
					                                    _mm512_set1_epi32((int32_t)r0[blocks+z]),bv);
				if (rows>2)acc2=_mm512_dpbusd_epi32(acc2,
					                                    _mm512_set1_epi32((int32_t)r0[2*blocks+z]),bv);
				if (rows>3)acc3=_mm512_dpbusd_epi32(acc3,
					                                    _mm512_set1_epi32((int32_t)r0[3*blocks+z]),bv);
			}
			__attribute__((aligned(64))) int32_t raw[4][16];
			_mm512_store_si512((void*)raw[0],acc0);
			if (rows>1) {
				_mm512_store_si512((void*)raw[1],acc1);
			}
			if (rows>2) {
				_mm512_store_si512((void*)raw[2],acc2);
			}
			if (rows>3) {
				_mm512_store_si512((void*)raw[3],acc3);
			}
			for (int row=0; row<rows; row++)for (int t=0; t<16; t++) {
					int64_t v=(int64_t)raw[row][t]-zb_s*suma[i+row]-
					          za_u*sumb[col0+t]+k*za_u*zb_s;
					if (v<INT32_MIN||v>INT32_MAX) {
						free(apack);
						free(suma);
						return 0;
					}
					out[(i+row)*n+col0+t]=(int32_t)v;
				}
		}
	}
	free(apack);
	free(suma);
	return 1;
}
#endif
static inline int cats_q8_matmul_vnni_cached(const float *a,
                const cats_q8_vnni_weights *w,int64_t batch,int64_t m,
                int a_signed,int32_t za,int32_t zb,int32_t *out)
{
#if CATS_SIMD_X86
	__builtin_cpu_init();
	if (cats_cpu_simd_level()>=3 && __builtin_cpu_supports("avx512vnni")) {
		return cats_q8_matmul_vnni_cached_impl(a,w,batch,m,a_signed,za,zb,out);
	}
#endif
	(void)a;
	(void)w;
	(void)batch;
	(void)m;
	(void)a_signed;
	(void)za;
	(void)zb;
	(void)out;
	return 0;
}

static inline int cats_cpu_has_vnni(void)
{
#if CATS_SIMD_X86
	return cats_cpu_simd_level()>=3 && __builtin_cpu_supports("avx512vnni");
#else
	return 0;
#endif
}
static inline int cats_q8_matmul_vnni(const float*a,const uint8_t*b,
                                      int64_t m,int64_t k,int64_t n,int a_signed,int b_signed,
                                      int32_t za,int32_t zb,int32_t*out)
{
#if CATS_SIMD_X86
	if (cats_cpu_has_vnni())return cats_q8_matmul_vnni_impl(a,b,m,k,n,
		                               a_signed,b_signed,za,zb,out);
#endif
	(void)a;
	(void)b;
	(void)m;
	(void)k;
	(void)n;
	(void)a_signed;
	(void)b_signed;
	(void)za;
	(void)zb;
	(void)out;
	return 0;
}


/* ===== Graph/tensor runtime and operators ================================= */
/*
 * Graph execution section — small, lazy-loading ONNX subset for Cat's eye.
 * MPL-2.0. C11, libc + libm only. Deliberately NOT a complete ONNX runtime.
 * Parses the protobuf wire format directly; weights loaded only when referenced.
 */
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <inttypes.h>
#include <errno.h>
#include <time.h>

#define CX_DIMS 8
#ifndef CX_IO
#define CX_IO 12
#endif
#define CX_ATTRS 12
#define CX_LIMIT 1000000u

typedef struct {
	char name[160];
	int type, nd;
	int64_t d[CX_DIMS];
	uint64_t count;
	long raw_at;
	uint64_t raw_len;
	int packed_type; /* field 4/5/7 for non-raw encodings */
	int external;
} cx_weight;
typedef struct {
	char name[64];
	int64_t iv, ints[CX_DIMS];
	float fv;
	char sv[64];
	int nints, kind;
	cx_weight tensor;
} cx_attr;
typedef struct {
	char op[64], domain[64];
	char input[CX_IO][160], output[CX_IO][160];
	int ni,no,na;
	cx_attr attrs[CX_ATTRS];
} cx_node;
typedef struct {
	int type;              /* ONNX element type, 0=unknown */
	int nd;                /* -1 when shape not present */
	int64_t d[CX_DIMS];    /* -1 when symbolic/unknown */
	char symbol[CX_DIMS][48];
} cx_value_meta;
#define CX_META 48
typedef struct {
	char key[96], value[160];
} cx_meta;
typedef struct {
	char domain[64];
	int64_t version;
} cx_opset;
/* Forward-use index for the compact interpreter. Keys point to immutable
   node.input[] strings; the graph owns their lifetime. */
typedef struct {
	const char *name;
	size_t last;
} cx_use;
typedef struct {
	FILE *f;
	cx_meta meta[CX_META];
	int nmeta;
	cx_opset opset[12];
	int nopset;
	cx_node *node;
	cx_weight *init;
	size_t nn,nw,cn,cw;
	cx_use *last_use;
	size_t last_use_capacity; /* built lazily once */
	/* Stage 12: once-built initializer name index and packed-weight policy.
	   Entries hold 1-based indices; zero is an empty hash slot. */
	size_t *weight_index;
	size_t weight_index_capacity;
	unsigned char *weight_packed; /* 0=no, 1=yes */
	char input[CX_IO][160], output[CX_IO][160];
	int ninput,noutput;
	cx_value_meta input_meta[CX_IO], output_meta[CX_IO];
	char error[160];
	FILE *trace; /* optional caller-owned trace sink; never opened/closed here */
	const char *trace_name;
	struct cx_profiler *profiler; /* optional borrowed profiler, no runtime allocations */
} cx_graph;
/* Stage 25: low-overhead opt-in, accumulated ONNX node timings.
 * The profiler is owned by its caller and must outlive the attached graph.
 * A monotonic clock is used where available. Profiling adds only a branch to
 * each node in the normal (disabled) execution path. */
typedef struct {
	uint64_t calls, total_ns, min_ns, max_ns, failures;
} cx_profile_stat;
typedef struct cx_profiler {
	cx_profile_stat *stats;
	size_t count;
	const char *name;
} cx_profiler;
static inline uint64_t cx_profile_clock_ns(void)
{
	struct timespec ts= {0};
#if defined(CLOCK_MONOTONIC)
	if (clock_gettime(CLOCK_MONOTONIC,&ts)==0) {
		return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
	}
#endif
#if defined(TIME_UTC)
	if (timespec_get(&ts,TIME_UTC)==TIME_UTC) {
		return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
	}
#endif
	return 0;
}
static inline int cx_profiler_init(cx_profiler*p,cx_graph*g,const char*name)
{
	if (!p||!g||g->profiler||g->nn>SIZE_MAX/sizeof(cx_profile_stat)) {
		return -1;
	}
	memset(p,0,sizeof(*p));
	p->stats=(cx_profile_stat*)calloc(g->nn?g->nn:1,sizeof(cx_profile_stat));
	if (!p->stats) {
		return -1;
	}
	p->count=g->nn;
	p->name=name?name:"model";
	g->profiler=p;
	return 0;
}
static inline void cx_profiler_detach(cx_graph*g)
{
	if (g) {
		g->profiler=NULL;
	}
}
static inline void cx_profiler_free(cx_profiler*p)
{
	if (p) {
		free(p->stats);
		memset(p,0,sizeof(*p));
	}
}
static inline void cx_profiler_record(cx_profiler*p,size_t index,uint64_t start,int failed)
{
	if (!p||!p->stats||index>=p->count) {
		return;
	}
	cx_profile_stat*s=&p->stats[index];
	if (failed) {
		s->failures++;
	}
	if (!start) {
		return;
	}
	uint64_t end=cx_profile_clock_ns();
	if (!end||end<start) {
		return;
	}
	uint64_t delta=end-start;
	if (!s->calls||delta<s->min_ns) {
		s->min_ns=delta;
	}
	if (delta>s->max_ns) {
		s->max_ns=delta;
	}
	s->calls++;
	s->total_ns=UINT64_MAX-s->total_ns<delta?UINT64_MAX:s->total_ns+delta;
}
typedef struct {
	size_t node;
	uint64_t total;
} cx_profile_order;
static int cx_profile_order_cmp(const void*a,const void*b)
{
	const cx_profile_order*x=(const cx_profile_order*)a;
	const cx_profile_order*y=(const cx_profile_order*)b;
	if (x->total!=y->total) {
		return x->total<y->total?1:-1;
	}
	return x->node<y->node?-1:(x->node>y->node?1:0);
}
static inline void cx_profiler_header(FILE*fp)
{
	if (fp) {
		fputs("graph\trank\tnode\top\tcalls\ttotal_ms\tavg_us\tmin_us\tmax_us\tfailures\n",fp);
	}
}
/* Hotspot-first TSV. Deliberately compute the sort only once on report,
 * never in the model execution loop. Output valid even after a failed run. */
static inline int cx_profiler_report(FILE*fp,const cx_graph*g,const cx_profiler*p)
{
	if (!fp||!g||!p||!p->stats||p->count!=g->nn) {
		return -1;
	}
	cx_profile_order *order=(cx_profile_order*)malloc((g->nn?g->nn:1)*sizeof(*order));
	if (!order) {
		return -1;
	}
	for (size_t i=0; i<g->nn; i++) {
		order[i].node=i;
		order[i].total=p->stats[i].total_ns;
	}
	qsort(order,g->nn,sizeof(*order),cx_profile_order_cmp);
	for (size_t k=0; k<g->nn; k++) {
		size_t i=order[k].node;
		const cx_profile_stat*t=&p->stats[i];
		fprintf(fp,"%s\t%zu\t%zu\t%s\t%" PRIu64 "\t%.6f\t%.3f\t%.3f\t%.3f\t%" PRIu64 "\n",
		        p->name,k+1,i,g->node[i].op,t->calls,t->total_ns/1000000.0,
		        t->calls?(t->total_ns/1000.0)/(double)t->calls:0.,
		        t->calls?t->min_ns/1000.0:0.,t->calls?t->max_ns/1000.0:0.,t->failures);
	}
	free(order);
	return ferror(fp)?-1:0;
}

typedef struct {
	char name[160];
	float *v;
	/* Optional compact representation of exclusively integer-matmul weights.
	   qv is allocated only for uint8/int8 initializers in MatMulInteger or
	   QLinearMatMul; normal tensors continue to use float32 storage. */
	uint8_t *qv;
	void *vnni_cache; /* immutable packed B matrix, session-scoped (stage 18) */
	unsigned char is_initializer; /* never prepack mutable activations */
	uint64_t n;
	int nd,type;
	int64_t d[CX_DIMS];
	size_t slot; /* position in cx_env.t, preserved across swap-delete */
} cx_tensor;
typedef struct {
	cx_tensor **t;
	size_t n,cap;
	cx_tensor **index; /* open-addressed name cache, tombstone=1 */
	size_t index_capacity, tombstones;
} cx_env;

static inline void cx_err(cx_graph *g,const char *s)
{
	if (g&& !g->error[0]) {
		snprintf(g->error,sizeof(g->error),"%s",s);
	}
}
static inline int cx_string(coi_reader *r,long end,char *out,size_t cap)
{
	long at=ftell(r->fp);
	if (at<0||end<at||!cap) {
		return -1;
	}
	size_t len=(size_t)(end-at);
	if (len>=cap) {
		return -1;
	}
	if (fread(out,1,len,r->fp)!=len) {
		return -1;
	}
	out[len]=0;
	return 0;
}
static inline int cx_enter(coi_reader *r,long end,long *save)
{
	if (end>r->end || end<ftell(r->fp)) {
		return -1;
	}
	*save=r->end;
	r->end=end;
	return 0;
}
static inline int cx_fixed32(coi_reader *r,long end,float *f)
{
	unsigned char b[4];
	if (end-ftell(r->fp)!=4||fread(b,1,4,r->fp)!=4) {
		return -1;
	}
	uint32_t x=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
	memcpy(f,&x,4);
	return 0;
}
static inline int cx_prev_varint(coi_reader *r,long after,uint64_t *v)
{
	if (after<=0) {
		return -1;
	}
	long at=after-1;
	int c;
	while (at>0 && after-at<10) {
		if (fseek(r->fp,at-1,SEEK_SET)) {
			return -1;
		}
		c=fgetc(r->fp);
		if (c==EOF) {
			return -1;
		}
		if (!(c&128)) {
			break;
		}
		at--;
	}
	if (fseek(r->fp,at,SEEK_SET)) {
		return -1;
	}
	if (coi_varint(r,v)) {
		return -1;
	}
	return ftell(r->fp)==after?0:-1;
}
static inline int cx_tensor_parse(coi_reader *r,long end,cx_weight *w)
{
	memset(w,0,sizeof(*w));
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int result=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int rc=coi_next(r,&field,&wire,&stop);
		if (rc==0) {
			result=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (field==1) { /* dims may be packed */
			if (wire==0) {
				if (w->nd==CX_DIMS) {
					break;
				}
				uint64_t v;
				if (cx_prev_varint(r,stop,&v)) {
					break;
				}
				w->d[w->nd++]=(int64_t)v;
			} else if (wire==2) {
				long old=r->end;
				r->end=stop;
				while (ftell(r->fp)<stop) {
					uint64_t v;
					if (w->nd==CX_DIMS||coi_varint(r,&v)) {
						r->end=old;
						goto done;
					}
					w->d[w->nd++]=(int64_t)v;
				}
				r->end=old;
			} else {
				break;
			}
		} else if (field==2&&wire==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)) {
				break;
			}
			w->type=(int)v;
		} else if (field==8&&wire==2) {
			if (cx_string(r,stop,w->name,sizeof(w->name))) {
				break;
			}
		} else if (field==9&&wire==2) {
			w->raw_at=ftell(r->fp);
			w->raw_len=(uint64_t)(stop-w->raw_at);
		} else if ((field==4||field==5||field==7)&&wire==2) {
			w->packed_type=(int)field;
			w->raw_at=ftell(r->fp);
			w->raw_len=(uint64_t)(stop-w->raw_at);
		} else if (field==13||field==14) {
			w->external=1;
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	if (result==0) {
		w->count=1;
		for (int j=0; j<w->nd; j++) {
			if (w->d[j]<0||w->count>UINT64_MAX/(uint64_t)(w->d[j]?w->d[j]:1)) {
				result=-1;
				break;
			}
			w->count*=w->d[j];
		}
	}
done:r->end=save;
	return result;
}
static inline int cx_attribute_parse(coi_reader *r,long end,cx_attr *a)
{
	memset(a,0,sizeof(*a));
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int ret=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int rc=coi_next(r,&field,&wire,&stop);
		if (!rc) {
			ret=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (field==1&&wire==2) {
			if (cx_string(r,stop,a->name,sizeof(a->name))) {
				break;
			}
		} else if (field==2&&wire==5) {
			if (cx_fixed32(r,stop,&a->fv)) {
				break;
			}
		} else if (field==3&&wire==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)) {
				break;
			}
			a->iv=(int64_t)v;
		} else if (field==4&&wire==2) {
			if (cx_string(r,stop,a->sv,sizeof(a->sv))) {
				break;
			}
		} else if (field==5&&wire==2) {
			if (cx_tensor_parse(r,stop,&a->tensor)) {
				break;
			}
		} else if (field==8&&wire==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)||a->nints==CX_DIMS) {
				break;
			}
			a->ints[a->nints++]=(int64_t)v;
		} else if (field==8&&wire==2) {
			long old=r->end;
			r->end=stop;
			while (ftell(r->fp)<stop) {
				uint64_t v;
				if (a->nints==CX_DIMS||coi_varint(r,&v)) {
					r->end=old;
					goto done_attr;
				}
				a->ints[a->nints++]=(int64_t)v;
			}
			r->end=old;
		} else if (field==20&&wire==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)) {
				break;
			}
			a->kind=(int)v;
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
done_attr:r->end=save;
	return ret;
}
static inline int cx_node_parse(coi_reader *r,long end,cx_node *node)
{
	memset(node,0,sizeof(*node));
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int ret=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int rc=coi_next(r,&field,&wire,&stop);
		if (!rc) {
			ret=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (field==1&&wire==2) {
			if (node->ni==CX_IO||cx_string(r,stop,node->input[node->ni++],160)) {
				break;
			}
		} else if (field==2&&wire==2) {
			if (node->no==CX_IO||cx_string(r,stop,node->output[node->no++],160)) {
				break;
			}
		} else if (field==4&&wire==2) {
			if (cx_string(r,stop,node->op,sizeof(node->op))) {
				break;
			}
		} else if (field==7&&wire==2) {
			if (cx_string(r,stop,node->domain,sizeof(node->domain))) {
				break;
			}
		} else if (field==5&&wire==2) {
			if (node->na==CX_ATTRS||cx_attribute_parse(r,stop,&node->attrs[node->na++])) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return ret;
}
static inline int cx_grow(void **p,size_t *cap,size_t needed,size_t elem)
{
	if (needed>*cap) {
		size_t n=*cap?*cap*2:16;
		if (n<needed) {
			n=needed;
		}
		if (n>CX_LIMIT || elem>SIZE_MAX/n) {
			return -1;
		}
		void *v=realloc(*p,n*elem);
		if (!v) {
			return -1;
		}
		*p=v;
		*cap=n;
	}
	return 0;
}
/* Parse ONNX TypeProto.tensor_type -> TensorShapeProto, retaining dynamic
 * dimension names instead of fabricating concrete dimensions. */
static inline int cx_parse_shape_dim(coi_reader*r,long end,cx_value_meta*m,int axis)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int status=-1;
	m->d[axis]=-1;
	for (;;) {
		unsigned f,w;
		long stop;
		int rc=coi_next(r,&f,&w,&stop);
		if (!rc) {
			status=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (f==1&&w==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)) {
				break;
			}
			m->d[axis]=(int64_t)v;
		}
		if (f==2&&w==2) {
			if (cx_string(r,stop,m->symbol[axis],sizeof(m->symbol[axis]))) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return status;
}
static inline int cx_parse_shape(coi_reader*r,long end,cx_value_meta*m)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int status=-1;
	m->nd=0;
	for (;;) {
		unsigned f,w;
		long stop;
		int rc=coi_next(r,&f,&w,&stop);
		if (!rc) {
			status=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (f==1&&w==2) {
			if (m->nd==CX_DIMS||cx_parse_shape_dim(r,stop,m,m->nd)) {
				break;
			}
			m->nd++;
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return status;
}
static inline int cx_parse_tensor_type(coi_reader*r,long end,cx_value_meta*m)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int status=-1;
	for (;;) {
		unsigned f,w;
		long stop;
		int rc=coi_next(r,&f,&w,&stop);
		if (!rc) {
			status=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (f==1&&w==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)) {
				break;
			}
			m->type=(int)v;
		}
		if (f==2&&w==2) {
			if (cx_parse_shape(r,stop,m)) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return status;
}
static inline int cx_parse_type(coi_reader*r,long end,cx_value_meta*m)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int status=-1;
	for (;;) {
		unsigned f,w;
		long stop;
		int rc=coi_next(r,&f,&w,&stop);
		if (!rc) {
			status=0;
			break;
		}
		if (rc<0) {
			break;
		}
		if (f==1&&w==2) {
			if (cx_parse_tensor_type(r,stop,m)) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return status;
}
static inline int cx_value_info(coi_reader *r,long end,char *out,size_t len,cx_value_meta*m)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int ret=-1;
	if (m) {
		memset(m,0,sizeof(*m));
		m->nd=-1;
	}
	for (;;) {
		unsigned f,w;
		long stop;
		int c=coi_next(r,&f,&w,&stop);
		if (!c) {
			ret=0;
			break;
		}
		if (c<0) {
			break;
		}
		if (f==1&&w==2) {
			if (cx_string(r,stop,out,len)) {
				break;
			}
		}
		if (f==2&&w==2&&m) {
			if (cx_parse_type(r,stop,m)) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return ret;
}
/* ModelProto.metadata_props: repeated StringStringEntryProto field 14.
   Sherpa-ONNX reads decoder context_size/vocab_size here, not by shape. */
static inline int cx_model_metadata_entry(coi_reader *r,long end,cx_meta *entry)
{
	long saved;
	if (cx_enter(r,end,&saved)) {
		return -1;
	}
	memset(entry,0,sizeof(*entry));
	int ok=-1;
	for (;;) {
		unsigned f,w;
		long stop;
		int c=coi_next(r,&f,&w,&stop);
		if (!c) {
			ok=0;
			break;
		}
		if (c<0) {
			break;
		}
		if (f==1&&w==2) {
			if (cx_string(r,stop,entry->key,sizeof(entry->key))) {
				break;
			}
		}
		if (f==2&&w==2) {
			if (cx_string(r,stop,entry->value,sizeof(entry->value))) {
				break;
			}
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=saved;
	return ok;
}
/* ModelProto.opset_import, OperatorSetIdProto {domain, version}.
   NodeProto.domain must be checked before claiming an operator is supported. */
static inline int cx_parse_opset(coi_reader *r,long end,cx_opset *o)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	memset(o,0,sizeof(*o));
	int rc=-1;
	for (;;) {
		unsigned field,wire;
		long stop;
		int next=coi_next(r,&field,&wire,&stop);
		if (!next) {
			rc=0;
			break;
		}
		if (next<0) {
			break;
		}
		if (field==1&&wire==2) {
			if (cx_string(r,stop,o->domain,sizeof(o->domain))) {
				break;
			}
		} else if (field==2&&wire==0) {
			uint64_t v;
			if (cx_prev_varint(r,stop,&v)||v>INT64_MAX) {
				break;
			}
			o->version=(int64_t)v;
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return rc;
}
static inline const char *cx_graph_metadata(const cx_graph *g,const char *key)
{
	if (!g||!key) {
		return NULL;
	}
	for (int i=0; i<g->nmeta; i++)if (!strcmp(g->meta[i].key,key)) {
			return g->meta[i].value;
		}
	return NULL;
}
static inline int cx_graph_metadata_int(const cx_graph *g,const char *key,int *out)
{
	const char *v=cx_graph_metadata(g,key);
	if (!v) {
		return 0;  /* not present */
	}
	if (!*v||*v=='-'||*v=='+') {
		return -1;
	}
	char *end=NULL;
	errno=0;
	long x=strtol(v,&end,10);
	if (errno||*end||x<=0||x>INT_MAX) {
		return -1;
	}
	*out=(int)x;
	return 1;
}
static inline int cx_graph_parse(coi_reader *r,long end,cx_graph *g)
{
	long save;
	if (cx_enter(r,end,&save)) {
		return -1;
	}
	int ret=-1;
	for (;;) {
		unsigned f,w;
		long stop;
		int c=coi_next(r,&f,&w,&stop);
		if (!c) {
			ret=0;
			break;
		}
		if (c<0) {
			break;
		}
		if (f==1&&w==2) {
			if (cx_grow((void**)&g->node,&g->cn,g->nn+1,sizeof(cx_node))) {
				break;
			}
			if (cx_node_parse(r,stop,&g->node[g->nn])) {
				break;
			}
			g->nn++;
		} else if (f==5&&w==2) {
			if (cx_grow((void**)&g->init,&g->cw,g->nw+1,sizeof(cx_weight))) {
				break;
			}
			if (cx_tensor_parse(r,stop,&g->init[g->nw])) {
				break;
			}
			g->nw++;
		} else if (f==11&&w==2) {
			if (g->ninput==CX_IO||cx_value_info(r,stop,g->input[g->ninput],160,&g->input_meta[g->ninput])) {
				break;
			}
			g->ninput++;
		} else if (f==12&&w==2) {
			if (g->noutput==CX_IO||cx_value_info(r,stop,g->output[g->noutput],160,&g->output_meta[g->noutput])) {
				break;
			}
			g->noutput++;
		}
		if (coi_skip(r,stop)) {
			break;
		}
	}
	r->end=save;
	return ret;
}
static inline void cx_graph_close(cx_graph *g)
{
	if (!g) {
		return;
	}
	if (g->f) {
		fclose(g->f);
	}
	free(g->last_use);
	free(g->weight_index);
	free(g->weight_packed);
	free(g->node);
	free(g->init);
	memset(g,0,sizeof(*g));
}
static inline int cx_graph_prepare_weights(cx_graph *g);
static inline int cx_graph_open(cx_graph *g,const char *path)
{
	if (!g||!path) {
		return -1;
	}
	memset(g,0,sizeof(*g));
	g->f=fopen(path,"rb");
	if (!g->f) {
		return -1;
	}
	if (fseek(g->f,0,SEEK_END)) {
		goto fail;
	}
	long end=ftell(g->f);
	if (end<0||fseek(g->f,0,SEEK_SET)) {
		goto fail;
	}
	coi_reader r= {g->f,end,0};
	int found=0;
	for (;;) {
		unsigned f,w;
		long stop;
		int c=coi_next(&r,&f,&w,&stop);
		if (!c) {
			break;
		}
		if (c<0) {
			goto fail;
		}
		if (f==7&&w==2) {
			if (found||cx_graph_parse(&r,stop,g)) {
				goto fail;
			}
			found=1;
		}
		if (f==8&&w==2) {
			if (g->nopset==12||cx_parse_opset(&r,stop,&g->opset[g->nopset])) {
				goto fail;
			}
			g->nopset++;
		}
		if (f==14&&w==2) {
			if (g->nmeta>=CX_META||cx_model_metadata_entry(&r,stop,&g->meta[g->nmeta])) {
				goto fail;
			}
			if (g->meta[g->nmeta].key[0]) {
				g->nmeta++;
			}
		}
		if (coi_skip(&r,stop)) {
			goto fail;
		}
	}
	if (!found || cx_graph_prepare_weights(g)) {
		goto fail;
	}
	return 0;
fail:cx_graph_close(g);
	return -1;
}
static inline const cx_attr *cx_attr_get(const cx_node *n,const char *name)
{
	for (int i=0; i<n->na; i++)if (!strcmp(n->attrs[i].name,name)) {
			return &n->attrs[i];
		}
	return NULL;
}
static inline int64_t cx_attr_int(const cx_node *n,const char *name,int64_t def)
{
	const cx_attr*a=cx_attr_get(n,name);
	return a?a->iv:def;
}
static inline float cx_attr_float(const cx_node *n,const char *name,float def)
{
	const cx_attr*a=cx_attr_get(n,name);
	return a?a->fv:def;
}
static inline const char *cx_type(int t)
{
	switch (t) {
	case 1:
		return "float32";
	case 2:
		return "uint8";
	case 3:
		return "int8";
	case 4:
		return "uint16";
	case 5:
		return "int16";
	case 6:
		return "int32";
	case 7:
		return "int64";
	case 9:
		return "bool";
	case 10:
		return "float16";
	case 11:
		return "float64";
	default:
		return "unsupported";
	}
}
static inline int cx_is_packed_weight(const cx_graph*g,const cx_weight*w);
static inline void cx_graph_report(const cx_graph *g,FILE *f)
{
	fprintf(f,"ONNX native graph: %zu nodes, %zu initializers; inputs:",g->nn,g->nw);
	for (int i=0; i<g->ninput; i++) {
		fprintf(f," %s",g->input[i]);
	}
	fprintf(f,"; outputs:");
	for (int i=0; i<g->noutput; i++) {
		fprintf(f," %s",g->output[i]);
	}
	fputc('\n',f);
	if (g->nopset) {
		fprintf(f,"ONNX opsets:");
		for (int i=0; i<g->nopset; i++) {
			fprintf(f," %s:%" PRId64,g->opset[i].domain[0]?g->opset[i].domain:"ai.onnx",g->opset[i].version);
		}
		fputc('\n',f);
	}
	if (g->nmeta) {
		fprintf(f,"Model metadata (%d):\n",g->nmeta);
		for (int i=0; i<g->nmeta; i++) {
			fprintf(f,"  %s = %s\n",g->meta[i].key,g->meta[i].value);
		}
	}
	/* Predicted persistent weight allocation, excluding transient activations. */
	uint64_t packed_bytes=0,float_elems=0;
	size_t packed_count=0;
	for (size_t i=0; i<g->nw; i++) {
		const cx_weight*w=&g->init[i];
		if (cx_is_packed_weight(g,w)) {
			packed_count++;
			packed_bytes+=w->count;
		} else {
			float_elems+=w->count;
		}
	}
	fprintf(f,"Native weight storage estimate: %" PRIu64 " bytes (float32 fallback + %zu byte-packed initializers; excludes graph activations)\n",
	        float_elems*4+packed_bytes,packed_count);
	fprintf(f,"Initializers (first 30):\n");
	for (size_t i=0; i<g->nw&&i<30; i++) {
		const cx_weight*w=&g->init[i];
		fprintf(f,"  %s %s [",w->name,cx_type(w->type));
		for (int k=0; k<w->nd; k++) {
			fprintf(f,"%s%" PRId64,k?",":"",w->d[k]);
		}
		fprintf(f,"] %" PRIu64 " elems %s\n",w->count,w->external?"EXTERNAL":"embedded");
	}
}
static inline void cx_graph_interfaces(const cx_graph*g,FILE*out)
{
	for (int side=0; side<2; side++) {
		int n=side?g->noutput:g->ninput;
		fprintf(out,"%s (%d):\n",side?"outputs":"inputs",n);
		for (int j=0; j<n; j++) {
			const cx_value_meta*m=side?&g->output_meta[j]:&g->input_meta[j];
			const char*name=side?g->output[j]:g->input[j];
			fprintf(out,"  %-44s %s ",name,m->type?cx_type(m->type):"(unknown)");
			if (m->nd<0) {
				fprintf(out,"[shape unspecified]");
			} else {
				fputc('[',out);
				for (int k=0; k<m->nd; k++) {
					if (k) {
						fputc(',',out);
					}
					if (m->symbol[k][0]) {
						fprintf(out,"%s",m->symbol[k]);
					} else if (m->d[k]>=0) {
						fprintf(out,"%" PRId64,m->d[k]);
					} else {
						fputc('?',out);
					}
				}
				fputc(']',out);
			}
			fputc('\n',out);
		}
	}
}
/* The name hash is also used by the tensor-lifetime index below. */
static inline uint64_t cx_name_hash(const char *name);
static inline size_t cx_find_weight_position(const cx_graph *g,const char *name)
{
	if (!g||!name||!*name) {
		return SIZE_MAX;
	}
	if (g->weight_index && g->weight_index_capacity) {
		size_t mask=g->weight_index_capacity-1;
		size_t h=(size_t)cx_name_hash(name)&mask;
		for (size_t n=0; n<g->weight_index_capacity; n++) {
			size_t pos=g->weight_index[h];
			if (!pos) {
				return SIZE_MAX;
			}
			if (!strcmp(g->init[pos-1].name,name)) {
				return pos-1;
			}
			h=(h+1)&mask;
		}
		return SIZE_MAX;
	}
	/* Manually constructed test graphs may not have a prebuilt index. */
	for (size_t i=0; i<g->nw; i++)if (!strcmp(g->init[i].name,name)) {
			return i;
		}
	return SIZE_MAX;
}
static inline const cx_weight *cx_find_weight(const cx_graph*g,const char *name)
{
	size_t i=cx_find_weight_position(g,name);
	return i==SIZE_MAX?NULL:&g->init[i];
}
/* O(number of weights + number of graph inputs) preparation. Packed int8
   initializers are permitted only when EVERY consumer is supported, and
   never when an initializer is also an external graph output. */
static inline int cx_graph_prepare_weights(cx_graph *g)
{
	if (!g) {
		return -1;
	}
	if (!g->nw) {
		return 0;
	}
	if (g->weight_index || g->weight_packed) {
		return -1;
	}
	if (g->nw>SIZE_MAX/2) {
		return -1;
	}
	size_t cap=16;
	while (cap<2*g->nw) {
		if (cap>SIZE_MAX/2||cap>SIZE_MAX/sizeof(size_t)/2) {
			return -1;
		}
		cap*=2;
	}
	size_t *index=(size_t*)calloc(cap,sizeof(*index));
	unsigned char *flags=(unsigned char*)calloc(g->nw,sizeof(*flags));
	if (!index||!flags) {
		free(index);
		free(flags);
		return -1;
	}
	size_t mask=cap-1;
	for (size_t i=0; i<g->nw; i++) {
		const cx_weight*w=&g->init[i];
		if (!w->name[0]) {
			free(index);
			free(flags);
			return -1;
		}
		size_t h=(size_t)cx_name_hash(w->name)&mask;
		while (index[h]) {
			/* Ambiguous initializer names must not silently select one weight. */
			if (!strcmp(g->init[index[h]-1].name,w->name)) {
				free(index);
				free(flags);
				return -1;
			}
			h=(h+1)&mask;
		}
		index[h]=i+1;
		flags[i]=(w->type==2||w->type==3) && !w->external && w->count>0 ? 1 : 0;
	}
	g->weight_index=index;
	g->weight_index_capacity=cap;
	g->weight_packed=flags;
	/* To qualify, a weight must be used at least once. */
	unsigned char *used=(unsigned char*)calloc(g->nw,1);
	if (!used) {
		return -1;  /* graph_close() owns the installed tables */
	}
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*n=&g->node[i];
		for (int j=0; j<n->ni; j++) {
			size_t p=cx_find_weight_position(g,n->input[j]);
			if (p==SIZE_MAX) {
				continue;
			}
			used[p]=1;
			if (!((!strcmp(n->op,"MatMulInteger")&&j==1)||
			                (!strcmp(n->op,"QLinearMatMul")&&j==3)||
			                (!strcmp(n->op,"DequantizeLinear")&&j==0)||
			                (!strcmp(n->op,"ConvInteger")&&j==1)||
			                (!strcmp(n->op,"QLinearConv")&&j==3))) {
				flags[p]=0;
			}
		}
	}
	for (int i=0; i<g->noutput; i++) {
		size_t p=cx_find_weight_position(g,g->output[i]);
		if (p!=SIZE_MAX) {
			flags[p]=0;
		}
	}
	for (size_t i=0; i<g->nw; i++)if (!used[i]) {
			flags[i]=0;
		}
	free(used);
	return 0;
}
static inline float cx_half(uint16_t h)
{
	uint32_t u=(uint32_t)(h&32768)<<16;
	uint32_t e=(h>>10)&31,m=h&1023;
	if (e==31) {
		u|=0x7f800000u|(m<<13);
	} else if (e) {
		u|=(e+112)<<23|(m<<13);
	} else if (m) {
		int ex=-14;
		while (!(m&1024)) {
			m<<=1;
			ex--;
		}
		u|=(uint32_t)(ex+127)<<23|((m&1023)<<13);
	}
	float f;
	memcpy(&f,&u,4);
	return f;
}
static inline int cx_load_weight(cx_graph *g,const cx_weight *w,cx_tensor *t)
{
	if (!g||!w||!t||w->external||!w->raw_at||w->count>400000000u||w->nd>CX_DIMS) {
		return -1;
	}
	memset(t,0,sizeof(*t));
	if (w->count>SIZE_MAX/sizeof(float)) {
		return -1;
	}
	t->v=(float*)malloc((size_t)(w->count?w->count:1)*sizeof(float));
	if (!t->v) {
		return -1;
	}
	memcpy(t->name,w->name,sizeof(t->name));
	t->type=w->type;
	t->nd=w->nd;
	t->n=w->count;
	memcpy(t->d,w->d,sizeof(t->d));
	int bytes=0;
	switch (w->type) {
	case 1:
	case 6:
		bytes=4;
		break;
	case 2:
	case 3:
	case 9:
		bytes=1;
		break;
	case 4:
	case 5:
	case 10:
		bytes=2;
		break;
	case 7:
	case 11:
		bytes=8;
		break;
	default:
		goto fail;
	}
	if (fseek(g->f,w->raw_at,SEEK_SET)) {
		goto fail;
	}
	if (w->packed_type==5||w->packed_type==7) {
		/* Packed protobuf signed ints used by shape constants. */
		coi_reader r= {g->f,w->raw_at+(long)w->raw_len,0};
		for (uint64_t i=0; i<w->count; i++) {
			uint64_t u;
			if (coi_varint(&r,&u)) {
				goto fail;
			}
			/* Integer shape indices live in float32 tensors here. Refuse a value
			   that may lose bits rather than silently changing an index. */
			int64_t integer=w->packed_type==5?(int64_t)(int32_t)u:(int64_t)u;
			if (integer < -16777216LL || integer > 16777216LL) {
				goto fail;
			}
			t->v[i]=(float)integer;
		}
		if (ftell(g->f)!=r.end) {
			goto fail;
		}
	} else {
		if (w->packed_type && w->packed_type!=4) {
			goto fail;
		}
		if (w->raw_len!=w->count*(uint64_t)bytes) {
			goto fail;
		}
		unsigned char chunk[16384];
		uint64_t done=0;
		while (done<w->count) {
			size_t n=(size_t)(w->count-done);
			if (n>sizeof(chunk)/(size_t)bytes) {
				n=sizeof(chunk)/(size_t)bytes;
			}
			if (fread(chunk,(size_t)bytes,n,g->f)!=n) {
				goto fail;
			}
			for (size_t j=0; j<n; j++) {
				const unsigned char*b=chunk+j*bytes;
				uint64_t u=0;
				for (int k=0; k<bytes; k++) {
					u|=(uint64_t)b[k]<<(8*k);
				}
				float v=0;
				if (w->type==1) {
					uint32_t u32=(uint32_t)u;
					memcpy(&v,&u32,4);
				} else if (w->type==10) {
					v=cx_half((uint16_t)u);
				} else if (w->type==11) {
					double dv;
					memcpy(&dv,&u,8);
					v=(float)dv;
				} else if (w->type==2||w->type==9) {
					v=(float)(uint8_t)u;
				} else if (w->type==3) {
					v=(float)(int8_t)u;
				} else if (w->type==4) {
					v=(float)(uint16_t)u;
				} else if (w->type==5) {
					v=(float)(int16_t)u;
				} else if (w->type==6) {
					int64_t iv=(int32_t)u;
					if (iv < -16777216LL || iv > 16777216LL) {
						goto fail;
					}
					v=(float)iv;
				} else if (w->type==7) {
					int64_t iv=(int64_t)u;
					if (iv < -16777216LL || iv > 16777216LL) {
						goto fail;
					}
					v=(float)iv;
				}
				t->v[done+j]=v;
			}
			done+=n;
		}
	}
	return 0;
fail:free(t->v);
	memset(t,0,sizeof(*t));
	return -1;
}
/* True when this initializer is consumed only by integer matmul kernels
   or as DequantizeLinear input zero. In that case, keeping it as a byte array reduces
   persistent decoder/joiner memory from 4 bytes/weight to 1 byte/weight. */
static inline int cx_is_packed_weight(const cx_graph*g,const cx_weight*w)
{
	if (!g||!w||(w->type!=2&&w->type!=3)||w->external||w->count==0) {
		return 0;
	}
	if (g->weight_packed && g->init && w>=g->init && w<g->init+g->nw) {
		return g->weight_packed[(size_t)(w-g->init)]!=0;
	}
	int uses=0;
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*n=&g->node[i];
		for (int j=0; j<n->ni; j++)if (!strcmp(n->input[j],w->name)) {
				if (!((!strcmp(n->op,"MatMulInteger")&&j==1)||
				                (!strcmp(n->op,"QLinearMatMul")&&j==3)||
				                (!strcmp(n->op,"DequantizeLinear")&&j==0)||
				                (!strcmp(n->op,"ConvInteger")&&j==1)||
				                (!strcmp(n->op,"QLinearConv")&&j==3))) {
					return 0;
				}
				uses++;
			}
	}
	for (int i=0; i<g->noutput; i++)if (!strcmp(g->output[i],w->name)) {
			return 0;
		}
	return uses>0;
}
static inline int cx_load_packed(cx_graph*g,const cx_weight*w,cx_tensor*t)
{
	if (!g||!w||!t||w->count>400000000u ||
	                (w->type!=2&&w->type!=3)||w->external) {
		return -1;
	}
	if (w->raw_len>LONG_MAX||fseek(g->f,w->raw_at,SEEK_SET)) {
		return -1;
	}
	uint8_t *p=(uint8_t*)malloc((size_t)w->count);
	if (!p) {
		return -1;
	}
	if (!w->packed_type) {
		if (w->raw_len!=w->count||fread(p,1,(size_t)w->count,g->f)!=(size_t)w->count) {
			goto fail;
		}
	} else if (w->packed_type==5) {
		/* TensorProto.int32_data, packed protobuf values (also used for int8). */
		coi_reader r= {g->f,w->raw_at+(long)w->raw_len,0};
		for (uint64_t k=0; k<w->count; k++) {
			uint64_t x;
			if (coi_varint(&r,&x)) {
				goto fail;
			}
			int32_t z=(int32_t)x;
			if (w->type==2?(z<0||z>255):(z< -128||z>127)) {
				goto fail;
			}
			p[k]=(uint8_t)z;
		}
		if (ftell(g->f)!=r.end) {
			goto fail;
		}
	} else {
		goto fail;
	}
	t->qv=p;
	t->v=NULL;
	return 0;
fail:free(p);
	return -1;
}
static inline float cx_get(const cx_tensor*t,uint64_t k)
{
	return t->qv?(t->type==3?(float)(int8_t)t->qv[k]:(float)t->qv[k]):t->v[k];
}
#define CX_ENV_TOMBSTONE ((cx_tensor*)(uintptr_t)1)
static inline void cx_env_index_clear(cx_env *e)
{
	free(e->index);
	e->index=NULL;
	e->index_capacity=e->tombstones=0;
}
/* Rebuild from pointers rather than array positions, so swap-removal does
   not invalidate the hash table. If allocation fails, linear lookup remains
   correct and inference can still proceed. */
static inline int cx_env_index_rebuild(cx_env *e,size_t cap)
{
	cx_tensor**slots=(cx_tensor**)calloc(cap,sizeof(*slots));
	if (!slots) {
		return -1;
	}
	size_t mask=cap-1;
	for (size_t j=0; j<e->n; j++) {
		cx_tensor*t=e->t[j];
		size_t k=(size_t)cx_name_hash(t->name)&mask;
		while (slots[k]) {
			k=(k+1)&mask;
		}
		slots[k]=t;
	}
	free(e->index);
	e->index=slots;
	e->index_capacity=cap;
	e->tombstones=0;
	return 0;
}
static inline void cx_env_index_add(cx_env *e,cx_tensor *v)
{
	if (!e->index && e->n<32) {
		return;
	}
	if (!e->index || (e->n+e->tombstones)*2>=e->index_capacity) {
		size_t cap=64;
		if (e->n>SIZE_MAX/4/sizeof(cx_tensor*)) {
			return;
		}
		while (cap<e->n*4) {
			if (cap>SIZE_MAX/2) {
				return;
			}
			cap*=2;
		}
		if (cx_env_index_rebuild(e,cap)) {
			cx_env_index_clear(e);
		}
		return;
	}
	size_t mask=e->index_capacity-1;
	size_t at=(size_t)cx_name_hash(v->name)&mask;
	size_t deleted=SIZE_MAX;
	for (size_t k=0; k<e->index_capacity; k++) {
		cx_tensor *p=e->index[at];
		if (!p) {
			if (deleted!=SIZE_MAX) {
				at=deleted;
				e->tombstones--;
			}
			e->index[at]=v;
			return;
		}
		if (p==CX_ENV_TOMBSTONE && deleted==SIZE_MAX) {
			deleted=at;
		}
		at=(at+1)&mask;
	}
	/* Defensive fallback for an unexpectedly full table. */
	cx_env_index_clear(e);
}
static inline void cx_env_free(cx_env*e)
{
	if (!e) {
		return;
	}
	for (size_t i=0; i<e->n; i++) {
		free(e->t[i]->v);
		free(e->t[i]->qv);
		cats_q8_vnni_weights_free(e->t[i]->vnni_cache);
		free(e->t[i]);
	}
	free(e->t);
	free(e->index);
	memset(e,0,sizeof(*e));
}
static inline cx_tensor *cx_env_find(cx_env*e,const char *name)
{
	if (!e||!name||!*name) {
		return NULL;
	}
	if (e->index) {
		size_t mask=e->index_capacity-1;
		size_t at=(size_t)cx_name_hash(name)&mask;
		for (size_t k=0; k<e->index_capacity; k++) {
			cx_tensor*t=e->index[at];
			if (!t) {
				return NULL;
			}
			if (t!=CX_ENV_TOMBSTONE && !strcmp(t->name,name)) {
				return t;
			}
			at=(at+1)&mask;
		}
		return NULL;
	}
	for (size_t i=0; i<e->n; i++)if (!strcmp(e->t[i]->name,name)) {
			return e->t[i];
		}
	return NULL;
}
static inline cx_tensor *cx_env_new(cx_env *e,const char *name,const int64_t*d,int nd,int type)
{
	if (!e||!name||!name[0]||nd>CX_DIMS||nd<0||strlen(name)>=160) {
		return NULL;
	}
	uint64_t n=1;
	for (int i=0; i<nd; i++) {
		if (d[i]<0|| (d[i]&&n>400000000u/(uint64_t)d[i])) {
			return NULL;
		}
		n*=d[i];
	}
	if (n>400000000u||n>SIZE_MAX/sizeof(float)||e->n>=CX_LIMIT) {
		return NULL;
	}
	if (cx_env_find(e,name)) {
		return NULL;
	}
	if (cx_grow((void**)&e->t,&e->cap,e->n+1,sizeof(cx_tensor*))) {
		return NULL;
	}
	cx_tensor *v=(cx_tensor*)calloc(1,sizeof(*v));
	if (!v) {
		return NULL;
	}
	strcpy(v->name,name);
	v->nd=nd;
	v->n=n;
	v->type=type;
	if (nd) {
		memcpy(v->d,d,(size_t)nd*sizeof(*d));
	}
	v->v=(float*)calloc(n?n:1,sizeof(float));
	if (!v->v) {
		free(v);
		return NULL;
	}
	v->slot=e->n;
	e->t[e->n++]=v;
	cx_env_index_add(e,v);
	return v;
}
static inline void cx_env_drop(cx_env*e,const char*name)
{
	cx_tensor*t=cx_env_find(e,name);
	if (!t) {
		return;
	}
	if (e->index) {
		size_t mask=e->index_capacity-1;
		size_t at=(size_t)cx_name_hash(name)&mask;
		for (size_t k=0; k<e->index_capacity; k++) {
			cx_tensor*p=e->index[at];
			if (!p) {
				break;
			}
			if (p==t) {
				e->index[at]=CX_ENV_TOMBSTONE;
				e->tombstones++;
				break;
			}
			at=(at+1)&mask;
		}
	}
	size_t at=t->slot;
	if (at>=e->n || e->t[at]!=t) {
		return;  /* environment invariant */
	}
	free(t->v);
	free(t->qv);
	cats_q8_vnni_weights_free(t->vnni_cache);
	free(t);
	size_t last=--e->n;
	if (at!=last) {
		e->t[at]=e->t[last];
		e->t[at]->slot=at;
	}
}
static inline cx_tensor *cx_resolve(cx_graph*g,cx_env*e,const char *name)
{
	if (!name||!*name) {
		return NULL;
	}
	cx_tensor*t=cx_env_find(e,name);
	if (t) {
		return t;
	}
	const cx_weight*w=cx_find_weight(g,name);
	if (!w) {
		return NULL;
	}
	/* Load initializer directly, without allocating a temporary float32
	   destination. That used to double peak memory for large matrices. */
	if (e->n>=CX_LIMIT||cx_grow((void**)&e->t,&e->cap,e->n+1,sizeof(cx_tensor*))) {
		return NULL;
	}
	t=(cx_tensor*)calloc(1,sizeof(*t));
	if (!t) {
		return NULL;
	}
	if (cx_is_packed_weight(g,w)) {
		memcpy(t->name,w->name,sizeof(t->name));
		t->nd=w->nd;
		t->n=w->count;
		t->type=w->type;
		memcpy(t->d,w->d,sizeof(t->d));
		if (cx_load_packed(g,w,t)) {
			free(t);
			return NULL;
		}
	} else if (cx_load_weight(g,w,t)) {
		free(t);
		return NULL;
	}
	t->is_initializer=1;
	t->slot=e->n;
	e->t[e->n++]=t;
	cx_env_index_add(e,t);
	return t;
}


/* Tensor interpreter; all arithmetic uses float32. Integer indices are represented
   exactly for the small dimensions in the supported subset. Unsupported operators
   are ERRORS, never ignored or interpreted as Identity. */
static inline int cx_same_shape(const cx_tensor*a,const cx_tensor*b)
{
	if (a->nd!=b->nd) {
		return 0;
	}
	for (int i=0; i<a->nd; i++)if (a->d[i]!=b->d[i]) {
			return 0;
		}
	return 1;
}
static inline int cx_tensor_copy(cx_env*e,const char*name,const cx_tensor*a)
{
	cx_tensor*y=cx_env_new(e,name,a->d,a->nd,a->type);
	if (!y) {
		return -1;
	}
	if (a->qv) {
		for (uint64_t k=0; k<y->n; k++) {
			y->v[k]=cx_get(a,k);
		}
	} else {
		memcpy(y->v,a->v,(size_t)y->n*sizeof(float));
	}
	return 0;
}
static inline int cx_is(const char*a,const char*b)
{
	return !strcmp(a,b);
}
static inline int cx_more_supported(const char*op);
static inline int cx_supported(const char*op)
{
	static const char *ops[]= {"Identity","Cast","Constant","Relu","Sigmoid","Tanh","Exp","Log","Sqrt","Neg","Abs","SiLU","Swish","Add","Sub","Mul","Div","Pow","MatMul","Gemm","Flatten","Reshape","Transpose","Concat","Gather","GatherElements","ScatterElements","Softmax","ReduceMean","Unsqueeze","Squeeze","LayerNormalization","DequantizeLinear","QuantizeLinear","Shape","Size","Slice","Expand","Tile","Range","ConstantOfShape","Where","Equal","Greater","Less","GreaterOrEqual","LessOrEqual","DynamicQuantizeLinear","MatMulInteger","QLinearMatMul","Conv","ConvInteger","QLinearConv","BatchNormalization","PRelu"};
	for (size_t i=0; i<sizeof(ops)/sizeof(*ops); i++)if (cx_is(op,ops[i])) {
			return 1;
		}
	return cx_more_supported(op);
}
/* Nonstandard node domains cannot be executed merely because the op name
 * matches an ONNX standard operator. */
static inline int cx_supported_node(const cx_node*n)
{
	return (!n->domain[0]||!strcmp(n->domain,"ai.onnx"))&&cx_supported(n->op);
}
/* Conservative ONNX compatibility preflight. Operator NAME is not enough:
 * older softmax schemas flatten trailing dimensions, custom domains are not
 * ONNX standard operators, and unsupported Constant/BatchNorm forms must
 * never be silently misinterpreted. A missing opset is reported UNKNOWN; it
 * remains executable for synthetic fixtures but does not imply validation. */
static inline int64_t cx_standard_opset(const cx_graph*g)
{
	for (int i=0; i<g->nopset; i++)
		if (!g->opset[i].domain[0] || !strcmp(g->opset[i].domain,"ai.onnx")) {
			return g->opset[i].version;
		}
	return 0;
}
static inline const char *cx_node_issue(const cx_graph*g,const cx_node*n)
{
	if (!cx_supported_node(n)) {
		return "unknown operator or nonstandard domain";
	}
	int64_t opset=cx_standard_opset(g);
	if (opset>21) {
		return "ONNX standard opset newer than reviewed subset (21)";
	}
	if (opset>0 && opset<13 &&
	                (!strcmp(n->op,"Softmax")||!strcmp(n->op,"LogSoftmax")) &&
	                cx_attr_int(n,"axis",1)!=-1) {
		return "legacy Softmax axis..last flatten semantics not implemented";
	}
	if (!strcmp(n->op,"GatherElements") && opset>0 && opset<11) {
		return "GatherElements needs ONNX opset 11 or newer";
	}
	if (!strcmp(n->op,"ScatterElements")) {
		if (opset>0 && opset<11) {
			return "ScatterElements needs ONNX opset 11 or newer";
		}
		const cx_attr *r=cx_attr_get(n,"reduction");
		if (r && r->sv[0] && strcmp(r->sv,"none")) {
			if (opset>0 && opset<16) {
				return "ScatterElements reduction needs opset 16+";
			}
			if ((!strcmp(r->sv,"max")||!strcmp(r->sv,"min")) && opset>0 && opset<18) {
				return "ScatterElements max/min reduction needs opset 18+";
			}
			if (strcmp(r->sv,"add")&&strcmp(r->sv,"mul")&&
			                strcmp(r->sv,"max")&&strcmp(r->sv,"min")) {
				return "unknown ScatterElements reduction";
			}
		}
	}
	if (!strcmp(n->op,"Constant")) {
		if (cx_attr_get(n,"value")||cx_attr_get(n,"value_int")||
		                cx_attr_get(n,"value_float")) {
			return NULL;
		}
		const cx_attr*a=cx_attr_get(n,"value_ints");
		return a&&a->nints>0?NULL:"Constant attribute encoding not supported";
	}
	if (!strcmp(n->op,"LayerNormalization")&&n->no!=1) {
		return "LayerNormalization auxiliary outputs unsupported";
	}
	if (!strcmp(n->op,"BatchNormalization")&&(n->no!=1||n->ni!=5)) {
		return "BatchNormalization training inputs/auxiliary outputs unsupported";
	}
	if (!strcmp(n->op,"Split") && opset>=18) {
		const cx_attr*a=cx_attr_get(n,"num_outputs");
		if (a && a->iv!=n->no) {
			return "Split num_outputs disagrees with outputs";
		}
	}
	return NULL;
}
static inline uint64_t cx_name_hash(const char*name);
/* Build a temporary symbol table for ONNX SSA/topological-order validation.
 * ONNX models may list an initializer also as a graph input (default input),
 * but two computational nodes must not define the same tensor name.
 * No change to inference; diagnostics run only when requested. */
static inline size_t cx_graph_check_flow(const cx_graph *g,FILE *out)
{
	if (!g) {
		return 1;
	}
	size_t all=(size_t)g->ninput+g->nw+(size_t)g->noutput;
	for (size_t i=0; i<g->nn; i++) {
		if ((size_t)g->node[i].no>SIZE_MAX-all) {
			return 1;
		}
		all+=(size_t)g->node[i].no;
	}
	if (all>SIZE_MAX/2 || all>SIZE_MAX/sizeof(const char*)/2) {
		return 1;
	}
	size_t cap=16;
	while (cap<all*2) {
		if (cap>SIZE_MAX/2) {
			return 1;
		}
		cap*=2;
	}
	const char **names=(const char**)calloc(cap,sizeof(*names));
	unsigned char *origin=(unsigned char*)calloc(cap,1);
	if (!names||!origin) {
		free(names);
		free(origin);
		return 1;
	}
	size_t issues=0,mask=cap-1;
	/* Entry kinds: graph input=1, initializer=2, node output=3. */
	for (int i=0; i<g->ninput; i++) {
		const char *name=g->input[i];
		if (!*name) {
			issues++;
			continue;
		}
		size_t h=(size_t)cx_name_hash(name)&mask;
		while (names[h] && strcmp(names[h],name)) {
			h=(h+1)&mask;
		}
		if (names[h]) {
			issues++;
			if (out) {
				fprintf(out,"  FLOW duplicate graph input: %s\n",name);
			}
		} else {
			names[h]=name;
			origin[h]=1;
		}
	}
	for (size_t i=0; i<g->nw; i++) {
		const char *name=g->init[i].name;
		if (!*name) {
			issues++;
			continue;
		}
		size_t h=(size_t)cx_name_hash(name)&mask;
		while (names[h] && strcmp(names[h],name)) {
			h=(h+1)&mask;
		}
		if (names[h] && origin[h]!=1) {
			issues++;
			if (out) {
				fprintf(out,"  FLOW duplicate initializer: %s\n",name);
			}
		} else {
			names[h]=name;
			origin[h]=2;
		}
	}
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*n=&g->node[i];
		for (int j=0; j<n->ni; j++) {
			const char *name=n->input[j];
			if (!*name) {
				continue;  /* optional input */
			}
			size_t h=(size_t)cx_name_hash(name)&mask;
			while (names[h] && strcmp(names[h],name)) {
				h=(h+1)&mask;
			}
			if (!names[h]) {
				issues++;
				if (out) {
					fprintf(out,"  FLOW node %zu (%s) missing/forward input: %s\n",i,n->op,name);
				}
			}
		}
		for (int j=0; j<n->no; j++) {
			const char *name=n->output[j];
			if (!*name) {
				continue;  /* optional output */
			}
			size_t h=(size_t)cx_name_hash(name)&mask;
			while (names[h] && strcmp(names[h],name)) {
				h=(h+1)&mask;
			}
			if (names[h]) {
				issues++;
				if (out) {
					fprintf(out,"  FLOW node %zu (%s) redefines tensor: %s\n",i,n->op,name);
				}
			} else {
				names[h]=name;
				origin[h]=3;
			}
		}
	}
	for (int j=0; j<g->noutput; j++) {
		const char *name=g->output[j];
		if (!*name) {
			issues++;
			continue;
		}
		size_t h=(size_t)cx_name_hash(name)&mask;
		while (names[h] && strcmp(names[h],name)) {
			h=(h+1)&mask;
		}
		if (!names[h]) {
			issues++;
			if (out) {
				fprintf(out,"  FLOW graph output is never defined: %s\n",name);
			}
		}
	}
	if (out) {
		fprintf(out,"Graph topology: %zu issue(s)\n",issues);
	}
	free(origin);
	free(names);
	return issues;
}
static inline size_t cx_graph_preflight(const cx_graph*g,FILE*out)
{
	size_t problems=0,missing=0;
	int64_t opset=cx_standard_opset(g);
	if (out)fprintf(out,"Preflight: ai.onnx opset=%" PRId64 "%s\n",opset,
		                opset?"":" (UNKNOWN: no declaration)");
	for (size_t i=0; i<g->nn; i++) {
		const char*reason=cx_node_issue(g,&g->node[i]);
		if (reason) {
			if (!cx_supported_node(&g->node[i])) {
				missing++;
			}
			problems++;
			if (out)fprintf(out,"  BLOCK node %zu: %s%s%s — %s\n",i,
				                g->node[i].domain,g->node[i].domain[0]?"::":"",
				                g->node[i].op,reason);
		}
	}
	problems+=cx_graph_check_flow(g,out);
	for (size_t i=0; i<g->nw; i++) {
		const cx_weight*w=&g->init[i];
		if (w->external || !strcmp(cx_type(w->type),"unsupported") ||
		                w->count>400000000u) {
			if (out) {
				fprintf(out,"  BLOCK initializer %s: unsupported external/type/size\n",w->name);
			}
			problems++;
		}
	}
	if (out)fprintf(out,"Preflight: %zu blocking issue(s), including %zu unimplemented node(s); file parsing alone does not verify numeric parity.\n",
		                problems,missing);
	return problems;
}
static inline void cx_audit(const cx_graph*g,FILE*out)
{
	size_t unsupported=0, total=0;
	fprintf(out,"Implemented native subset: ");
	for (size_t i=0; i<g->nn; i++) {
		if (!cx_supported_node(&g->node[i])) {
			unsupported++;
			total++;
		}
	}
	fprintf(out,"%zu/%zu supported node instances; missing %zu.\n",g->nn-unsupported,g->nn,unsupported);
	char names[64][160];
	size_t cnt[64]= {0},u=0;
	for (size_t i=0; i<g->nn; i++)if (!cx_supported_node(&g->node[i])) {
			char op_id[160];
			const cx_node*n=&g->node[i];
			snprintf(op_id,sizeof(op_id),"%s%s%s",n->domain,n->domain[0]?"::":"",n->op);
			size_t j=0;
			for (; j<u; j++)if (cx_is(names[j],op_id)) {
					break;
				}
			if (j==u && u<64) {
				strcpy(names[u],op_id);
				u++;
			}
			if (j<64) {
				cnt[j]++;
			}
		}
	for (size_t i=0; i<u; i++) {
		fprintf(out,"  MISSING %-26s %zu\n",names[i],cnt[i]);
	}
	if (total && u==64) {
		fprintf(out,"  ...more types omitted\n");
	}
}
static inline uint64_t cx_index(const int64_t *dims,int nd,const int64_t *coord)
{
	uint64_t at=0;
	for (int i=0; i<nd; i++) {
		at=at*(uint64_t)dims[i]+(uint64_t)coord[i];
	}
	return at;
}
static inline void cx_unravel(uint64_t at,const int64_t *dims,int nd,int64_t *coord)
{
	for (int i=nd-1; i>=0; i--) {
		int64_t d=dims[i];
		coord[i]=d?(int64_t)(at%(uint64_t)d):0;
		at=d?at/(uint64_t)d:0;
	}
}
static inline int cx_binary_dims(const cx_tensor*a,const cx_tensor*b,int64_t*d,int*nd)
{
	*nd=a->nd>b->nd?a->nd:b->nd;
	for (int i=0; i<*nd; i++) {
		int ai=i-(int)(*nd-a->nd),bi=i-(int)(*nd-b->nd);
		int64_t av=ai<0?1:a->d[ai],bv=bi<0?1:b->d[bi];
		if (av!=bv && av!=1 && bv!=1) {
			return -1;
		}
		d[i]=av==1?bv:av;
	}
	return 0;
}
static inline float cx_binary_apply(const char*op,float a,float b)
{
	if (cx_is(op,"Add")) {
		return a+b;
	}
	if (cx_is(op,"Sub")) {
		return a-b;
	}
	if (cx_is(op,"Mul")) {
		return a*b;
	}
	if (cx_is(op,"Div")) {
		return a/b;
	}
	return powf(a,b);
}
static inline int cx_binary(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*b)
{
	int64_t d[CX_DIMS],coord[CX_DIMS];
	int nd;
	if (cx_binary_dims(a,b,d,&nd)) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,1);
	if (!y) {
		return -1;
	}
	if (cx_same_shape(a,b)) {
		int op=cx_is(n->op,"Add")?0:cx_is(n->op,"Sub")?1:cx_is(n->op,"Mul")?2:cx_is(n->op,"Div")?3:-1;
		if (op>=0) {
			cats_f32_elementwise(y->v,a->v,b->v,y->n,op);
		} else for (uint64_t i=0; i<y->n; i++) {
				y->v[i]=cx_binary_apply(n->op,a->v[i],b->v[i]);
			}
		return 0;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,d,nd,coord);
		uint64_t ai=0,bi=0;
		for (int j=0; j<a->nd; j++) {
			int k=nd-a->nd+j;
			ai=ai*(uint64_t)a->d[j]+(a->d[j]==1?0:(uint64_t)coord[k]);
		}
		for (int j=0; j<b->nd; j++) {
			int k=nd-b->nd+j;
			bi=bi*(uint64_t)b->d[j]+(b->d[j]==1?0:(uint64_t)coord[k]);
		}
		y->v[i]=cx_binary_apply(n->op,a->v[ai],b->v[bi]);
	}
	return 0;
}
/* ONNX MatMul: 1D / 2D / batched N-D; leading dimensions broadcast.
   A[...,M,K] @ B[...,K,N] -> [...,M,N] with 1D promotions removed. */
static inline int cx_matmul(cx_env*e,const char*name,const cx_tensor*a,const cx_tensor*b)
{
	if (a->nd<1||b->nd<1) {
		return -1;
	}
	int abad=a->nd>2?a->nd-2:0,bbad=b->nd>2?b->nd-2:0;
	int batch=abad>bbad?abad:bbad;
	int output_nd=batch+(a->nd==1?0:1)+(b->nd==1?0:1);
	if (output_nd>CX_DIMS) {
		return -1;
	}
	int64_t M=a->nd==1?1:a->d[a->nd-2],K=a->d[a->nd-1];
	int64_t KB=b->nd==1?b->d[0]:b->d[b->nd-2],N=b->nd==1?1:b->d[b->nd-1];
	if (K!=KB || M<0 || N<0 || K<0) {
		return -1;
	}
	int64_t d[CX_DIMS],bi[CX_DIMS]= {0};
	for (int i=0; i<batch; i++) {
		int ai=i-batch+abad,ib=i-batch+bbad;
		int64_t aa=ai<0?1:a->d[ai],bb=ib<0?1:b->d[ib];
		if (aa!=bb&&aa!=1&&bb!=1) {
			return -1;
		}
		d[i]=aa==1?bb:aa;
	}
	int k=batch;
	if (a->nd!=1) {
		d[k++]=M;
	}
	if (b->nd!=1) {
		d[k++]=N;
	}
	cx_tensor*y=cx_env_new(e,name,d,output_nd,1);
	if (!y) {
		return -1;
	}
	uint64_t batches=1;
	for (int i=0; i<batch; i++) {
		batches*=d[i];
	}
	for (uint64_t t=0; t<batches; t++) {
		cx_unravel(t,d,batch,bi);
		uint64_t oa=0,ob=0;
		for (int i=0; i<abad; i++) {
			int64_t dim=a->d[i],j=bi[batch-abad+i];
			oa=oa*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		for (int i=0; i<bbad; i++) {
			int64_t dim=b->d[i],j=bi[batch-bbad+i];
			ob=ob*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		const float *ap=a->v+oa*(uint64_t)M*(uint64_t)K;
		const float *bp=b->v+ob*(uint64_t)K*(uint64_t)N;
		float *dst=y->v+t*(uint64_t)M*(uint64_t)N;
		cats_sgemm_f32(ap,bp,dst,M,N,K);
	}
	return 0;
}
/* ONNX Constant supports tensor, scalar and repeated integer variants.
 * The scalar forms are common in torch-exported shape subgraphs. */
static inline int cx_constant(cx_graph*g,cx_env*e,const cx_node*n)
{
	const cx_attr *a=cx_attr_get(n,"value");
	if (a) {
		const cx_weight*w=&a->tensor;
		cx_tensor *y=cx_env_new(e,n->output[0],w->d,w->nd,w->type);
		if (!y) {
			return -1;
		}
		cx_tensor v;
		if (cx_load_weight(g,w,&v)) {
			return -1;
		}
		if (v.n!=y->n) {
			free(v.v);
			return -1;
		}
		memcpy(y->v,v.v,(size_t)v.n*sizeof(float));
		free(v.v);
		return 0;
	}
	a=cx_attr_get(n,"value_int");
	if (a) {
		/* Reject integers that cannot be represented exactly in float32. */
		if (a->iv < -16777216LL || a->iv > 16777216LL) {
			return -1;
		}
		cx_tensor*y=cx_env_new(e,n->output[0],NULL,0,7);
		if (!y) {
			return -1;
		}
		y->v[0]=(float)a->iv;
		return 0;
	}
	a=cx_attr_get(n,"value_float");
	if (a) {
		cx_tensor*y=cx_env_new(e,n->output[0],NULL,0,1);
		if (!y) {
			return -1;
		}
		y->v[0]=a->fv;
		return 0;
	}
	a=cx_attr_get(n,"value_ints");
	if (a && a->nints>0) {
		int64_t dims[1]= {a->nints};
		for (int k=0; k<a->nints; k++)
			if (a->ints[k]<-16777216LL||a->ints[k]>16777216LL) {
				return -1;
			}
		cx_tensor*y=cx_env_new(e,n->output[0],dims,1,7);
		if (!y) {
			return -1;
		}
		for (int k=0; k<a->nints; k++) {
			y->v[k]=(float)a->ints[k];
		}
		return 0;
	}
	return -1;
}
static inline int cx_unary(cx_env*e,const cx_node*n,const cx_tensor*a)
{
	cx_tensor*y=cx_env_new(e,n->output[0],a->d,a->nd,a->type);
	if (!y) {
		return -1;
	}
	if (cx_is(n->op,"Relu")) {
		cats_f32_relu(y->v,a->v,y->n);
		return 0;
	}
	for (uint64_t i=0; i<y->n; i++) {
		float v=a->v[i];
		if (cx_is(n->op,"Relu")) {
			v=v>0?v:0;
		} else if (cx_is(n->op,"Sigmoid")) {
			v=1.f/(1.f+expf(-v));
		} else if (cx_is(n->op,"Tanh")) {
			v=tanhf(v);
		} else if (cx_is(n->op,"Exp")) {
			v=expf(v);
		} else if (cx_is(n->op,"Log")) {
			v=logf(v);
		} else if (cx_is(n->op,"Sqrt")) {
			v=sqrtf(v);
		} else if (cx_is(n->op,"Neg")) {
			v=-v;
		} else if (cx_is(n->op,"Abs")) {
			v=fabsf(v);
		} else if (cx_is(n->op,"SiLU")||cx_is(n->op,"Swish")) {
			v=v/(1.f+expf(-v));
		}
		y->v[i]=v;
	}
	return 0;
}
static inline int cx_i64(float f,int64_t *out); /* defined in the kernel section below */
static inline int cx_reshape(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*shapes)
{
	int nd=0;
	int64_t d[CX_DIMS];
	uint64_t known=1;
	int missing=-1;
	if (!shapes||shapes->n>CX_DIMS) {
		return -1;
	}
	nd=(int)shapes->n;
	for (int i=0; i<nd; i++) {
		int64_t v;
		/* A shape is an integer tensor.  Silent float->int truncation (or a NaN
		   conversion) corrupts the entire downstream attention graph. */
		if (cx_i64(cx_get(shapes,(uint64_t)i),&v)) {
			return -1;
		}
		if (v==0&&cx_attr_int(n,"allowzero",0)==0) {
			if (i>=a->nd) {
				return -1;
			}
			v=a->d[i];
		}
		if (v==-1) {
			if (missing!=-1) {
				return -1;
			}
			missing=i;
			d[i]=1;
		} else {
			if (v<0|| (v && known>UINT64_MAX/(uint64_t)v)) {
				return -1;
			}
			known*=v;
			d[i]=v;
		}
	}
	if (missing!=-1) {
		if (known==0||a->n%known) {
			return -1;
		}
		d[missing]=(int64_t)(a->n/known);
	} else if (known!=a->n) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,a->type);
	if (!y||y->n!=a->n) {
		return -1;
	}
	memcpy(y->v,a->v,(size_t)a->n*sizeof(float));
	return 0;
}
static inline int cx_transpose(cx_env*e,const cx_node*n,const cx_tensor*a)
{
	if (!a||a->nd==0) {
		return -1;
	}
	int nd=a->nd,perm[CX_DIMS];
	int64_t d[CX_DIMS],loc[CX_DIMS],src[CX_DIMS];
	const cx_attr*attr=cx_attr_get(n,"perm");
	for (int i=0; i<nd; i++) {
		perm[i]=attr&&attr->nints==nd?(int)attr->ints[i]:nd-i-1;
		if (perm[i]<0||perm[i]>=nd) {
			return -1;
		}
		for (int j=0; j<i; j++)if (perm[j]==perm[i]) {
				return -1;
			}
		d[i]=a->d[perm[i]];
	}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,a->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,d,nd,loc);
		for (int k=0; k<nd; k++) {
			src[perm[k]]=loc[k];
		}
		y->v[i]=a->v[cx_index(a->d,nd,src)];
	}
	return 0;
}
static inline int cx_concat(cx_env*e,const cx_node*n,cx_tensor**in)
{
	int64_t dims[CX_DIMS],axis=cx_attr_int(n,"axis",0);
	int nd=in[0]->nd;
	if (axis<0) {
		axis+=nd;
	}
	if (axis<0||axis>=nd||nd==0) {
		return -1;
	}
	memcpy(dims,in[0]->d,(size_t)nd*sizeof(*dims));
	uint64_t sum=0;
	for (int j=0; j<n->ni; j++) {
		cx_tensor*a=in[j];
		if (!a||a->nd!=nd) {
			return -1;
		}
		for (int k=0; k<nd; k++)if (k!=axis&&a->d[k]!=dims[k]) {
				return -1;
			}
		sum+=(uint64_t)a->d[axis];
		if (sum>INT64_MAX) {
			return -1;
		}
	}
	dims[axis]=(int64_t)sum;
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,1);
	if (!y) {
		return -1;
	}
	uint64_t outer=1,inner=1;
	for (int i=0; i<axis; i++) {
		outer*=dims[i];
	}
	for (int i=axis+1; i<nd; i++) {
		inner*=dims[i];
	}
	for (uint64_t t=0; t<outer; t++) {
		uint64_t written=0;
		for (int i=0; i<n->ni; i++) {
			cx_tensor*a=in[i];
			uint64_t block=(uint64_t)a->d[axis]*inner;
			memcpy(y->v+t*sum*inner+written,a->v+t*block,(size_t)block*sizeof(float));
			written+=block;
		}
	}
	return 0;
}
static inline int cx_gather(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*ind)
{
	int axis=(int)cx_attr_int(n,"axis",0);
	if (axis<0) {
		axis+=a->nd;
	}
	if (axis<0||axis>=a->nd||a->nd+ind->nd-1>CX_DIMS) {
		return -1;
	}
	int nd=a->nd+ind->nd-1;
	int64_t d[CX_DIMS],loc[CX_DIMS],src[CX_DIMS],iloc[CX_DIMS];
	for (int i=0; i<axis; i++) {
		d[i]=a->d[i];
	}
	for (int i=0; i<ind->nd; i++) {
		d[axis+i]=ind->d[i];
	}
	for (int i=axis+1; i<a->nd; i++) {
		d[i+ind->nd-1]=a->d[i];
	}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,a->type);
	if (!y) {
		return -1;
	}
	for (uint64_t at=0; at<y->n; at++) {
		cx_unravel(at,d,nd,loc);
		for (int i=0; i<axis; i++) {
			src[i]=loc[i];
		}
		for (int i=0; i<ind->nd; i++) {
			iloc[i]=loc[axis+i];
		}
		int64_t pick;
		if (cx_i64(cx_get(ind,cx_index(ind->d,ind->nd,iloc)),&pick)) {
			return -1;
		}
		if (pick<0) {
			pick+=a->d[axis];
		}
		if (pick<0||pick>=a->d[axis]) {
			return -1;
		}
		src[axis]=pick;
		for (int i=axis+1; i<a->nd; i++) {
			src[i]=loc[i+ind->nd-1];
		}
		y->v[at]=a->v[cx_index(a->d,a->nd,src)];
	}
	return 0;
}
static inline int cx_softmax(cx_env*e,const cx_node*n,const cx_tensor*a)
{
	int ax=(int)cx_attr_int(n,"axis",-1);
	if (ax<0) {
		ax+=a->nd;
	}
	if (ax<0||ax>=a->nd) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],a->d,a->nd,1);
	if (!y) {
		return -1;
	}
	uint64_t outer=1,inner=1,classes=(uint64_t)a->d[ax];
	for (int k=0; k<ax; k++) {
		outer*=a->d[k];
	}
	for (int k=ax+1; k<a->nd; k++) {
		inner*=a->d[k];
	}
	for (uint64_t o=0; o<outer; o++)for (uint64_t b=0; b<inner; b++) {
			uint64_t at=o*classes*inner+b;
			float max=-INFINITY;
			for (uint64_t k=0; k<classes; k++)if (a->v[at+k*inner]>max) {
					max=a->v[at+k*inner];
				}
			float sum=0;
			for (uint64_t k=0; k<classes; k++) {
				float v=expf(a->v[at+k*inner]-max);
				sum+=v;
				y->v[at+k*inner]=v;
			}
			if (sum<=0) {
				return -1;
			}
			for (uint64_t k=0; k<classes; k++) {
				y->v[at+k*inner]/=sum;
			}
		}
	return 0;
}
static inline int cx_reduce_mean(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*axes)
{
	int reduce[CX_DIMS]= {0};
	const cx_attr*ax=cx_attr_get(n,"axes");
	if (ax&&ax->nints) {
		for (int k=0; k<ax->nints; k++) {
			int v=(int)ax->ints[k];
			if (v<0) {
				v+=a->nd;
			}
			if (v<0||v>=a->nd) {
				return -1;
			}
			reduce[v]=1;
		}
	} else if (axes) {
		for (uint64_t k=0; k<axes->n; k++) {
			int v=(int)axes->v[k];
			if (v<0) {
				v+=a->nd;
			}
			if (v<0||v>=a->nd) {
				return -1;
			}
			reduce[v]=1;
		}
	} else for (int k=0; k<a->nd; k++) {
			reduce[k]=1;
		}
	int keep=(int)cx_attr_int(n,"keepdims",1);
	int64_t d[CX_DIMS],loc[CX_DIMS],orig[CX_DIMS];
	int nd=0;
	for (int k=0; k<a->nd; k++)if (!reduce[k]||keep) {
			d[nd++]=reduce[k]?1:a->d[k];
		}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,1);
	if (!y) {
		return -1;
	}
	uint64_t*cnt=(uint64_t*)calloc((size_t)(y->n?y->n:1),sizeof(uint64_t));
	if (!cnt) {
		return -1;
	}
	for (uint64_t i=0; i<a->n; i++) {
		cx_unravel(i,a->d,a->nd,orig);
		int j=0;
		for (int k=0; k<a->nd; k++)if (!reduce[k]||keep) {
				loc[j++]=reduce[k]?0:orig[k];
			}
		uint64_t at=cx_index(d,nd,loc);
		y->v[at]+=a->v[i];
		cnt[at]++;
	}
	for (uint64_t k=0; k<y->n; k++)if (cnt[k]) {
			y->v[k]/=(float)cnt[k];
		}
	free(cnt);
	return 0;
}
static inline int cx_layernorm(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*scale,const cx_tensor*bias)
{
	int ax=(int)cx_attr_int(n,"axis",-1);
	if (ax<0) {
		ax+=x->nd;
	}
	if (ax<0||ax>=x->nd) {
		return -1;
	}
	uint64_t inner=1;
	for (int i=ax; i<x->nd; i++) {
		inner*=x->d[i];
	}
	if (scale->n!=inner && !(scale->nd==1&&scale->n==(uint64_t)x->d[x->nd-1])) {
		return -1;
	}
	if (bias && bias->n!=scale->n) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,1);
	if (!y) {
		return -1;
	}
	float eps=cx_attr_float(n,"epsilon",1e-5f);
	for (uint64_t start=0; start<x->n; start+=inner) {
		double sum=0,var=0;
		for (uint64_t k=0; k<inner; k++) {
			sum+=x->v[start+k];
		}
		double mean=sum/inner;
		for (uint64_t k=0; k<inner; k++) {
			double v=x->v[start+k]-mean;
			var+=v*v;
		}
		float norm=1.f/sqrtf((float)(var/inner)+eps);
		for (uint64_t k=0; k<inner; k++) {
			uint64_t si=k%scale->n;
			y->v[start+k]=(x->v[start+k]-(float)mean)*norm*scale->v[si]+(bias?bias->v[si]:0);
		}
	}
	return 0;
}
static inline int cx_squeeze(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*axes,int unsqueeze)
{
	int64_t d[CX_DIMS]= {0};
	int mask[CX_DIMS]= {0},nd=0;
	const cx_attr*att=cx_attr_get(n,"axes");
	int n_axes=axes?(int)axes->n:(att?att->nints:0);
	if (n_axes>CX_DIMS) {
		return -1;
	}
	if (unsqueeze) {
		nd=a->nd+n_axes;
		if (nd>CX_DIMS) {
			return -1;
		}
		for (int k=0; k<n_axes; k++) {
			int64_t iv;
			if (axes) {
				if (cx_i64(cx_get(axes,k),&iv)) {
					return -1;
				}
			} else {
				iv=att->ints[k];
			}
			if (iv< -nd||iv>=nd) {
				return -1;
			}
			int v=(int)iv;
			if (v<0) {
				v+=nd;
			}
			if (mask[v]) {
				return -1;
			}
			mask[v]=1;
		}
		int j=0;
		for (int i=0; i<nd; i++) {
			d[i]=mask[i]?1:a->d[j++];
		}
	} else {
		if (n_axes) {
			for (int k=0; k<n_axes; k++) {
				int64_t iv;
				if (axes) {
					if (cx_i64(cx_get(axes,k),&iv)) {
						return -1;
					}
				} else {
					iv=att->ints[k];
				}
				if (iv< -a->nd||iv>=a->nd) {
					return -1;
				}
				int v=(int)iv;
				if (v<0) {
					v+=a->nd;
				}
				if (a->d[v]!=1||mask[v]) {
					return -1;
				}
				mask[v]=1;
			}
		} else for (int k=0; k<a->nd; k++)if (a->d[k]==1) {
					mask[k]=1;
				}
		for (int k=0; k<a->nd; k++)if (!mask[k]) {
				d[nd++]=a->d[k];
			}
	}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,a->type);
	if (!y||y->n!=a->n) {
		return -1;
	}
	memcpy(y->v,a->v,(size_t)a->n*sizeof(float));
	return 0;
}
static inline int cx_quant(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*scale,const cx_tensor*zero,int decode)
{
	if (scale->n!=1|| (zero&&zero->n!=1)) {
		return -1;  /* per-channel unsupported */
	}
	if (scale->v[0]<=0) {
		return -1;
	}
	int type=decode?1:(zero?zero->type:2);
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<x->n; i++) {
		float z=zero?zero->v[0]:0;
		if (decode) {
			y->v[i]=(x->v[i]-z)*scale->v[0];
		} else {
			float v=nearbyintf(x->v[i]/scale->v[0])+z;
			y->v[i]=fminf(fmaxf(v,type==3?-128.f:0.f),type==3?127.f:255.f);
		}
	}
	return 0;
}

/* ========================================================================
 * Supplemental ONNX numeric kernels
 * ======================================================================== */

/* Cat's eye ONNX supplemental kernels, C11/libm only. MPL-2.0.
 * Defined after tensor/graph helpers.
 * Not a complete ONNX runtime: see README for supported ops and limits.
 */

/* BatchNormalization inference mode: ONNX channel axis 1, 1D/2D/3D
 * spatial shapes, scale/bias/mean/variance all shape [C]. */
static inline int cx_batchnorm(cx_env*e,const cx_node*n,cx_tensor**v)
{
	if (n->ni!=5||n->no!=1) {
		return -1;
	}
	for (int k=0; k<5; k++)if (!v[k]) {
			return -1;
		}
	const cx_tensor*x=v[0];
	if (x->nd<2||x->nd>CX_DIMS || x->d[1]<1) {
		return -1;
	}
	int64_t C=x->d[1];
	for (int k=1; k<5; k++)if (v[k]->nd!=1||v[k]->n!=(uint64_t)C) {
			return -1;
		}
	float eps=cx_attr_float(n,"epsilon",1.e-5f);
	if (!isfinite(eps)||eps<=0) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,1);
	if (!y) {
		return -1;
	}
	uint64_t spatial=1;
	for (int k=2; k<x->nd; k++) {
		if (x->d[k]<0 || (x->d[k] && spatial>UINT64_MAX/(uint64_t)x->d[k])) {
			return -1;
		}
		spatial*=(uint64_t)x->d[k];
	}
	for (int64_t c=0; c<C; c++) {
		float variance=v[4]->v[c];
		if (!isfinite(variance)||variance<0) {
			return -1;
		}
		float factor=v[1]->v[c]/sqrtf(variance+eps);
		float offset=v[2]->v[c]-v[3]->v[c]*factor;
		for (int64_t batch=0; batch<x->d[0]; batch++) {
			uint64_t start=((uint64_t)batch*(uint64_t)C+(uint64_t)c)*spatial;
			for (uint64_t k=0; k<spatial; k++) {
				y->v[start+k]=factor*x->v[start+k]+offset;
			}
		}
	}
	return 0;
}
static inline int cx_prelu(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*slope)
{
	if (n->ni!=2||n->no!=1) {
		return -1;
	}
	int64_t dims[CX_DIMS]= {0},coord[CX_DIMS]= {0};
	int nd=0;
	if (cx_binary_dims(x,slope,dims,&nd)) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,x->type);
	if (!y) {
		return -1;
	}
	for (uint64_t idx=0; idx<y->n; idx++) {
		cx_unravel(idx,dims,nd,coord);
		uint64_t xi=0,si=0;
		for (int k=0; k<x->nd; k++) {
			xi=xi*(uint64_t)x->d[k]+(x->d[k]==1?0:(uint64_t)coord[nd-x->nd+k]);
		}
		for (int k=0; k<slope->nd; k++) {
			si=si*(uint64_t)slope->d[k]+(slope->d[k]==1?0:(uint64_t)coord[nd-slope->nd+k]);
		}
		float z=x->v[xi];
		y->v[idx]=z>=0?z:z*slope->v[si];
	}
	return 0;
}
/* Product over a half-open dimension range (also handles rank zero). */
static inline int cx_prod(const int64_t *d,int a,int b,uint64_t *v)
{
	uint64_t n=1;
	for (int i=a; i<b; i++) {
		if (d[i]<0|| (d[i] && n>UINT64_MAX/(uint64_t)d[i])) {
			return -1;
		}
		n*=(uint64_t)d[i];
	}
	*v=n;
	return 0;
}
static inline int cx_i64(float f,int64_t *out)
{
	if (!isfinite(f)||f<-(float)16777216||f>(float)16777216||floorf(f)!=f) {
		return -1;
	}
	*out=(int64_t)f;
	return 0;
}
static inline int cx_axis(int axis,int rank,int *out)
{
	if (axis<0) {
		axis+=rank;
	}
	if (axis<0||axis>=rank) {
		return -1;
	}
	*out=axis;
	return 0;
}
static inline int cx_index_at(const cx_tensor*t,uint64_t flat,int axis,int64_t *v)
{
	if (!t||axis<0||axis>=t->nd) {
		return -1;
	}
	uint64_t inner=1;
	for (int i=axis+1; i<t->nd; i++) {
		inner*=t->d[i];
	}
	return cx_i64(t->v[(flat/inner)%(uint64_t)t->d[axis]],v);
}
static inline int cx_shape_op(cx_env*e,const cx_node*n,const cx_tensor*x)
{
	int start=(int)cx_attr_int(n,"start",0),end=(int)cx_attr_int(n,"end",x->nd);
	if (start<0) {
		start+=x->nd;
	}
	if (end<0) {
		end+=x->nd;
	}
	if (start<0) {
		start=0;
	}
	if (end<0) {
		end=0;
	}
	if (start>x->nd) {
		start=x->nd;
	}
	if (end>x->nd) {
		end=x->nd;
	}
	if (end<start) {
		end=start;
	}
	int64_t dims[1]= {end-start};
	cx_tensor*y=cx_env_new(e,n->output[0],dims,1,7);
	if (!y) {
		return -1;
	}
	for (int i=start; i<end; i++) {
		y->v[i-start]=(float)x->d[i];
	}
	return 0;
}
static inline int cx_size_op(cx_env*e,const cx_node*n,const cx_tensor*x)
{
	cx_tensor*y=cx_env_new(e,n->output[0],NULL,0,7);
	if (!y) {
		return -1;
	}
	if (x->n>16777216u) {
		return -1;
	}
	y->v[0]=(float)x->n;
	return 0;
}
static inline int cx_expand_op(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*target)
{
	if (!target||target->nd!=1||target->n>CX_DIMS) {
		return -1;
	}
	if (target->n>CX_DIMS) {
		return -1;
	}
	int nd=(int)(target->n>(uint64_t)x->nd?target->n:(uint64_t)x->nd);
	int64_t dims[CX_DIMS];
	for (int i=0; i<nd; i++) {
		int j=i-(nd-x->nd),q=i-(nd-(int)target->n);
		int64_t a=j<0?1:x->d[j],b=1;
		if (q>=0 && cx_i64(target->v[q],&b)) {
			return -1;
		}
		if (b<0 || (a!=b && a!=1 && b!=1)) {
			return -1;
		}
		dims[i]=a==1?b:a;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,x->type);
	if (!y) {
		return -1;
	}
	int64_t coord[CX_DIMS];
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,nd,coord);
		uint64_t idx=0;
		for (int j=0; j<x->nd; j++) {
			idx=idx*(uint64_t)x->d[j]+(x->d[j]==1?0:(uint64_t)coord[nd-x->nd+j]);
		}
		y->v[i]=x->v[idx];
	}
	return 0;
}
static inline int cx_tile_op(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*repeats)
{
	if (!repeats||repeats->n!=(uint64_t)x->nd) {
		return -1;
	}
	int64_t dims[CX_DIMS],rep[CX_DIMS],loc[CX_DIMS];
	for (int i=0; i<x->nd; i++) {
		if (cx_i64(repeats->v[i],&rep[i])||rep[i]<0|| (rep[i] && x->d[i]>INT64_MAX/rep[i])) {
			return -1;
		}
		dims[i]=x->d[i]*rep[i];
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,x->nd,x->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,x->nd,loc);
		uint64_t k=0;
		for (int j=0; j<x->nd; j++) {
			k=k*(uint64_t)x->d[j]+(uint64_t)(loc[j]%x->d[j]);
		}
		y->v[i]=x->v[k];
	}
	return 0;
}
static inline int cx_slice_op(cx_env*e,const cx_node*n,cx_tensor**v)
{
	cx_tensor *x=v[0],*starts=v[1],*ends=v[2],*axes=v[3],*steps=v[4];
	if (!x||!starts||!ends||starts->n!=ends->n||starts->n>CX_DIMS||
	                (axes && axes->n!=starts->n)||(steps&&steps->n!=starts->n)) {
		return -1;
	}
	int64_t start[CX_DIMS],step[CX_DIMS],dims[CX_DIMS],loc[CX_DIMS],src[CX_DIMS];
	int seen[CX_DIMS]= {0};
	for (int i=0; i<x->nd; i++) {
		start[i]=0;
		step[i]=1;
		dims[i]=x->d[i];
	}
	for (uint64_t i=0; i<starts->n; i++) {
		int64_t ax=(int64_t)i,st,en,sp=1;
		if (axes&&cx_i64(axes->v[i],&ax)) {
			return -1;
		}
		if (ax<0) {
			ax+=x->nd;
		}
		if (ax<0||ax>=x->nd||seen[ax]++) {
			return -1;
		}
		/* Slice often uses +/-INT64_MAX as open-ended sentinels. Their exact
		   integer value was lost when read into the float-only tensor store;
		   clamp them to a large safe bound before resolving against dimensions. */
		float a=starts->v[i],b=ends->v[i];
		if (!isfinite(a)||!isfinite(b)) {
			return -1;
		}
		if (fabsf(a)>=16777216.f) {
			st=a>0?INT32_MAX:INT32_MIN;
		} else if (cx_i64(a,&st)) {
			return -1;
		}
		if (fabsf(b)>=16777216.f) {
			en=b>0?INT32_MAX:INT32_MIN;
		} else if (cx_i64(b,&en)) {
			return -1;
		}
		if ((steps&&cx_i64(steps->v[i],&sp))||sp==0) {
			return -1;
		}
		int64_t d=x->d[ax];
		/* ONNX negative bounds use relative indexing, except INT_MAX/MIN sentinel
		   values which cannot be exact in this float-based interpreter. */
		if (st<0) {
			st+=d;
		}
		if (en<0) {
			en+=d;
		}
		if (sp>0) {
			if (st<0) {
				st=0;
			}
			if (st>d) {
				st=d;
			}
			if (en<0) {
				en=0;
			}
			if (en>d) {
				en=d;
			}
			dims[ax]=en>st?1+(en-st-1)/sp:0;
		} else {
			if (st<0) {
				st=-1;
			}
			if (st>=d) {
				st=d-1;
			}
			if (en< -1) {
				en=-1;
			}
			if (en>=d) {
				en=d-1;
			}
			dims[ax]=st>en?1+(st-en-1)/(-sp):0;
		}
		start[ax]=st;
		step[ax]=sp;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,x->nd,x->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,x->nd,loc);
		for (int k=0; k<x->nd; k++) {
			src[k]=start[k]+loc[k]*step[k];
		}
		y->v[i]=x->v[cx_index(x->d,x->nd,src)];
	}
	return 0;
}
static inline int cx_range_op(cx_env*e,const cx_node*n,cx_tensor**v)
{
	if (!v[0]||!v[1]||!v[2]||v[0]->n!=1||v[1]->n!=1||v[2]->n!=1) {
		return -1;
	}
	double start=v[0]->v[0],end=v[1]->v[0],delta=v[2]->v[0];
	if (!isfinite(start)||!isfinite(end)||!isfinite(delta)||delta==0) {
		return -1;
	}
	double count=ceil((end-start)/delta);
	if (!isfinite(count)) {
		return -1;
	}
	if (count<0) {
		count=0;
	}
	if (count>16777216) {
		return -1;
	}
	int64_t dims[1]= {(int64_t)count};
	cx_tensor*y=cx_env_new(e,n->output[0],dims,1,v[0]->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		y->v[i]=(float)(start+(double)i*delta);
	}
	return 0;
}
static inline int cx_constant_of_shape_op(cx_env*e,cx_graph*g,const cx_node*n,const cx_tensor*shape)
{
	if (!shape||shape->nd!=1||shape->n>CX_DIMS) {
		return -1;
	}
	int64_t dims[CX_DIMS];
	for (uint64_t i=0; i<shape->n; i++)if (cx_i64(shape->v[i],&dims[i])||dims[i]<0) {
			return -1;
		}
	const cx_attr*a=cx_attr_get(n,"value");
	int type=1;
	float value=0;
	if (a) {
		cx_tensor c= {0};
		if (cx_load_weight(g,&a->tensor,&c)) {
			return -1;
		}
		if (c.n!=1) {
			free(c.v);
			return -1;
		}
		value=c.v[0];
		type=c.type;
		free(c.v);
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,(int)shape->n,type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		y->v[i]=value;
	}
	return 0;
}
static inline int cx_conditional_op(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*b,const cx_tensor*c)
{
	int64_t temp[CX_DIMS]= {0},dims[CX_DIMS]= {0},loc[CX_DIMS]= {0};
	int nd1,nd;
	if (cx_binary_dims(a,b,temp,&nd1)) {
		return -1;
	}
	cx_tensor fake= {0};
	fake.nd=nd1;
	memcpy(fake.d,temp,sizeof(temp));
	if (cx_binary_dims(&fake,c,dims,&nd)) {
		return -1;
	}
	int typ=cx_is(n->op,"Where")?b->type:9;
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,typ);
	if (!y) {
		return -1;
	}
	const cx_tensor*ins[]= {a,b,c};
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,nd,loc);
		float vals[3];
		for (int j=0; j<3; j++) {
			const cx_tensor*t=ins[j];
			uint64_t at=0;
			for (int k=0; k<t->nd; k++) {
				at=at*(uint64_t)t->d[k]+(t->d[k]==1?0:(uint64_t)loc[nd-t->nd+k]);
			}
			vals[j]=t->v[at];
		}
		y->v[i]=vals[0]!=0?vals[1]:vals[2];
	}
	return 0;
}
static inline int cx_compare_op(cx_env*e,const cx_node*n,const cx_tensor*a,const cx_tensor*b)
{
	int64_t dims[CX_DIMS],coord[CX_DIMS];
	int nd;
	if (cx_binary_dims(a,b,dims,&nd)) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,9);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,nd,coord);
		uint64_t ai=0,bi=0;
		for (int j=0; j<a->nd; j++) {
			ai=ai*(uint64_t)a->d[j]+(a->d[j]==1?0:(uint64_t)coord[nd-a->nd+j]);
		}
		for (int j=0; j<b->nd; j++) {
			bi=bi*(uint64_t)b->d[j]+(b->d[j]==1?0:(uint64_t)coord[nd-b->nd+j]);
		}
		float u=a->v[ai],v=b->v[bi];
		int result=0;
		if (cx_is(n->op,"Equal")) {
			result=u==v;
		}
		if (cx_is(n->op,"Greater")) {
			result=u>v;
		}
		if (cx_is(n->op,"Less")) {
			result=u<v;
		}
		if (cx_is(n->op,"GreaterOrEqual")) {
			result=u>=v;
		}
		if (cx_is(n->op,"LessOrEqual")) {
			result=u<=v;
		}
		y->v[i]=(float)result;
	}
	return 0;
}
static inline int cx_dynamic_quant(cx_env*e,const cx_node*n,const cx_tensor*x)
{
	if (n->no!=3||x->n==0) {
		return -1;
	}
	if (!isfinite(x->v[0])) {
		return -1;
	}
	float lo=x->v[0],hi=x->v[0];
	for (uint64_t i=1; i<x->n; i++) {
		if (!isfinite(x->v[i])) {
			return -1;
		}
		if (x->v[i]<lo) {
			lo=x->v[i];
		}
		if (x->v[i]>hi) {
			hi=x->v[i];
		}
	}
	lo=fminf(lo,0);
	hi=fmaxf(hi,0);
	float scale=(hi-lo)/255.f;
	if (!(scale>0)) {
		scale=1.f;
	}
	int zp=(int)fminf(fmaxf(nearbyintf(-lo/scale),0.f),255.f);
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,2);
	cx_tensor*s=cx_env_new(e,n->output[1],NULL,0,1);
	cx_tensor*z=cx_env_new(e,n->output[2],NULL,0,2);
	if (!y||!s||!z) {
		return -1;
	}
	s->v[0]=scale;
	z->v[0]=(float)zp;
	for (uint64_t i=0; i<x->n; i++) {
		y->v[i]=fminf(fmaxf(nearbyintf(x->v[i]/scale)+zp,0),255);
	}
	return 0;
}
/* ONNX MatMulInteger: rank 2..8, broadcast batch dims, scalar or 1-D
   per-row (A) / per-column (B) zero points. Float storage is exact only
   when all intermediate integer sums remain within +/-2^24. */
static inline int cx_matmul_integer(cx_env*e,const cx_node*n,cx_tensor**v)
{
	cx_tensor*a=v[0],*b=v[1],*za=v[2],*zb=v[3];
	if (!a||!b||a->nd<2||b->nd<2) {
		return -1;
	}
	int ad=a->nd-2,bd=b->nd-2,batch=ad>bd?ad:bd;
	int64_t M=a->d[a->nd-2],K=a->d[a->nd-1],N=b->d[b->nd-1];
	if (K!=b->d[b->nd-2]||batch+2>CX_DIMS) {
		return -1;
	}
	if (za && za->n!=1 && (za->nd!=1||za->n!=(uint64_t)M)) {
		return -1;
	}
	if (zb && zb->n!=1 && (zb->nd!=1||zb->n!=(uint64_t)N)) {
		return -1;
	}
	int64_t dims[CX_DIMS],coord[CX_DIMS];
	for (int i=0; i<batch; i++) {
		int ai=i-(batch-ad),bi=i-(batch-bd);
		int64_t x=ai<0?1:a->d[ai],y=bi<0?1:b->d[bi];
		if (x!=y&&x!=1&&y!=1) {
			return -1;
		}
		dims[i]=x==1?y:x;
	}
	dims[batch]=M;
	dims[batch+1]=N;
	cx_tensor*out=cx_env_new(e,n->output[0],dims,batch+2,6);
	if (!out) {
		return -1;
	}
	uint64_t total=1;
	if (cx_prod(dims,0,batch,&total)) {
		return -1;
	}
	for (uint64_t t=0; t<total; t++) {
		cx_unravel(t,dims,batch,coord);
		uint64_t oa=0,ob=0;
		for (int i=0; i<ad; i++) {
			int64_t dim=a->d[i],j=coord[batch-ad+i];
			oa=oa*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		for (int i=0; i<bd; i++) {
			int64_t dim=b->d[i],j=coord[batch-bd+i];
			ob=ob*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		const float *ap=a->v+oa*(uint64_t)M*(uint64_t)K;
#if CATS_SIMD_X86
		/* Stage 19: MatMulInteger shares the VNNI packed-weight cache with
		   QLinearMatMul. ONNX allows per-row/per-column zero-points, but VNNI
		   handles only the scalar case; all others retain the existing path.
		   K<=258 bounds the maximum absolute sum by 258*255*255 < 2^24,
		   ensuring the float32 output is EXACT, not merely rounded. */
		if (ap && b->qv && M>=4 && N>=16 && !(N&15) && K>=16 && K<=258 &&
		                (a->type==2||a->type==3) && (b->type==2||b->type==3) &&
		                (!za||za->n==1) && (!zb||zb->n==1) && cats_cpu_has_vnni() &&
		                (uint64_t)M<=SIZE_MAX/(uint64_t)N/sizeof(int32_t)) {
			int64_t zA=0,zB=0;
			if ((za&&cx_i64(cx_get(za,0),&zA)) ||
			                (zb&&cx_i64(cx_get(zb,0),&zB))) {
				return -1;
			}
			if (zA>=(a->type==3?-128:0) && zA<=(a->type==3?127:255) &&
			                zB>=(b->type==3?-128:0) && zB<=(b->type==3?127:255)) {
				int32_t *acc=(int32_t*)malloc((size_t)M*(size_t)N*sizeof(*acc));
				if (acc) {
					int worked=0;
					if (b->is_initializer) {
						if (!b->vnni_cache && b->n%((uint64_t)K*(uint64_t)N)==0)
							b->vnni_cache=cats_q8_vnni_weights_create(b->qv,
							              (int64_t)(b->n/((uint64_t)K*(uint64_t)N)),K,N,b->type==3);
						const cats_q8_vnni_weights *cache=(const cats_q8_vnni_weights*)b->vnni_cache;
						if (cache && cache->k==K && cache->n==N &&
						                cache->b_signed==(b->type==3) && ob<(uint64_t)cache->matrices)
							worked=cats_q8_matmul_vnni_cached(ap,cache,(int64_t)ob,M,
							                                  a->type==3,(int32_t)zA,(int32_t)zB,acc);
					}
					if (!worked)worked=cats_q8_matmul_vnni(ap,
						                                       b->qv+ob*(uint64_t)K*(uint64_t)N,M,K,N,
						                                       a->type==3,b->type==3,(int32_t)zA,(int32_t)zB,acc);
					if (worked) {
						for (uint64_t q=0; q<(uint64_t)M*(uint64_t)N; q++) {
							out->v[t*(uint64_t)M*(uint64_t)N+q]=(float)acc[q];
						}
						free(acc);
						continue;
					}
					free(acc);
				}
			}
		}
#endif
		for (int64_t i=0; i<M; i++) {
			int64_t j=0;
#if CATS_SIMD_X86
			/* 258 * 255^2 < 2^24, so all partial sums are exactly representable.
			   This fast path preserves the scalar interpreter's exactness checks. */
			int fast=(a->v&&b->qv&&K>0&&K<=258&&N>=8&&
			          (a->type==2||a->type==3)&&(b->type==2||b->type==3)&&
			          cats_cpu_simd_level()>=2);
			int64_t azero=0;
			if (fast&&za&&cx_i64(za->v[za->n==1?0:i],&azero)) {
				return -1;
			}
			if (fast && (azero<(a->type==3?-128:0)||azero>(a->type==3?127:255))) {
				fast=0;
			}
			if (fast) {
				for (int64_t p=0; p<K; p++) {
					float q=ap[i*K+p];
					if (!isfinite(q)||q!=(float)(int32_t)q ||
					                q<(a->type==3?-128.f:0.f)||q>(a->type==3?127.f:255.f)) {
						fast=0;
						break;
					}
				}
			}
			if (fast) {
				for (; j+8<=N; j+=8) {
					int32_t zcols[8],sums[8];
					int good=1;
					for (int c=0; c<8; c++) {
						int64_t z=0;
						if (zb&&cx_i64(zb->v[zb->n==1?0:j+c],&z)) {
							return -1;
						}
						if (z<(b->type==3?-128:0)||z>(b->type==3?127:255)) {
							good=0;
							break;
						}
						zcols[c]=(int32_t)z;
					}
					if (!good) {
						break;
					}
					cats_q8_row8_avx2_zcols(ap+i*K,b->qv+ob*(uint64_t)K*N,
					                        b->type==3,K,N,j,(int32_t)azero,zcols,sums);
					for (int c=0; c<8; c++) {
						out->v[t*(uint64_t)M*N+(uint64_t)i*N+j+c]=(float)sums[c];
					}
				}
			}
#endif
			for (; j<N; j++) {
				int64_t sum=0,zA=0,zB=0;
				if (za&&cx_i64(za->v[za->n==1?0:i],&zA)) {
					return -1;
				}
				if (zb&&cx_i64(zb->v[zb->n==1?0:j],&zB)) {
					return -1;
				}
				for (int64_t k=0; k<K; k++) {
					int64_t A,B;
					if (cx_i64(ap[i*K+k],&A)||cx_i64(cx_get(b,ob*(uint64_t)K*(uint64_t)N+(uint64_t)k*N+j),&B)) {
						return -1;
					}
					sum+=(A-zA)*(B-zB);
					if (sum>16777216||sum< -16777216) {
						return -1;
					}
				}
				out->v[t*(uint64_t)M*(uint64_t)N+(uint64_t)i*N+j]=(float)sum;
			}
		}
	}
	return 0;
}
/* QLinearMatMul, ONNX batch MatMul broadcasting with int8/uint8 data.
 * Accumulate in int64 to avoid the previous float rounding for large K.
 * Per-tensor scales and zero points are required by this operator.
 * A:[batch...,M,K], B:[batch...,K,N] -> Y:[batch...,M,N]. */
static inline int cx_qlinear_matmul(cx_env*e,const cx_node*n,cx_tensor**v)
{
	if (n->ni!=8||n->no!=1) {
		return -1;
	}
	for (int i=0; i<8; i++)if (!v[i]) {
			return -1;
		}
	const cx_tensor*a=v[0];
	cx_tensor*b=v[3];
	if (a->nd<2||b->nd<2||a->nd>CX_DIMS||b->nd>CX_DIMS||
	                (a->type!=2&&a->type!=3)||(b->type!=2&&b->type!=3)||
	                (v[2]->type!=a->type)||(v[5]->type!=b->type)||
	                (v[7]->type!=2&&v[7]->type!=3)) {
		return -1;
	}
	for (int i=1; i<8; i++)if (i!=3&&v[i]->n!=1) {
			return -1;
		}
	const double as=v[1]->v[0],bs=v[4]->v[0],ys=v[6]->v[0];
	if (!isfinite(as)||!isfinite(bs)||!isfinite(ys)||as<=0||bs<=0||ys<=0) {
		return -1;
	}
	int64_t az,bz,yz;
	if (cx_i64(cx_get(v[2],0),&az)||cx_i64(cx_get(v[5],0),&bz)||
	                cx_i64(cx_get(v[7],0),&yz)) {
		return -1;
	}
	const int64_t M=a->d[a->nd-2],K=a->d[a->nd-1];
	const int64_t N=b->d[b->nd-1];
	if (K!=b->d[b->nd-2]||M<0||K<0||N<0) {
		return -1;
	}
	int abad=a->nd-2,bbad=b->nd-2,batch=abad>bbad?abad:bbad;
	if (batch+2>CX_DIMS) {
		return -1;
	}
	int64_t dims[CX_DIMS],coord[CX_DIMS]= {0};
	for (int i=0; i<batch; i++) {
		int ai=i-(batch-abad),bi=i-(batch-bbad);
		int64_t aa=ai<0?1:a->d[ai],bb=bi<0?1:b->d[bi];
		if (aa<0||bb<0||(aa!=bb&&aa!=1&&bb!=1)) {
			return -1;
		}
		dims[i]=aa==1?bb:aa;
	}
	dims[batch]=M;
	dims[batch+1]=N;
	cx_tensor*y=cx_env_new(e,n->output[0],dims,batch+2,v[7]->type);
	if (!y) {
		return -1;
	}
	uint64_t batches=1;
	if (cx_prod(dims,0,batch,&batches)) {
		return -1;
	}
	const double ratio=as*bs/ys;
	if (!isfinite(ratio)) {
		return -1;
	}
	for (uint64_t t=0; t<batches; t++) {
		cx_unravel(t,dims,batch,coord);
		uint64_t oa=0,ob=0;
		for (int i=0; i<abad; i++) {
			int64_t dim=a->d[i],j=coord[batch-abad+i];
			oa=oa*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		for (int i=0; i<bbad; i++) {
			int64_t dim=b->d[i],j=coord[batch-bbad+i];
			ob=ob*(uint64_t)dim+(uint64_t)(dim==1?0:j);
		}
		const uint64_t abase=oa*(uint64_t)M*(uint64_t)K;
		const uint64_t bbase=ob*(uint64_t)K*(uint64_t)N;
#if CATS_SIMD_X86
		/* VNNI computes the integer accumulator matrix directly from packed
		   weights. Requantization remains identical to the scalar path. */
		if (b->qv && a->v && M>=4 && N>=16 && !(N&15) && K>=16 && K<=32768 &&
		                az>=(a->type==3?-128:0) && az<=(a->type==3?127:255) &&
		                bz>=(b->type==3?-128:0) && bz<=(b->type==3?127:255) &&
		                (uint64_t)M <= SIZE_MAX/(uint64_t)N/sizeof(int32_t) &&
		                cats_cpu_has_vnni()) {
			int32_t *sums=(int32_t*)malloc((size_t)M*(size_t)N*sizeof(*sums));
			if (sums) {
				int worked=0;
				/* Cache only immutable initializer weights. Mutating a runtime
				   tensor never leaves a stale prepack behind. The cache covers all
				   leading batch slices, and is freed with this tensor/session. */
				if (b->is_initializer) {
					if (!b->vnni_cache && K>0 && N>0 &&
					                (uint64_t)K<=SIZE_MAX/(uint64_t)N &&
					                b->n%((uint64_t)K*(uint64_t)N)==0) {
						b->vnni_cache=cats_q8_vnni_weights_create(b->qv,
						              (int64_t)(b->n/((uint64_t)K*(uint64_t)N)),K,N,b->type==3);
					}
					const cats_q8_vnni_weights *cache=(const cats_q8_vnni_weights*)b->vnni_cache;
					if (cache && cache->k==K && cache->n==N &&
					                cache->b_signed==(b->type==3) && ob< (uint64_t)cache->matrices)
						worked=cats_q8_matmul_vnni_cached(a->v+abase,cache,
						                                  (int64_t)ob,M,a->type==3,(int32_t)az,(int32_t)bz,sums);
				}
				if (!worked)worked=cats_q8_matmul_vnni(a->v+abase,b->qv+bbase,M,K,N,
					                                       a->type==3,b->type==3,(int32_t)az,(int32_t)bz,sums);
				if (worked) {
					for (uint64_t q=0; q<(uint64_t)M*(uint64_t)N; q++) {
						double quant=nearbyint((double)sums[q]*ratio+(double)yz);
						if (!isfinite(quant)) {
							free(sums);
							return -1;
						}
						y->v[t*(uint64_t)M*N+q]=(float)fmin(fmax(quant,
						                                    y->type==2?0.0:-128.0),y->type==2?255.0:127.0);
					}
					free(sums);
					continue;
				}
				free(sums);
			}
		}
#endif
		for (int64_t i=0; i<M; i++) {
			int64_t j=0;
#if CATS_SIMD_X86
			/* AVX2: B is packed 1-byte quantized weights, A is an ordinary
			   integer-valued float32 tensor. Use int32 accumulators only in the
			   provably overflow-free range; all other inputs keep scalar semantics. */
			int fast=(b->qv && a->v && N>=8 && K<=32768 && K>=1 &&
			          cats_cpu_simd_level()>=2 &&
			          az>=(a->type==3?-128:0) && az<=(a->type==3?127:255) &&
			          bz>=(b->type==3?-128:0) && bz<=(b->type==3?127:255));
			if (fast) {
				const float *arow=a->v+abase+(uint64_t)i*K;
				for (int64_t p=0; p<K; p++) {
					int64_t qa;
					if (cx_i64(arow[p],&qa) || qa<(a->type==3?-128:0) ||
					                qa>(a->type==3?127:255)) {
						fast=0;
						break;
					}
				}
				if (fast) {
					const uint8_t *brow=b->qv+bbase;
					for (; j+8<=N; j+=8) {
						int32_t sums[8];
						cats_q8_row8_avx2(arow,brow,b->type==3,K,N,j,
						                  (int32_t)az,(int32_t)bz,sums);
						for (int z=0; z<8; z++) {
							double q=nearbyint((double)sums[z]*ratio+(double)yz);
							if (!isfinite(q)) {
								return -1;
							}
							y->v[t*(uint64_t)M*N+(uint64_t)i*N+(uint64_t)j+(uint64_t)z]=
							        (float)fmin(fmax(q,y->type==2?0.0:-128.0),
							                    y->type==2?255.0:127.0);
						}
					}
				}
			}
#endif
			for (; j<N; j++) {
				int64_t sum=0;
				for (int64_t k=0; k<K; k++) {
					int64_t qa,qb;
					if (cx_i64(cx_get(a,abase+(uint64_t)i*K+k),&qa)||
					                cx_i64(cx_get(b,bbase+(uint64_t)k*N+j),&qb)) {
						return -1;
					}
					sum+=(qa-az)*(qb-bz);
				}
				double q=nearbyint((double)sum*ratio+(double)yz);
				if (!isfinite(q)) {
					return -1;
				}
				y->v[t*(uint64_t)M*N+(uint64_t)i*N+j]=(float)fmin(
				                fmax(q,y->type==2?0.0:-128.0),y->type==2?255.0:127.0);
			}
		}
	}
	return 0;
}
static inline int cx_quant_axis(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*scale,const cx_tensor*z,int decode)
{
	int axis=(int)cx_attr_int(n,"axis",1);
	if (axis<0) {
		axis+=x->nd;
	}
	if (!scale||scale->nd>1||!scale->n||
	                (z&&z->n!=scale->n)||
	                (scale->n!=1&&(axis<0||axis>=x->nd||scale->n!=(uint64_t)x->d[axis]))) {
		return -1;
	}
	int type=decode?1:(z?z->type:2);
	if (!decode&&type!=2&&type!=3) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,type);
	if (!y) {
		return -1;
	}
	uint64_t inner=1;
	if (scale->n!=1)for (int i=axis+1; i<x->nd; i++) {
			inner*=x->d[i];
		}
	for (uint64_t i=0; i<x->n; i++) {
		uint64_t j=scale->n==1?0:(i/inner)%scale->n;
		float s=scale->v[j],zz=z?z->v[j]:0;
		if (!(s>0)||!isfinite(s)) {
			return -1;
		}
		if (decode) {
			y->v[i]=(cx_get(x,i)-zz)*s;
		} else {
			float q=nearbyintf(x->v[i]/s)+zz;
			y->v[i]=fminf(fmaxf(q,type==3?-128.f:0.f),type==3?127.f:255.f);
		}
	}
	return 0;
}
/* ONNX 1D/2D NCHW Conv: supports groups, pads, dilations, strides.
   Deliberately scalar reference kernel; optimize after model shape profiling. */
static inline int cx_conv_op(cx_env*e,const cx_node*n,cx_tensor**v)
{
	const cx_tensor*x=v[0],*w=v[1],*bias=v[2];
	if (!x||!w||x->nd<3||x->nd>4||x->nd!=w->nd) {
		return -1;
	}
	int spatial=x->nd-2;
	int64_t groups=cx_attr_int(n,"group",1);
	int64_t batch=x->d[0],ci=x->d[1],co=w->d[0],cki=w->d[1];
	if (groups<1||ci%groups||co%groups||cki!=ci/groups|| (bias&&bias->n!=(uint64_t)co)) {
		return -1;
	}
	int64_t kd[2]= {1,1},id[2]= {1,1},od[2]= {1,1},stride[2]= {1,1},dilation[2]= {1,1},pad[4]= {0};
	const cx_attr*p=cx_attr_get(n,"pads"),*st=cx_attr_get(n,"strides"),*dl=cx_attr_get(n,"dilations"),*aut=cx_attr_get(n,"auto_pad");
	const char *auto_pad=aut?aut->sv:"NOTSET";
	int same_upper=!strcmp(auto_pad,"SAME_UPPER"),same_lower=!strcmp(auto_pad,"SAME_LOWER");
	if (strcmp(auto_pad,"NOTSET")&&strcmp(auto_pad,"VALID")&&!same_upper&&!same_lower) {
		return -1;
	}
	if ((p&&p->nints!=spatial*2)||(st&&st->nints!=spatial)||(dl&&dl->nints!=spatial)) {
		return -1;
	}
	if (p&&(same_upper||same_lower)) {
		return -1;
	}
	for (int j=0; j<spatial; j++) {
		kd[j]=w->d[2+j];
		id[j]=x->d[2+j];
		if (p) {
			pad[j]=p->ints[j];
			pad[j+spatial]=p->ints[j+spatial];
		}
		if (st) {
			stride[j]=st->ints[j];
		}
		if (dl) {
			dilation[j]=dl->ints[j];
		}
		if (kd[j]<1||id[j]<0||stride[j]<1||dilation[j]<1||pad[j]<0||pad[j+spatial]<0) {
			return -1;
		}
		if (kd[j]>INT64_MAX/dilation[j]) {
			return -1;
		}
		int64_t extent=dilation[j]*(kd[j]-1)+1;
		if (same_upper||same_lower) {
			od[j]=(id[j]+stride[j]-1)/stride[j];
			int64_t total=od[j]>0?(od[j]-1)*stride[j]+extent-id[j]:0;
			if (total<0) {
				total=0;
			}
			pad[j]=same_upper?total/2:total-total/2;
			pad[j+spatial]=total-pad[j];
		} else {
			int64_t total=id[j]+pad[j]+pad[j+spatial]-extent;
			od[j]=total<0?0:total/stride[j]+1;
		}
	}
	int64_t outdims[4]= {batch,co,od[0],od[1]};
	cx_tensor*y=cx_env_new(e,n->output[0],outdims,spatial+2,1);
	if (!y) {
		return -1;
	}
	/* Uniform 2D indexing; the trailing 1D width is implied. */
	int64_t HI=spatial==2?id[0]:1,WI=spatial==2?id[1]:id[0];
	int64_t HO=spatial==2?od[0]:1,WO=spatial==2?od[1]:od[0];
	int64_t KH=spatial==2?kd[0]:1,KW=spatial==2?kd[1]:kd[0];
	int64_t SH=spatial==2?stride[0]:1,SW=spatial==2?stride[1]:stride[0];
	int64_t DH=spatial==2?dilation[0]:1,DW=spatial==2?dilation[1]:dilation[0];
	int64_t PH=spatial==2?pad[0]:0,PW=spatial==2?pad[1]:pad[0];
	/* Fast path for the depthwise time convolution typical of Zipformer.
	   Fallback below preserves original double accumulation for all other
	   shapes, and for machines without AVX. */
	if (spatial==1 && groups==ci && ci==co && cki==1 &&
	                stride[0]==1 && dilation[0]==1 && WI>=8 && WO>=8) {
		int accelerated=1;
		for (int64_t bb=0; bb<batch && accelerated; bb++)for (int64_t oc=0; oc<co; oc++) {
				const float *src=x->v+((uint64_t)bb*ci+oc)*(uint64_t)WI;
				const float *filter=w->v+(uint64_t)oc*(uint64_t)KW;
				float *dst=y->v+((uint64_t)bb*co+oc)*(uint64_t)WO;
				if (!cats_dwconv1d_f32(dst,src,filter,WI,WO,KW,PW,bias?bias->v[oc]:0.f)) {
					accelerated=0;
					break;
				}
			}
		if (accelerated) {
			return 0;
		}
	}
	for (int64_t b=0; b<batch; b++)for (int64_t oc=0; oc<co; oc++) {
			int64_t group=oc/(co/groups);
			for (int64_t h=0; h<HO; h++)for (int64_t z=0; z<WO; z++) {

					double sum=bias?bias->v[oc]:0;
					for (int64_t ic=0; ic<cki; ic++)for (int64_t kh=0; kh<KH; kh++)for (int64_t kw=0; kw<KW; kw++) {
								int64_t ih=h*SH+kh*DH-PH,iw=z*SW+kw*DW-PW;
								if (ih<0||ih>=HI||iw<0||iw>=WI) {
									continue;
								}
								uint64_t ix=((uint64_t)b*ci+(uint64_t)(group*cki+ic))*(uint64_t)HI*WI+(uint64_t)ih*WI+iw;
								uint64_t iwgt=((uint64_t)oc*cki+(uint64_t)ic)*(uint64_t)KH*KW+(uint64_t)kh*KW+kw;
								sum+=(double)x->v[ix]*w->v[iwgt];
							}
					y->v[((uint64_t)b*co+oc)*(uint64_t)HO*WO+(uint64_t)h*WO+z]=(float)sum;
				}
		}
	return 0;
}
/* Stage 24: VNNI grouped convolution prepass. Only interior output pixels
 * are computed here; scalar/AVX2 below handles padding and any fallback.
 * Returns 1 if it populated ALL interior pixels, 0 if unsupported/resources
 * unavailable, or -1 on a ConvInteger exact-float32 range violation.
 * Weight packing is local to this invocation: mutable tensors are never
 * attached to the persistent, immutable MatMul weight cache. */
static inline int cx_quant_conv_vnni_interior(
        const cx_tensor*x,const cx_tensor*w,const cx_tensor*xz,const cx_tensor*wz,
        const cx_tensor*xs,const cx_tensor*ws,const cx_tensor*ys,const cx_tensor*yz,
        const cx_tensor*bias,cx_tensor*y,int qlinear,int64_t G,
        int64_t HI,int64_t WI,int64_t HO,int64_t WO,
        int64_t KH,int64_t KW,int64_t PH,int64_t PW)
{
#if CATS_SIMD_X86
	const int64_t B=x->d[0],C=x->d[1],O=w->d[0],CG=w->d[1],OPG=O/G;
	if (!cats_cpu_has_vnni()||!w->qv||(!x->qv&&!x->v)||
	                CG<1||KH<1||KW<1||CG>32768/KH/KW||OPG<4||
	                HI<KH||WI<KW||HO<1||WO<1||
	                HI>INT32_MAX||WI>INT32_MAX||HO>INT32_MAX||WO>INT32_MAX) {
		return 0;
	}
	int64_t zx=(int64_t)(xz?cx_get(xz,0):0);
	if (zx<(x->type==3?-128:0)||zx>(x->type==3?127:255)) {
		return 0;
	}
	for (int64_t oc=0; oc<O; oc++) {
		int64_t zw=(int64_t)(wz?cx_get(wz,wz->n==1?0:(uint64_t)oc):0);
		if (zw<(w->type==3?-128:0)||zw>(w->type==3?127:255)) {
			return 0;
		}
	}
	/* Float-backed inputs have to represent exact INT8/UINT8 values. All
	 * channels validated before writing any output, not inside SIMD hot path. */
	if (x->v) {
		for (uint64_t i=0; i<x->n; i++) {
			float a=x->v[i];
			if (!isfinite(a)||a<(x->type==3?-128.f:0.f)||
			                a>(x->type==3?127.f:255.f)||a!=(float)(int32_t)a) {
				return 0;
			}
		}
	}
	const int64_t K=CG*KH*KW,blocks=(K+3)/4,tiles=(OPG+15)/16;
	if ((uint64_t)blocks>SIZE_MAX/64||
	                (uint64_t)tiles>SIZE_MAX/((size_t)blocks*64)||
	                (uint64_t)OPG>SIZE_MAX/sizeof(int64_t)) {
		return 0;
	}
	const size_t tile_bytes=(size_t)blocks*64;
	/* Cap this temporary per-group packing at 8 MiB. */
	if ((size_t)tiles>8u*1024u*1024u/tile_bytes) {
		return 0;
	}
	uint8_t *packed=(uint8_t*)malloc(tile_bytes*(size_t)tiles);
	int64_t *sums=(int64_t*)malloc((size_t)tiles*16*sizeof(*sums));
	int32_t *zws=(int32_t*)malloc((size_t)tiles*16*sizeof(*zws));
	double *scales=qlinear?(double*)malloc((size_t)tiles*16*sizeof(*scales)):NULL;
	uint32_t *ap=(uint32_t*)malloc((size_t)blocks*sizeof(*ap));
	if (!packed||!sums||!zws||!ap||(qlinear&&!scales)) {
		free(packed);
		free(sums);
		free(zws);
		free(scales);
		free(ap);
		return 0;
	}
	const int64_t zx_u=zx+(x->type==3?128:0);
	/* Traverse only fully interior locations. For 1x1 and valid convolution
	 * this is every output pixel. Both negative and asymmetric pads work. */
	for (int64_t group=0; group<G; group++) {
		memset(packed,0,tile_bytes*(size_t)tiles);
		memset(sums,0,(size_t)tiles*16*sizeof(*sums));
		for (int64_t local=0; local<OPG; local++) {
			const int64_t oc=group*OPG+local;
			const uint8_t *src=w->qv+(uint64_t)oc*(uint64_t)K;
			int64_t *sum=sums+(local/16)*16+(local&15);
			zws[local]=(int32_t)(wz?cx_get(wz,wz->n==1?0:(uint64_t)oc):0)-
			           (w->type==3?0:128);
			if (qlinear)scales[local]=(double)xs->v[0]*
				                          ws->v[ws->n==1?0:(uint64_t)oc]/ys->v[0];
			uint8_t *dst=packed+(size_t)(local/16)*tile_bytes;
			for (int64_t p=0; p<K; p++) {
				int32_t v=w->type==3?(int32_t)(int8_t)src[p]:(int32_t)src[p]-128;
				*sum+=v;
				dst[(size_t)(p/4)*64+(size_t)(local&15)*4+(size_t)(p&3)]=(uint8_t)v;
			}
		}
		for (int64_t b=0; b<B; b++)for (int64_t h=0; h<HO; h++) {
				const int64_t ih=h-PH;
				if (ih<0||ih+KH>HI) {
					continue;
				}
				for (int64_t z=0; z<WO; z++) {
					const int64_t iw=z-PW;
					if (iw<0||iw+KW>WI) {
						continue;
					}
					int64_t suma=0;
					memset(ap,0,(size_t)blocks*sizeof(*ap));
					int64_t pos=0;
					for (int64_t ic=0; ic<CG; ic++)for (int64_t ky=0; ky<KH; ky++)for (int64_t kx=0; kx<KW; kx++,pos++) {
								uint64_t at=(((uint64_t)b*C+(uint64_t)group*CG+(uint64_t)ic)*
								             (uint64_t)HI+(uint64_t)(ih+ky))*(uint64_t)WI+(uint64_t)(iw+kx);
								uint32_t v=x->qv?(uint32_t)(x->type==3?(int32_t)(int8_t)x->qv[at]+128:
								                            (int32_t)x->qv[at]):
								           (uint32_t)((int32_t)x->v[at]+(x->type==3?128:0));
								ap[pos/4]|=v<<(8*(pos&3));
								suma+=v;
							}
					for (int64_t tile=0; tile<tiles; tile++) {
						__attribute__((aligned(64))) int32_t raw[16];
						cats_q8_conv2d_vnni_dot(ap,packed+(size_t)tile*tile_bytes,blocks,raw);
						for (int lane=0; lane<16; lane++) {
							int64_t local=tile*16+lane;
							if (local>=OPG) {
								break;
							}
							int64_t oc=group*OPG+local;
							int64_t zw_s=zws[local];
							int64_t sum=(int64_t)raw[lane]-zw_s*suma-
							            zx_u*sums[tile*16+lane]+K*zx_u*zw_s;
							uint64_t oi=((uint64_t)b*O+(uint64_t)oc)*(uint64_t)HO*(uint64_t)WO+
							            (uint64_t)h*(uint64_t)WO+(uint64_t)z;
							if (qlinear) {
								double acc=(double)sum+(bias?bias->v[oc]:0);
								double quant=nearbyint(acc*scales[local]+yz->v[0]);
								y->v[oi]=(float)fmin(fmax(quant,yz->type==2?0:-128),yz->type==2?255:127);
							} else {
								if (sum<-16777216LL||sum>16777216LL) {
									free(packed);
									free(sums);
									free(zws);
									free(scales);
									free(ap);
									return -1;
								}
								y->v[oi]=(float)sum;
							}
						}
					}
				}
			}
	}
	free(packed);
	free(sums);
	free(zws);
	free(scales);
	free(ap);
	return 1;
#else
	(void)x;
	(void)w;
	(void)xz;
	(void)wz;
	(void)xs;
	(void)ws;
	(void)ys;
	(void)yz;
	(void)bias;
	(void)y;
	(void)qlinear;
	(void)G;
	(void)HI;
	(void)WI;
	(void)HO;
	(void)WO;
	(void)KH;
	(void)KW;
	(void)PH;
	(void)PW;
	return 0;
#endif
}

/* Stage 10: ONNX ConvInteger / QLinearConv for 1D and 2D NCHW tensors.
 * Signed/unsigned 8-bit input and weight; group/depthwise, per-output-channel
 * weight scales/zero-points, optional int32 bias. This is a correctness-first
 * scalar kernel, not a SIMD-optimized implementation.
 * Float32 tensor storage cannot encode all int32 results exactly: ConvInteger
 * rejects magnitudes outside the exact [-2^24,2^24] range.
 */
static inline int cx_quant_conv_op(cx_env *e,const cx_node*n,cx_tensor **v)
{
	const int qlinear=cx_is(n->op,"QLinearConv");
	const cx_tensor *x=v[0],*w=qlinear?v[3]:v[1];
	const cx_tensor *xz=qlinear?v[2]:v[2],*wz=qlinear?v[5]:v[3];
	const cx_tensor *xs=qlinear?v[1]:NULL,*ws=qlinear?v[4]:NULL;
	const cx_tensor *ys=qlinear?v[6]:NULL,*yz=qlinear?v[7]:NULL;
	const cx_tensor *bias=(qlinear && n->ni>=9)?v[8]:NULL;
	if (!x||!w||x->nd<3||x->nd>4||w->nd!=x->nd ||
	                (x->type!=2&&x->type!=3)||(w->type!=2&&w->type!=3)||
	                (xz&&(xz->type!=x->type||xz->n!=1))||
	                (wz&&(wz->type!=w->type||(wz->n!=1 &&
	                                (wz->nd!=1||wz->n!=(uint64_t)w->d[0]))))) {
		return -1;
	}
	if (qlinear) {
		if (n->ni<8||!xs||!ws||!ys||!xz||!wz||!yz||
		                xs->n!=1||ys->n!=1||yz->n!=1||
		                (yz->type!=2&&yz->type!=3)||
		                (ws->n!=1&&(ws->nd!=1||ws->n!=(uint64_t)w->d[0]))||
		                (bias&&(bias->type!=6||bias->nd!=1||bias->n!=(uint64_t)w->d[0]))) {
			return -1;
		}
		if (!isfinite(xs->v[0])||xs->v[0]<=0||!isfinite(ys->v[0])||ys->v[0]<=0) {
			return -1;
		}
		for (uint64_t k=0; k<ws->n; k++)if (!isfinite(ws->v[k])||ws->v[k]<=0) {
				return -1;
			}
	}
	int sp=x->nd-2;
	const int64_t B=x->d[0],C=x->d[1],O=w->d[0],CG=w->d[1],G=cx_attr_int(n,"group",1);
	if (G<=0||C<0||O<0||C%G||O%G||CG!=C/G) {
		return -1;
	}
	int64_t id[2]= {1,1},kd[2]= {1,1},od[2]= {1,1};
	int64_t st[2]= {1,1},dil[2]= {1,1},pads[4]= {0};
	const cx_attr *pa=cx_attr_get(n,"pads"),*sa=cx_attr_get(n,"strides"),
	                                         *da=cx_attr_get(n,"dilations"),*au=cx_attr_get(n,"auto_pad");
	const char*auto_pad=au?au->sv:"NOTSET";
	int upper=!strcmp(auto_pad,"SAME_UPPER"),lower=!strcmp(auto_pad,"SAME_LOWER");
	if (strcmp(auto_pad,"NOTSET")&&strcmp(auto_pad,"VALID")&&!upper&&!lower) {
		return -1;
	}
	if ((pa&&pa->nints!=sp*2)||(sa&&sa->nints!=sp)||(da&&da->nints!=sp)||
	                (pa&&(upper||lower))) {
		return -1;
	}
	for (int i=0; i<sp; i++) {
		id[i]=x->d[i+2];
		kd[i]=w->d[i+2];
		if (sa) {
			st[i]=sa->ints[i];
		}
		if (da) {
			dil[i]=da->ints[i];
		}
		if (pa) {
			pads[i]=pa->ints[i];
			pads[sp+i]=pa->ints[sp+i];
		}
		if (id[i]<0||kd[i]<1||st[i]<1||dil[i]<1||pads[i]<0||pads[i+sp]<0||
		                kd[i]-1>(INT64_MAX-1)/dil[i]) {
			return -1;
		}
		const int64_t extent=(kd[i]-1)*dil[i]+1;
		if (upper||lower) {
			od[i]=id[i]/st[i]+(id[i]%st[i]!=0);
			if (od[i] && (od[i]-1)>(INT64_MAX-extent)/st[i]) {
				return -1;
			}
			int64_t padding=od[i]?(od[i]-1)*st[i]+extent-id[i]:0;
			if (padding<0) {
				padding=0;
			}
			pads[i]=upper?padding/2:padding-padding/2;
			pads[i+sp]=padding-pads[i];
		} else {
			if (id[i]>INT64_MAX-pads[i]||id[i]+pads[i]>INT64_MAX-pads[i+sp]) {
				return -1;
			}
			int64_t length=id[i]+pads[i]+pads[i+sp]-extent;
			od[i]=length>=0?length/st[i]+1:0;
		}
	}
	int64_t out_d[4]= {B,O,od[0],od[1]};
	cx_tensor*y=cx_env_new(e,n->output[0],out_d,sp+2,qlinear?yz->type:6);
	if (!y) {
		return -1;
	}
	const int64_t HI=sp==2?id[0]:1, WI=sp==2?id[1]:id[0];
	const int64_t HO=sp==2?od[0]:1, WO=sp==2?od[1]:od[0];
	const int64_t KH=sp==2?kd[0]:1, KW=sp==2?kd[1]:kd[0];
	const int64_t SH=sp==2?st[0]:1, SW=sp==2?st[1]:st[0];
	const int64_t DH=sp==2?dil[0]:1, DW=sp==2?dil[1]:dil[0];
	const int64_t PH=sp==2?pads[0]:0,PW=sp==2?pads[1]:pads[0];
	const int64_t output_per_group=O/G;
	/* Stage 24: precompute every interior output using 16-channel VNNI dot.
	   The ordinary loop below retains exact scalar boundary behavior. */
	int vnni_done=0;
	if (sp==2 && SH==1 && SW==1 && DH==1 && DW==1 &&
	                KH>0 && KW>0 && G>0 && O/G>=4) {
		vnni_done=cx_quant_conv_vnni_interior(x,w,xz,wz,xs,ws,ys,yz,
		                                      bias,y,qlinear,G,HI,WI,HO,WO,KH,KW,PH,PW);
		if (vnni_done<0) {
			return -1;
		}
	}

	/* Stage 22: bounded scratch (<=64 KiB) reused for each output channel.
	   Allocation failure is harmless: retain the Stage 21 AVX2 kernel. */
	int16_t *conv_coeff=NULL;
#if CATS_SIMD_X86
	if (sp==2 && CG>=2 && KH>0 && KW>0 && CG<=32768/KH/KW &&
	                cats_cpu_simd_level()>=2 && w->qv) {
		conv_coeff=(int16_t*)malloc((size_t)(CG*KH*KW)*sizeof(*conv_coeff));
	}
#endif
	const int64_t x_zero=xz?(int64_t)cx_get(xz,0):0;
	/* Stage 23: input validity depends on batch and group, not on the
	   output channel. Reuse the scan for every output channel in a group. */
	int64_t checked_batch=-1, checked_group=-1;
	int checked_valid=1;
	for (int64_t b=0; b<B; b++)for (int64_t oc=0; oc<O; oc++) {
			int64_t w_zero=wz?(int64_t)cx_get(wz,wz->n==1?0:(uint64_t)oc):0;
			/* Keep the ONNX scale calculation identical, but compute it once
			   per output channel rather than for every output pixel. */
			const double conv_scale=qlinear?
			                        (double)xs->v[0]*ws->v[ws->n==1?0:(uint64_t)oc]/ys->v[0]:0.0;
			int64_t group=oc/output_per_group;
			int dw_fast=0,dw2_fast=0,grp2_fast=0;
			int32_t conv_coeff_sum=0;
#if CATS_SIMD_X86
			/* Stage 20: 1-D and 2-D depthwise INT8 convolution. All interior
			   output tiles use 8-lane AVX2 integer MAC; padded boundaries use
			   the existing scalar reference for identical ONNX semantics. */
			int dw_common=(CG==1 && C==G && O==C && SW==1 && DW==1 &&
			               KW>0 && WO>=8 && WI>=8 && w->qv && (x->v||x->qv) &&
			               x_zero>=(x->type==3?-128:0) && x_zero<=(x->type==3?127:255) &&
			               w_zero>=(w->type==3?-128:0) && w_zero<=(w->type==3?127:255) &&
			               cats_cpu_simd_level()>=2);
			dw_fast=dw_common && sp==1 && KW<=32768;
			dw2_fast=dw_common && sp==2 && SH==1 && DH==1 &&
			         KH<=32768/KW;
			/* Stage 21: grouped/standard 2-D INT8 conv, all output channels.
			   Use the same strict input validity and integer overflow guards. */
			grp2_fast=(sp==2 && CG>=2 && SW==1 && DW==1 && SH==1 && DH==1 &&
			           KW>0 && KH>0 && CG<=32768/KH/KW && WO>=8 && WI>=8 &&
			           w->qv && (x->v||x->qv) && cats_cpu_simd_level()>=2 &&
			           x_zero>=(x->type==3?-128:0) && x_zero<=(x->type==3?127:255) &&
			           w_zero>=(w->type==3?-128:0) && w_zero<=(w->type==3?127:255));
			if ((dw_fast||dw2_fast||grp2_fast) && x->v) {
				if (checked_batch!=b || checked_group!=group) {
					checked_batch=b;
					checked_group=group;
					checked_valid=1;
					uint64_t base=((uint64_t)b*C+(uint64_t)group*CG)*(uint64_t)HI*WI;
					for (uint64_t q=0; q<(uint64_t)CG*(uint64_t)HI*(uint64_t)WI; q++) {
						float a=x->v[base+q];
						/* Range-check before float-to-int to avoid undefined conversion. */
						if (!isfinite(a)||a<(x->type==3?-128.f:0.f)||
						                a>(x->type==3?127.f:255.f)||a!=(float)(int32_t)a) {
							checked_valid=0;
							break;
						}
					}
				}
				if (!checked_valid) {
					dw_fast=0;
					dw2_fast=0;
					grp2_fast=0;
				}
			}
#endif
#if CATS_SIMD_X86
			if (grp2_fast && conv_coeff) {
				uint64_t wb=(uint64_t)oc*(uint64_t)CG*(uint64_t)KH*(uint64_t)KW;
				conv_coeff_sum=cats_q8_groupconv_pack_weights(w->qv+wb,
				               w->type==3,(int32_t)w_zero,CG*KH*KW,conv_coeff);
			}
#endif
			for (int64_t h=0; h<HO; h++)for (int64_t z=0; z<WO; z++) {
					if (vnni_done && h>=PH && h+KH-PH<=HI &&
					                z>=PW && z+KW-PW<=WI) {
						continue;
					}
#if CATS_SIMD_X86
					if (grp2_fast && z+7<WO && h>=PH && h+KH-PH<=HI &&
					                z>=PW && z+KW-PW+7<=WI) {
						int32_t sums[8];
						uint64_t xb=((uint64_t)b*C+(uint64_t)group*CG)*(uint64_t)HI*WI;
						uint64_t wb=(uint64_t)oc*(uint64_t)CG*(uint64_t)KH*(uint64_t)KW;
						if (conv_coeff)
							cats_q8_groupconv2d8_packed_avx2(x->v?x->v+xb:NULL,
							                                 x->qv?x->qv+xb:NULL,conv_coeff,x->type==3,
							                                 CG,HI,WI,KH,KW,h-PH,z-PW,(int32_t)x_zero,conv_coeff_sum,sums);
						else
							cats_q8_groupconv2d8_avx2(x->v?x->v+xb:NULL,
							                          x->qv?x->qv+xb:NULL,w->qv+wb,x->type==3,w->type==3,
							                          CG,HI,WI,KH,KW,h-PH,z-PW,(int32_t)x_zero,(int32_t)w_zero,sums);
						for (int q=0; q<8; q++) {
							uint64_t oi=((uint64_t)b*O+(uint64_t)oc)*(uint64_t)HO*WO+
							            (uint64_t)h*WO+(uint64_t)z+(uint64_t)q;
							if (qlinear) {
								double sc=conv_scale;
								double acc=(double)sums[q]+(bias?bias->v[oc]:0);
								double quant=nearbyint(acc*sc+yz->v[0]);
								y->v[oi]=(float)fmin(fmax(quant,yz->type==2?0:-128),yz->type==2?255:127);
							} else {
								if (sums[q]<-16777216||sums[q]>16777216) {
									goto conv_error;
								}
								y->v[oi]=(float)sums[q];
							}
						}
						z+=7;
						continue;
					}
					if (dw2_fast && z+7<WO && h>=PH && h+KH-PH<=HI &&
					                z>=PW && z+KW-PW+7<=WI) {
						int32_t sums[8];
						uint64_t xb=((uint64_t)b*C+(uint64_t)group)*(uint64_t)HI*WI;
						uint64_t wb=(uint64_t)oc*(uint64_t)KH*(uint64_t)KW;
						cats_q8_dwconv2d8_avx2(x->v?x->v+xb:NULL,
						                       x->qv?x->qv+xb:NULL,w->qv+wb,x->type==3,w->type==3,
						                       WI,KH,KW,h-PH,z-PW,(int32_t)x_zero,(int32_t)w_zero,sums);
						for (int q=0; q<8; q++) {
							uint64_t oi=((uint64_t)b*O+(uint64_t)oc)*(uint64_t)HO*WO+
							            (uint64_t)h*WO+(uint64_t)z+(uint64_t)q;
							if (qlinear) {
								double sc=conv_scale;
								double acc=(double)sums[q]+(bias?bias->v[oc]:0);
								double quant=nearbyint(acc*sc+yz->v[0]);
								y->v[oi]=(float)fmin(fmax(quant,yz->type==2?0:-128),yz->type==2?255:127);
							} else {
								if (sums[q]<-16777216||sums[q]>16777216) {
									goto conv_error;
								}
								y->v[oi]=(float)sums[q];
							}
						}
						z+=7;
						continue;
					}
					if (dw_fast && z>=PW && z+7<WO && z+KW-PW+7<=WI) {
						int32_t sums[8];
						uint64_t xb=((uint64_t)b*C+(uint64_t)group)*(uint64_t)WI;
						uint64_t wb=(uint64_t)oc*(uint64_t)KW;
						cats_q8_dwconv8_avx2(x->v?x->v+xb:NULL,x->qv?x->qv+xb:NULL,
						                     w->qv+wb,x->type==3,w->type==3,WI,KW,z,PW,
						                     (int32_t)x_zero,(int32_t)w_zero,sums);
						for (int q=0; q<8; q++) {
							uint64_t oi=((uint64_t)b*O+(uint64_t)oc)*(uint64_t)WO+(uint64_t)z+q;
							if (qlinear) {
								double sc=conv_scale;
								double acc=(double)sums[q]+(bias?bias->v[oc]:0);
								double quant=nearbyint(acc*sc+yz->v[0]);
								y->v[oi]=(float)fmin(fmax(quant,yz->type==2?0:-128),yz->type==2?255:127);
							} else {
								if (sums[q]<-16777216||sums[q]>16777216) {
									goto conv_error;
								}
								y->v[oi]=(float)sums[q];
							}
						}
						z+=7;
						continue;
					}
#endif
					int64_t sum=0;
					for (int64_t ic=0; ic<CG; ic++)for (int64_t kh=0; kh<KH; kh++)for (int64_t kw=0; kw<KW; kw++) {
								int64_t ih=h*SH+kh*DH-PH,iw=z*SW+kw*DW-PW;
								if (ih<0||ih>=HI||iw<0||iw>=WI) {
									continue;
								}
								uint64_t ix=((uint64_t)b*C+(uint64_t)(group*CG+ic))*(uint64_t)HI*WI+(uint64_t)ih*WI+iw;
								uint64_t wi=((uint64_t)oc*CG+(uint64_t)ic)*(uint64_t)KH*KW+(uint64_t)kh*KW+kw;
								int64_t xa=(int64_t)cx_get(x,ix)-x_zero;
								int64_t wa=(int64_t)cx_get(w,wi)-w_zero;
								sum+=xa*wa;
								if (sum<INT32_MIN||sum>INT32_MAX) {
									goto conv_error;
								}
							}
					uint64_t out_at=((uint64_t)b*O+(uint64_t)oc)*(uint64_t)HO*WO+(uint64_t)h*WO+z;
					if (qlinear) {
						double sc=conv_scale;
						double acc=(double)sum+(bias?bias->v[oc]:0);
						double quant=nearbyint(acc*sc+yz->v[0]);
						y->v[out_at]=(float)fmin(fmax(quant,yz->type==2?0:-128),yz->type==2?255:127);
					} else {
						if (sum< -16777216LL||sum>16777216LL) {
							goto conv_error;
						}
						y->v[out_at]=(float)sum;
					}
				}
		}
	free(conv_coeff);
	return 0;
conv_error:
	free(conv_coeff);
	return -1;
}


/* ========================================================================
 * GatherElements and ScatterElements
 * ======================================================================== */

/* ONNX GatherElements / ScatterElements subset.
 * MPL-2.0. Defined in this header after
 * cx_i64 and tensor helpers have been defined. C11 + libm only.
 */

/* GatherElements: output shape equals indices; no implicit broadcasting. */
static inline int cx_gather_elements(cx_env *e,const cx_node *node,
                                     const cx_tensor *data,const cx_tensor *indices)
{
	if (!data||!indices||data->nd<1||data->nd!=indices->nd||
	                (indices->type!=6&&indices->type!=7)||node->ni!=2||node->no!=1) {
		return -1;
	}
	int axis=(int)cx_attr_int(node,"axis",0);
	if (cx_axis(axis,data->nd,&axis)) {
		return -1;
	}
	for (int k=0; k<data->nd; k++)if (k!=axis&&indices->d[k]>data->d[k]) {
			return -1;
		}
	cx_tensor *out=cx_env_new(e,node->output[0],indices->d,indices->nd,data->type);
	if (!out) {
		return -1;
	}
	int64_t coords[CX_DIMS],source[CX_DIMS];
	for (uint64_t i=0; i<out->n; i++) {
		cx_unravel(i,indices->d,indices->nd,coords);
		memcpy(source,coords,(size_t)data->nd*sizeof(int64_t));
		int64_t selected=0;
		if (cx_i64(cx_get(indices,i),&selected)) {
			return -1;
		}
		if (selected<0) {
			selected+=data->d[axis];
		}
		if (selected<0||selected>=data->d[axis]) {
			return -1;
		}
		source[axis]=selected;
		out->v[i]=cx_get(data,cx_index(data->d,data->nd,source));
	}
	return 0;
}
/* ScatterElements: indices/updates have identical shapes. With reduction
 * 'none', repeated output coordinates are invalid (not deterministic). */
static inline int cx_scatter_elements(cx_env *e,const cx_node *node,
                                      const cx_tensor *data,const cx_tensor *indices,const cx_tensor *updates)
{
	if (!data||!indices||!updates||data->nd<1||data->nd!=indices->nd||
	                data->nd!=updates->nd||data->type!=updates->type||
	                (indices->type!=6&&indices->type!=7)||node->ni!=3||node->no!=1||
	                !cx_same_shape(indices,updates)) {
		return -1;
	}
	int axis=(int)cx_attr_int(node,"axis",0);
	if (cx_axis(axis,data->nd,&axis)) {
		return -1;
	}
	for (int k=0; k<data->nd; k++)if (k!=axis&&indices->d[k]>data->d[k]) {
			return -1;
		}
	const cx_attr*att=cx_attr_get(node,"reduction");
	const char *mode=att&&att->sv[0]?att->sv:"none";
	if (strcmp(mode,"none")&&strcmp(mode,"add")&&strcmp(mode,"mul")&&
	                strcmp(mode,"max")&&strcmp(mode,"min")) {
		return -1;
	}
	cx_tensor *out=cx_env_new(e,node->output[0],data->d,data->nd,data->type);
	if (!out) {
		return -1;
	}
	for (uint64_t i=0; i<data->n; i++) {
		out->v[i]=cx_get(data,i);
	}
	unsigned char *seen=NULL;
	if (!strcmp(mode,"none")) {
		if (out->n>SIZE_MAX) {
			return -1;
		}
		seen=(unsigned char*)calloc((size_t)(out->n?out->n:1),1);
		if (!seen) {
			return -1;
		}
	}
	int64_t coords[CX_DIMS],dest[CX_DIMS];
	int rc=-1;
	for (uint64_t i=0; i<indices->n; i++) {
		cx_unravel(i,indices->d,indices->nd,coords);
		memcpy(dest,coords,(size_t)data->nd*sizeof(int64_t));
		int64_t selected=0;
		if (cx_i64(cx_get(indices,i),&selected)) {
			goto done;
		}
		if (selected<0) {
			selected+=data->d[axis];
		}
		if (selected<0||selected>=data->d[axis]) {
			goto done;
		}
		dest[axis]=selected;
		uint64_t p=cx_index(data->d,data->nd,dest);
		if (seen && seen[p]++) {
			goto done;
		}
		float value=cx_get(updates,i);
		if (!strcmp(mode,"none")) {
			out->v[p]=value;
		} else if (!strcmp(mode,"add")) {
			out->v[p]+=value;
		} else if (!strcmp(mode,"mul")) {
			out->v[p]*=value;
		} else if (!strcmp(mode,"max")) {
			out->v[p]=fmaxf(out->v[p],value);
		} else {
			out->v[p]=fminf(out->v[p],value);
		}
	}
	rc=0;
done: free(seen);
	return rc;
}


/* ========================================================================
 * Additional ONNX operators
 * ======================================================================== */

/* Additional common ONNX kernels: C11 + libm only, no third-party dependencies.
 * Defined after the primary numeric kernels.
 * Reference implementations. Unsupported attributes fail instead of approximating.
 */

static inline int cx_more_supported(const char *op)
{
	static const char *names[]= {"Split","Pad","ReduceSum","ReduceMax","ReduceMin",
	                             "ArgMax","ArgMin","LogSoftmax","CumSum","Clip","Max","Min","Erf",
	                             "LeakyRelu","Elu","Softplus","Not","And","Or"
	                            };
	for (size_t i=0; i<sizeof(names)/sizeof(*names); i++)if (cx_is(op,names[i])) {
			return 1;
		}
	return 0;
}
static inline int cx_more_unary(cx_env *e,const cx_node *n,const cx_tensor *x)
{
	cx_tensor *y=cx_env_new(e,n->output[0],x->d,x->nd,cx_is(n->op,"Not")?9:1);
	if (!y) {
		return -1;
	}
	float alpha=cx_attr_float(n,"alpha",cx_is(n->op,"LeakyRelu")?0.01f:1.f);
	for (uint64_t k=0; k<x->n; k++) {
		float v=x->v[k];
		if (cx_is(n->op,"Erf")) {
			v=erff(v);
		} else if (cx_is(n->op,"LeakyRelu")) {
			v=v>=0?v:v*alpha;
		} else if (cx_is(n->op,"Elu")) {
			v=v>=0?v:alpha*expm1f(v);
		} else if (cx_is(n->op,"Softplus")) {
			v=fmaxf(v,0)+log1pf(expf(-fabsf(v)));
		} else if (cx_is(n->op,"Not")) {
			v=v==0?1.f:0.f;
		}
		y->v[k]=v;
	}
	return 0;
}
static inline int cx_more_clip(cx_env *e,const cx_node*n,cx_tensor**v)
{
	cx_tensor*x=v[0];
	float lo=-INFINITY,hi=INFINITY;
	if (n->ni>1 && v[1]) {
		if (v[1]->n!=1) {
			return -1;
		}
		lo=v[1]->v[0];
	} else {
		lo=cx_attr_float(n,"min",lo);
	}
	if (n->ni>2 && v[2]) {
		if (v[2]->n!=1) {
			return -1;
		}
		hi=v[2]->v[0];
	} else {
		hi=cx_attr_float(n,"max",hi);
	}
	if (lo>hi) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,x->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<x->n; i++) {
		y->v[i]=fminf(fmaxf(x->v[i],lo),hi);
	}
	return 0;
}
static inline int cx_more_multinary(cx_env*e,const cx_node*n,cx_tensor**v)
{
	if (n->ni<2||!v[0]||!v[1]) {
		return -1;
	}
	const char *op=n->op;
	int64_t dims[CX_DIMS]= {0},scratch[CX_DIMS]= {0},coord[CX_DIMS];
	int nd=0,old=0;
	cx_tensor fake= {0};
	fake.nd=0;
	for (int j=0; j<n->ni; j++) {
		if (!v[j]) {
			return -1;
		}
		if (cx_binary_dims(j?&fake:v[j],v[j],scratch,&nd)) {
			return -1;
		}
		if (j==0) {
			nd=v[j]->nd;
			memcpy(dims,v[j]->d,sizeof(dims));
		} else {
			memcpy(dims,scratch,(size_t)nd*sizeof(int64_t));
		}
		fake.nd=nd;
		memcpy(fake.d,dims,(size_t)nd*sizeof(int64_t));
		old=nd;
	}
	(void)old;
	int ismax=cx_is(op,"Max"),ismin=cx_is(op,"Min"),logic=cx_is(op,"And")||cx_is(op,"Or");
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,logic?9:1);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,nd,coord);
		float result=0;
		for (int j=0; j<n->ni; j++) {
			const cx_tensor*t=v[j];
			uint64_t at=0;
			for (int z=0; z<t->nd; z++) {
				at=at*(uint64_t)t->d[z]+(t->d[z]==1?0:(uint64_t)coord[nd-t->nd+z]);
			}
			float a=t->v[at];
			if (!j) {
				result=a;
			} else if (ismax) {
				result=fmaxf(result,a);
			} else if (ismin) {
				result=fminf(result,a);
			} else if (cx_is(op,"And")) {
				result=(result!=0 && a!=0)?1.f:0.f;
			} else {
				result=(result!=0 || a!=0)?1.f:0.f;
			}
		}
		y->v[i]=result;
	}
	return 0;
}
static inline int cx_more_logsoftmax(cx_env*e,const cx_node*n,const cx_tensor*x)
{
	int axis=(int)cx_attr_int(n,"axis",-1);
	if (cx_axis(axis,x->nd,&axis)) {
		return -1;
	}
	uint64_t outer=1,inner=1,width=(uint64_t)x->d[axis];
	if (cx_prod(x->d,0,axis,&outer)||cx_prod(x->d,axis+1,x->nd,&inner)||!width) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,1);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<outer; i++)for (uint64_t j=0; j<inner; j++) {
			uint64_t idx=i*width*inner+j;
			float hi=-INFINITY;
			for (uint64_t k=0; k<width; k++)if (x->v[idx+k*inner]>hi) {
					hi=x->v[idx+k*inner];
				}
			double s=0;
			for (uint64_t k=0; k<width; k++) {
				s+=exp((double)x->v[idx+k*inner]-hi);
			}
			float logsum=hi+(float)log(s);
			for (uint64_t k=0; k<width; k++) {
				y->v[idx+k*inner]=x->v[idx+k*inner]-logsum;
			}
		}
	return 0;
}
static inline int cx_more_reduce(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*axes)
{
	int mask[CX_DIMS]= {0};
	const cx_attr*att=cx_attr_get(n,"axes");
	int count=axes?(int)axes->n:(att?att->nints:0);
	if (count>CX_DIMS) {
		return -1;
	}
	if (count==0 && !cx_attr_int(n,"noop_with_empty_axes",0)) {
		for (int i=0; i<x->nd; i++) {
			mask[i]=1;
		}
	} else for (int i=0; i<count; i++) {
			int64_t v;
			if (axes) {
				if (cx_i64(axes->v[i],&v)) {
					return -1;
				}
			} else {
				v=att->ints[i];
			}
			int ax=(int)v;
			if (cx_axis(ax,x->nd,&ax)||mask[ax]) {
				return -1;
			}
			mask[ax]=1;
		}
	int keep=(int)cx_attr_int(n,"keepdims",1);
	int64_t dims[CX_DIMS],src[CX_DIMS],dst[CX_DIMS];
	int nd=0;
	for (int i=0; i<x->nd; i++)if (!mask[i]||keep) {
			dims[nd++]=mask[i]?1:x->d[i];
		}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,nd,1);
	if (!y) {
		return -1;
	}
	int sum=cx_is(n->op,"ReduceSum"),ismax=cx_is(n->op,"ReduceMax");
	for (uint64_t i=0; i<y->n; i++) {
		y->v[i]=sum?0:ismax?-INFINITY:INFINITY;
	}
	for (uint64_t i=0; i<x->n; i++) {
		cx_unravel(i,x->d,x->nd,src);
		int j=0;
		for (int k=0; k<x->nd; k++)if (!mask[k]||keep) {
				dst[j++]=mask[k]?0:src[k];
			}
		uint64_t at=cx_index(dims,nd,dst);
		float v=x->v[i];
		if (sum) {
			y->v[at]+=v;
		} else if (ismax) {
			y->v[at]=fmaxf(y->v[at],v);
		} else {
			y->v[at]=fminf(y->v[at],v);
		}
	}
	return 0;
}
static inline int cx_more_arg(cx_env*e,const cx_node*n,const cx_tensor*x)
{
	int axis=(int)cx_attr_int(n,"axis",0);
	if (cx_axis(axis,x->nd,&axis)||x->d[axis]<=0) {
		return -1;
	}
	int keep=(int)cx_attr_int(n,"keepdims",1),last=(int)cx_attr_int(n,"select_last_index",0);
	int64_t d[CX_DIMS];
	int nd=0;
	for (int k=0; k<x->nd; k++)if (k!=axis||keep) {
			d[nd++]=k==axis?1:x->d[k];
		}
	cx_tensor*y=cx_env_new(e,n->output[0],d,nd,7);
	if (!y) {
		return -1;
	}
	uint64_t outer=1,inner=1,width=(uint64_t)x->d[axis];
	if (cx_prod(x->d,0,axis,&outer)||cx_prod(x->d,axis+1,x->nd,&inner)) {
		return -1;
	}
	int ismax=cx_is(n->op,"ArgMax");
	for (uint64_t i=0; i<outer; i++)for (uint64_t j=0; j<inner; j++) {
			uint64_t at=i*width*inner+j,best=0;
			float value=x->v[at];
			for (uint64_t k=1; k<width; k++) {
				float v=x->v[at+k*inner];
				int better=ismax?(v>value ||(last&&v==value)):(v<value ||(last&&v==value));
				if (better) {
					value=v;
					best=k;
				}
			}
			if (best>16777216u) {
				return -1;
			}
			y->v[i*inner+j]=(float)best;
		}
	return 0;
}
static inline int cx_more_cumsum(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*axisv)
{
	if (!axisv||axisv->n!=1) {
		return -1;
	}
	int64_t v;
	if (cx_i64(axisv->v[0],&v)) {
		return -1;
	}
	int axis=(int)v;
	if (cx_axis(axis,x->nd,&axis)) {
		return -1;
	}
	int exclusive=(int)cx_attr_int(n,"exclusive",0),reverse=(int)cx_attr_int(n,"reverse",0);
	uint64_t outer=1,inner=1,width=(uint64_t)x->d[axis];
	if (cx_prod(x->d,0,axis,&outer)||cx_prod(x->d,axis+1,x->nd,&inner)) {
		return -1;
	}
	cx_tensor*y=cx_env_new(e,n->output[0],x->d,x->nd,x->type);
	if (!y) {
		return -1;
	}
	for (uint64_t i=0; i<outer; i++)for (uint64_t j=0; j<inner; j++) {
			float running=0;
			for (uint64_t k=0; k<width; k++) {
				uint64_t ak=reverse?width-k-1:k,at=(i*width+ak)*inner+j;
				float val=x->v[at];
				if (!exclusive) {
					running+=val;
				}
				y->v[at]=running;
				if (exclusive) {
					running+=val;
				}
			}
		}
	return 0;
}
static inline int cx_more_split(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*lengths)
{
	int axis=(int)cx_attr_int(n,"axis",0);
	if (cx_axis(axis,x->nd,&axis)||n->no<1) {
		return -1;
	}
	const cx_attr*attr=cx_attr_get(n,"split");
	if (lengths&&lengths->n!=(uint64_t)n->no) {
		return -1;
	}
	if (!lengths && attr && attr->nints!=n->no) {
		return -1;
	}
	int64_t parts[CX_IO],sum=0;
	for (int k=0; k<n->no; k++) {
		int64_t a;
		if (lengths) {
			if (cx_i64(lengths->v[k],&a)) {
				return -1;
			}
		} else if (attr) {
			a=attr->ints[k];
		} else {
			int64_t w=x->d[axis],q=w/n->no,r=w%n->no;
			a=q+(k<r);
		}
		if (a<0 || a>INT64_MAX-sum) {
			return -1;
		}
		sum+=a;
		parts[k]=a;
	}
	if (sum!=x->d[axis]) {
		return -1;
	}
	uint64_t outer=1,inner=1;
	if (cx_prod(x->d,0,axis,&outer)||cx_prod(x->d,axis+1,x->nd,&inner)) {
		return -1;
	}
	int64_t at=0;
	for (int k=0; k<n->no; k++) {
		if (!n->output[k][0]) {
			at+=parts[k];
			continue;
		}
		int64_t dims[CX_DIMS];
		memcpy(dims,x->d,(size_t)x->nd*sizeof(int64_t));
		dims[axis]=parts[k];
		cx_tensor*y=cx_env_new(e,n->output[k],dims,x->nd,x->type);
		if (!y) {
			return -1;
		}
		uint64_t ncopy=(uint64_t)parts[k]*inner;
		for (uint64_t b=0; b<outer; b++) {
			memcpy(y->v+b*ncopy,x->v+(b*(uint64_t)x->d[axis]+(uint64_t)at)*inner,(size_t)ncopy*sizeof(float));
		}
		at+=parts[k];
	}
	return 0;
}
static inline int cx_more_pad(cx_env*e,const cx_node*n,const cx_tensor*x,const cx_tensor*pads,
                              const cx_tensor*value,const cx_tensor*axes)
{
	if (!pads) {
		return -1;
	}
	const cx_attr*legacy=cx_attr_get(n,"mode");
	const char*mode=legacy&&legacy->sv[0]?legacy->sv:"constant";
	if (strcmp(mode,"constant")&&strcmp(mode,"edge")) {
		return -1;
	}
	if (value&&value->n!=1) {
		return -1;
	}
	int64_t lo[CX_DIMS]= {0},hi[CX_DIMS]= {0},dims[CX_DIMS],coord[CX_DIMS],src[CX_DIMS];
	int rank=axes?(int)axes->n:x->nd;
	if (rank<0||rank>CX_DIMS||pads->n!=(uint64_t)(rank*2)) {
		return -1;
	}
	int seen[CX_DIMS]= {0};
	for (int k=0; k<rank; k++) {
		int64_t ax=k;
		if (axes && cx_i64(axes->v[k],&ax)) {
			return -1;
		}
		int a=(int)ax;
		if (cx_axis(a,x->nd,&a)||seen[a]++) {
			return -1;
		}
		if (cx_i64(pads->v[k],&lo[a])||cx_i64(pads->v[k+rank],&hi[a])) {
			return -1;
		}
	}
	for (int k=0; k<x->nd; k++) {
		if (lo[k]>INT64_MAX-x->d[k] || hi[k]>INT64_MAX-x->d[k]-lo[k]) {
			return -1;
		}
		dims[k]=x->d[k]+lo[k]+hi[k];
		if (dims[k]<0) {
			return -1;
		}
	}
	cx_tensor*y=cx_env_new(e,n->output[0],dims,x->nd,x->type);
	if (!y) {
		return -1;
	}
	float val=value?value->v[0]:cx_attr_float(n,"value",0);
	for (uint64_t i=0; i<y->n; i++) {
		cx_unravel(i,dims,x->nd,coord);
		int valid=1;
		for (int k=0; k<x->nd; k++) {
			int64_t at=coord[k]-lo[k];
			if (at<0||at>=x->d[k]) {
				if (cx_is(mode,"edge")&&x->d[k]>0) {
					if (at<0) {
						at=0;
					} else {
						at=x->d[k]-1;
					}
				} else {
					valid=0;
					break;
				}
			}
			src[k]=at;
		}
		y->v[i]=valid?x->v[cx_index(x->d,x->nd,src)]:val;
	}
	return 0;
}
static inline int cx_more_execute(cx_graph*g,cx_env*e,const cx_node*n,cx_tensor**v)
{
	(void)g;
	const char*op=n->op;
	if (cx_is(op,"Erf")||cx_is(op,"LeakyRelu")||cx_is(op,"Elu")||cx_is(op,"Softplus")||cx_is(op,"Not")) {
		return cx_more_unary(e,n,v[0]);
	}
	if (cx_is(op,"Clip")) {
		return cx_more_clip(e,n,v);
	}
	if (cx_is(op,"Max")||cx_is(op,"Min")||cx_is(op,"And")||cx_is(op,"Or")) {
		return cx_more_multinary(e,n,v);
	}
	if (cx_is(op,"LogSoftmax")) {
		return cx_more_logsoftmax(e,n,v[0]);
	}
	if (cx_is(op,"ReduceSum")||cx_is(op,"ReduceMax")||cx_is(op,"ReduceMin")) {
		return cx_more_reduce(e,n,v[0],n->ni>1?v[1]:NULL);
	}
	if (cx_is(op,"ArgMax")||cx_is(op,"ArgMin")) {
		return cx_more_arg(e,n,v[0]);
	}
	if (cx_is(op,"CumSum")) {
		return n->ni>1?cx_more_cumsum(e,n,v[0],v[1]):-1;
	}
	if (cx_is(op,"Split")) {
		return cx_more_split(e,n,v[0],n->ni>1?v[1]:NULL);
	}
	if (cx_is(op,"Pad")) {
		return n->ni>1?cx_more_pad(e,n,v[0],v[1],n->ni>2?v[2]:NULL,n->ni>3?v[3]:NULL):-1;
	}
	return -1;
}


static inline int cx_execute_node(cx_graph*g,cx_env*e,const cx_node*n)
{
	cx_tensor*in[CX_IO]= {0};
	for (int i=0; i<n->ni; i++) {
		if (n->input[i][0] && !(in[i]=cx_resolve(g,e,n->input[i]))) {
			return -1;
		}
	}
	const char*op=n->op;
	if (cx_is(op,"Constant")) {
		return cx_constant(g,e,n);
	}
	if (n->ni<1||!in[0]||n->no<1) {
		return -1;
	}
	if (cx_is(op,"Identity")) {
		return cx_tensor_copy(e,n->output[0],in[0]);
	}
	if (cx_is(op,"Cast")) {
		int t=(int)cx_attr_int(n,"to",0);
		if (t!=1&&t!=2&&t!=3&&t!=6&&t!=7&&t!=9&&t!=10) {
			return -1;
		}
		cx_tensor*y=cx_env_new(e,n->output[0],in[0]->d,in[0]->nd,t);
		if (!y) {
			return -1;
		}
		for (uint64_t i=0; i<y->n; i++) {
			float v=in[0]->v[i];
			if (!isfinite(v)) {
				return -1;
			}
			if (t==2) {
				v=fminf(fmaxf(truncf(v),0),255);
			}
			if (t==3) {
				v=fminf(fmaxf(truncf(v),-128),127);
			}
			if (t==6||t==7) {
				v=truncf(v);
			}
			if (t==9) {
				v=v!=0?1.f:0.f;
			}
			y->v[i]=v;
		}
		return 0;
	}
	if (cx_is(op,"Relu")||cx_is(op,"Sigmoid")||cx_is(op,"Tanh")||cx_is(op,"Exp")||cx_is(op,"Log")||cx_is(op,"Sqrt")||cx_is(op,"Neg")||cx_is(op,"Abs")||cx_is(op,"SiLU")||cx_is(op,"Swish")) {
		return cx_unary(e,n,in[0]);
	}
	if (cx_is(op,"Add")||cx_is(op,"Sub")||cx_is(op,"Mul")||cx_is(op,"Div")||cx_is(op,"Pow")) {
		return n->ni>=2&&in[1]?cx_binary(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"MatMul")) {
		return n->ni>=2&&in[1]?cx_matmul(e,n->output[0],in[0],in[1]):-1;
	}
	if (cx_is(op,"Gemm")) {
		if (n->ni<2||!in[1]||in[0]->nd!=2||in[1]->nd!=2) {
			return -1;
		}
		int transa=(int)cx_attr_int(n,"transA",0),transb=(int)cx_attr_int(n,"transB",0);
		int64_t m=transa?in[0]->d[1]:in[0]->d[0], k=transa?in[0]->d[0]:in[0]->d[1];
		int64_t kb=transb?in[1]->d[1]:in[1]->d[0],dn=transb?in[1]->d[0]:in[1]->d[1];
		if (k!=kb) {
			return -1;
		}
		int64_t dims[]= {m,dn};
		cx_tensor*y=cx_env_new(e,n->output[0],dims,2,1);
		if (!y) {
			return -1;
		}
		float alpha=cx_attr_float(n,"alpha",1.f),beta=cx_attr_float(n,"beta",1.f);
		if (!transa && !transb) {
			cats_sgemm_f32(in[0]->v,in[1]->v,y->v,m,dn,k);
		}
		for (int64_t i=0; i<m; i++)for (int64_t j=0; j<dn; j++) {
				float sum=0;
				if (!transa && !transb) {
					sum=y->v[i*dn+j];
				} else for (int64_t t=0; t<k; t++) {
						float a=in[0]->v[transa?t*m+i:i*k+t];
						float b=in[1]->v[transb?j*k+t:t*dn+j];
						sum+=a*b;
					}
				float c=0;
				if (n->ni>2&&in[2]) {
					const cx_tensor*cb=in[2];
					/* ONNX Gemm C is unidirectionally broadcastable to [M,N].
					   In particular, [M,1], [1,N], [1,1], [N] and scalars are valid. */
					if (cb->nd==0) {
						c=cb->v[0];
					} else if (cb->nd==1 && (cb->d[0]==1||cb->d[0]==dn)) {
						c=cb->v[cb->d[0]==1?0:j];
					} else if (cb->nd==2 && (cb->d[0]==1||cb->d[0]==m) &&
					                (cb->d[1]==1||cb->d[1]==dn)) {
						c=cb->v[(cb->d[0]==1?0:i)*cb->d[1]+(cb->d[1]==1?0:j)];
					} else {
						return -1;
					}
				}
				y->v[i*dn+j]=alpha*sum+beta*c;
			}
		return 0;
	}
	if (cx_is(op,"Reshape")) {
		return n->ni>=2&&in[1]?cx_reshape(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"Flatten")) {
		int axis=(int)cx_attr_int(n,"axis",1);
		if (axis<0) {
			axis+=in[0]->nd;
		}
		if (axis<0||axis>in[0]->nd) {
			return -1;
		}
		int64_t d[]= {1,1};
		for (int k=0; k<axis; k++) {
			d[0]*=in[0]->d[k];
		}
		for (int k=axis; k<in[0]->nd; k++) {
			d[1]*=in[0]->d[k];
		}
		cx_tensor*y=cx_env_new(e,n->output[0],d,2,1);
		if (!y) {
			return -1;
		}
		memcpy(y->v,in[0]->v,(size_t)y->n*sizeof(float));
		return 0;
	}
	if (cx_is(op,"Transpose")) {
		return cx_transpose(e,n,in[0]);
	}
	if (cx_is(op,"Concat")) {
		return cx_concat(e,n,in);
	}
	if (cx_is(op,"Gather")) {
		return n->ni>=2&&in[1]?cx_gather(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"GatherElements")) {
		return n->ni==2&&in[1]?cx_gather_elements(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"ScatterElements")) {
		return n->ni==3&&in[1]&&in[2]?cx_scatter_elements(e,n,in[0],in[1],in[2]):-1;
	}
	if (cx_is(op,"Softmax")) {
		return cx_softmax(e,n,in[0]);
	}
	if (cx_is(op,"ReduceMean")) {
		return cx_reduce_mean(e,n,in[0],n->ni>1?in[1]:NULL);
	}
	if (cx_is(op,"LayerNormalization")) {
		return n->ni>=2&&in[1]?cx_layernorm(e,n,in[0],in[1],n->ni>2?in[2]:NULL):-1;
	}
	if (cx_is(op,"Unsqueeze")||cx_is(op,"Squeeze")) {
		return cx_squeeze(e,n,in[0],n->ni>1?in[1]:NULL,cx_is(op,"Unsqueeze"));
	}
	if (cx_is(op,"DequantizeLinear")||cx_is(op,"QuantizeLinear")) {
		return n->ni>=2&&in[1]?cx_quant_axis(e,n,in[0],in[1],n->ni>2?in[2]:NULL,cx_is(op,"DequantizeLinear")):-1;
	}
	if (cx_is(op,"Shape")) {
		return cx_shape_op(e,n,in[0]);
	}
	if (cx_is(op,"Size")) {
		return cx_size_op(e,n,in[0]);
	}
	if (cx_is(op,"Slice")) {
		return n->ni>=3?cx_slice_op(e,n,in):-1;
	}
	if (cx_is(op,"Expand")) {
		return n->ni>=2?cx_expand_op(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"Tile")) {
		return n->ni>=2?cx_tile_op(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"Range")) {
		return n->ni>=3?cx_range_op(e,n,in):-1;
	}
	if (cx_is(op,"ConstantOfShape")) {
		return cx_constant_of_shape_op(e,g,n,in[0]);
	}
	if (cx_is(op,"Where")) {
		return n->ni>=3?cx_conditional_op(e,n,in[0],in[1],in[2]):-1;
	}
	if (cx_is(op,"Equal")||cx_is(op,"Greater")||cx_is(op,"Less")||cx_is(op,"GreaterOrEqual")||cx_is(op,"LessOrEqual")) {
		return n->ni>=2&&in[1]?cx_compare_op(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"DynamicQuantizeLinear")) {
		return cx_dynamic_quant(e,n,in[0]);
	}
	if (cx_is(op,"MatMulInteger")) {
		return n->ni>=2?cx_matmul_integer(e,n,in):-1;
	}
	if (cx_is(op,"QLinearMatMul")) {
		return n->ni>=8?cx_qlinear_matmul(e,n,in):-1;
	}
	if (cx_is(op,"Conv")) {
		return n->ni>=2?cx_conv_op(e,n,in):-1;
	}
	if (cx_is(op,"BatchNormalization")) {
		return cx_batchnorm(e,n,in);
	}
	if (cx_is(op,"PRelu")) {
		return n->ni==2&&in[1]?cx_prelu(e,n,in[0],in[1]):-1;
	}
	if (cx_is(op,"ConvInteger")||cx_is(op,"QLinearConv")) {
		return cx_quant_conv_op(e,n,in);
	}
	if (cx_more_supported(op)) {
		return cx_more_execute(g,e,n,in);
	}
	return -1;
}
/* Stage 15: optional deterministic per-node trace, for locating the first
 * divergence against a reference ONNX engine.  Emits raw IEEE-754 bit hashes,
 * full shapes and finite statistics BEFORE compact execution evicts outputs.
 * The trace is diagnostic only; it does not imply numerical parity. */
static inline void cx_trace_after(cx_graph*g,const cx_env*e,
                                  const cx_node*n,size_t node_index)
{
	if (!g || !g->trace || !e || !n) {
		return;
	}
	FILE *fp=g->trace;
	for (int j=0; j<n->no; j++) {
		if (!n->output[j][0]) {
			continue;
		}
		const cx_tensor*t=cx_env_find((cx_env*)e,n->output[j]);
		if (!t) {
			fprintf(fp,"TRACE_ERROR\t%zu\t%s\t%s\tmissing\n",node_index,n->op,n->output[j]);
			continue;
		}
		double sum=0;
		float lo=INFINITY,hi=-INFINITY;
		uint64_t hash=UINT64_C(14695981039346656037),nan_count=0;
		for (uint64_t k=0; k<t->n; k++) {
			float v=cx_get(t,k);
			uint32_t bits=0;
			memcpy(&bits,&v,sizeof(bits));
			for (int b=0; b<4; b++) {
				hash^=(bits>>(b*8))&255u;
				hash*=UINT64_C(1099511628211);
			}
			if (isfinite(v)) {
				sum+=v;
				if (v<lo) {
					lo=v;
				}
				if (v>hi) {
					hi=v;
				}
			} else {
				nan_count++;
			}
		}
		fprintf(fp,"%s\t%zu\t%s\t%s\t%s\t[",
		        g->trace_name?g->trace_name:"graph",node_index,n->op,n->output[j],cx_type(t->type));
		for (int d=0; d<t->nd; d++) {
			fprintf(fp,"%s%" PRId64,d?",":"",t->d[d]);
		}
		fprintf(fp,"]\t%" PRIu64 "\t%.9g\t%.9g\t%.17g\t%" PRIu64 "\t%016" PRIx64 "\n",
		        t->n,t->n==nan_count?NAN:lo,t->n==nan_count?NAN:hi,
		        t->n==nan_count?NAN:sum/(double)(t->n-nan_count),nan_count,hash);
	}
}
static inline int cx_graph_run(cx_graph*g,cx_env*e)
{
	if (!g||!e) {
		return -1;
	}
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*n=&g->node[i];
		const char*issue=cx_node_issue(g,n);
		if (issue) {
			if (g->profiler) {
				g->profiler->stats[i].failures++;
			}
			fprintf(stderr,"native ONNX: blocked node %zu (%s): %s\n",i,n->op,issue);
			return -1;
		}
		uint64_t profile_start=g->profiler?cx_profile_clock_ns():0;
		int node_failed=cx_execute_node(g,e,n);
		if (g->profiler) {
			cx_profiler_record(g->profiler,i,profile_start,node_failed);
		}
		if (node_failed) {
			fprintf(stderr,"native ONNX: node %zu (%s) failed (missing tensor, shape or datatype)\n",i,n->op);
			return -1;
		}
		cx_trace_after(g,e,n,i);
	}
	return 0;
}


/* Memory-saving execution mode. Keep graph inputs/outputs, drop temporary
 * tensors immediately after their last consumer.
 *
 * The old implementation rescanned all remaining nodes for every consumed
 * tensor: O(N^2) on long chains. Stage 9 builds a stable open-addressing
 * index once per model and then does O(1) average-case lookups. Names live in
 * immutable cx_node arrays, so entries hold no extra string allocations.
 * It is safe to call the compact runner repeatedly; the index is owned by g.
 */
static inline int cx_graph_keep(const cx_graph*g,const char*name)
{
	for (int j=0; j<g->ninput; j++)if (!strcmp(g->input[j],name)) {
			return 1;
		}
	for (int j=0; j<g->noutput; j++)if (!strcmp(g->output[j],name)) {
			return 1;
		}
	return 0;
}
static inline uint64_t cx_name_hash(const char*name)
{
	uint64_t h=14695981039346656037ULL;
	for (const unsigned char*p=(const unsigned char*)name; *p; p++) {
		h^=*p;
		h*=1099511628211ULL;
	}
	return h;
}
static inline const cx_use *cx_use_lookup(const cx_graph*g,const char*name)
{
	if (!g||!g->last_use||!name||!*name) {
		return NULL;
	}
	size_t mask=g->last_use_capacity-1;
	size_t i=(size_t)cx_name_hash(name)&mask;
	for (size_t n=0; n<g->last_use_capacity; n++) {
		const cx_use*slot=&g->last_use[i];
		if (!slot->name) {
			return NULL;
		}
		if (!strcmp(slot->name,name)) {
			return slot;
		}
		i=(i+1)&mask;
	}
	return NULL;
}
static inline int cx_graph_prepare_last_use(cx_graph*g)
{
	if (!g) {
		return -1;
	}
	if (g->last_use) {
		return 0;
	}
	size_t n=0;
	for (size_t i=0; i<g->nn; i++) {
		if ((size_t)g->node[i].ni>SIZE_MAX-n) {
			return -1;
		}
		n+=(size_t)g->node[i].ni;
	}
	/* Allocate a power-of-two table with at most 50% occupancy. */
	if (n>SIZE_MAX/2) {
		return -1;
	}
	size_t capacity=16;
	while (capacity<2*n) {
		if (capacity>SIZE_MAX/2 || capacity>SIZE_MAX/sizeof(cx_use)/2) {
			return -1;
		}
		capacity*=2;
	}
	cx_use*map=(cx_use*)calloc(capacity,sizeof(*map));
	if (!map) {
		return -1;
	}
	const size_t mask=capacity-1;
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*node=&g->node[i];
		for (int j=0; j<node->ni; j++) {
			const char *name=node->input[j];
			if (!*name) {
				continue;  /* optional ONNX input */
			}
			size_t h=(size_t)cx_name_hash(name)&mask;
			for (;;) {
				cx_use*slot=&map[h];
				if (!slot->name) {
					slot->name=name;
					slot->last=i;
					break;
				}
				if (!strcmp(slot->name,name)) {
					slot->last=i;
					break;
				}
				h=(h+1)&mask;
			}
		}
	}
	g->last_use=map;
	g->last_use_capacity=capacity;
	return 0;
}
/* Compact interpreter with optional persistent initializer cache.
 * Decoder/joiner call this repeatedly: dropping intermediate tensors after
 * their last use is compatible with keeping immutable initializer weights.
 * An initializer used directly as a graph output is already preserved.
 */
static inline int cx_graph_run_compact_cached(cx_graph*g,cx_env*e,int cache_weights)
{
	if (!g||!e||cx_graph_prepare_last_use(g)) {
		return -1;
	}
	for (size_t i=0; i<g->nn; i++) {
		const cx_node*n=&g->node[i];
		const char*issue=cx_node_issue(g,n);
		if (issue) {
			if (g->profiler) {
				g->profiler->stats[i].failures++;
			}
			fprintf(stderr,"native ONNX: blocked node %zu (%s): %s\n",i,n->op,issue);
			return -1;
		}
		uint64_t profile_start=g->profiler?cx_profile_clock_ns():0;
		int node_failed=cx_execute_node(g,e,n);
		if (g->profiler) {
			cx_profiler_record(g->profiler,i,profile_start,node_failed);
		}
		if (node_failed) {
			fprintf(stderr,"native ONNX: node %zu (%s) failed\n",i,n->op);
			return -1;
		}
		cx_trace_after(g,e,n,i);
		for (int j=0; j<n->ni; j++) {
			const char*name=n->input[j];
			if (!*name||cx_graph_keep(g,name)||
			                (cache_weights && cx_find_weight(g,name))) {
				continue;
			}
			const cx_use*slot=cx_use_lookup(g,name);
			if (slot && slot->last==i) {
				cx_env_drop(e,name);
			}
		}
		for (int j=0; j<n->no; j++) {
			const char*name=n->output[j];
			if (!*name||cx_graph_keep(g,name)||
			                (cache_weights && cx_find_weight(g,name))) {
				continue;
			}
			/* A terminal/unused node output may have no consumer entry. */
			const cx_use*slot=cx_use_lookup(g,name);
			if (!slot||slot->last<=i) {
				cx_env_drop(e,name);
			}
		}
	}
	return 0;
}

static inline int cx_graph_run_compact(cx_graph*g,cx_env*e)
{
	return cx_graph_run_compact_cached(g,e,0);
}

/* ========================================================================
 * Reusable ONNX sessions
 * ======================================================================== */

/* Minimal reusable multi-input ONNX graph sessions for Cat's eye.
 * No external libraries; copies data at graph boundaries for safety.
 * The graph interpreter is still a PARTIAL ONNX implementation.
 * License: MPL-2.0.
 */

typedef struct {
	cx_graph graph;
	cx_env tensors;
	unsigned long long runs;
	int cache_weights;  /* opt-in for repeatedly executed decoder/joiner */
} cx_session;

static inline int cx_session_open(cx_session *s,const char *path)
{
	if (!s) {
		return -1;
	}
	memset(s,0,sizeof(*s));
	return cx_graph_open(&s->graph,path);
}
static inline void cx_session_close(cx_session*s)
{
	if (!s) {
		return;
	}
	cx_env_free(&s->tensors);
	cx_graph_close(&s->graph);
	s->runs=0;
}
static inline int cx_session_input_index(const cx_graph*g,const char*name)
{
	for (int k=0; k<g->ninput; k++)if (cx_is(g->input[k],name)) {
			return k;
		}
	return -1;
}
/* Exact names, element type and fully known dimensions validated when available.
 * Unknown symbolic dimensions are accepted but must remain nonnegative at run time. */
static inline int cx_session_feed(cx_session*s,const char*name,const int64_t*d,
                                  int nd,int type,const float*values,uint64_t count)
{
	if (!s||!name||!values||nd<0||nd>CX_DIMS) {
		return -1;
	}
	int index=cx_session_input_index(&s->graph,name);
	if (index<0) {
		return -1;
	}
	const cx_value_meta*m=&s->graph.input_meta[index];
	if (m->type && m->type!=type) {
		return -1;
	}
	if (m->nd>=0) {
		if (m->nd!=nd) {
			return -1;
		}
		for (int k=0; k<nd; k++)if (m->d[k]>=0 && m->d[k]!=d[k]) {
				return -1;
			}
	}
	uint64_t size=1;
	for (int k=0; k<nd; k++) {
		if (!d||d[k]<0 || (d[k] && size>400000000u/(uint64_t)d[k])) {
			return -1;
		}
		size*=d[k];
	}
	if (size!=count) {
		return -1;
	}
	cx_env_drop(&s->tensors,name);
	cx_tensor*t=cx_env_new(&s->tensors,name,d,nd,type);
	if (!t) {
		return -1;
	}
	memcpy(t->v,values,(size_t)size*sizeof(float));
	return 0;
}
static inline int cx_session_feed_tensor(cx_session*s,const char*name,const cx_tensor*t)
{
	if (!t) {
		return -1;
	}
	return cx_session_feed(s,name,t->d,t->nd,t->type,t->v,t->n);
}
static inline const cx_tensor *cx_session_output(cx_session*s,const char*name)
{
	if (!s||!name) {
		return NULL;
	}
	int ok=0;
	for (int i=0; i<s->graph.noutput; i++)if (cx_is(s->graph.output[i],name)) {
			ok=1;
		}
	return ok?cx_env_find(&s->tensors,name):NULL;
}
/* The output tensors remain valid until the next feed/run/close. */
static inline int cx_session_run(cx_session*s)
{
	if (!s) {
		return -1;
	}
	/* Keep persistent input tensors; evict previous outputs and intermediates.
	 * This also prevents accidental in-place overwriting by the executor. */
	for (size_t i=0; i<s->tensors.n;) {
		const char*name=s->tensors.t[i]->name;
		if (cx_session_input_index(&s->graph,name)>=0 ||
		                (s->cache_weights && cx_find_weight(&s->graph,name))) {
			i++;
		} else {
			cx_env_drop(&s->tensors,name);
		}
	}
	for (int j=0; j<s->graph.ninput; j++) {
		if (cx_find_weight(&s->graph,s->graph.input[j])) {
			continue;  /* older ONNX initializers */
		}
		if (!cx_env_find(&s->tensors,s->graph.input[j])) {
			return -1;
		}
	}
	/* Decoder/joiner may run thousands of times: preserve lazily-loaded
	   weights across calls. Compacted encoder stays low-memory. */
	/* Reuse initializer weights WITHOUT retaining intermediate activations. */
	if (cx_graph_run_compact_cached(&s->graph,&s->tensors,s->cache_weights)) {
		return -1;
	}
	for (int j=0; j<s->graph.noutput; j++)
		if (!cx_env_find(&s->tensors,s->graph.output[j])) {
			return -1;
		}
	s->runs++;
	return 0;
}
/* In transducer models: pass decoder output to joiner, or roll a state
 * output into the matching state input between steps. The copy prevents
 * aliasing bugs when old tensors are evicted. */
static inline int cx_session_feed_from(cx_session*dst,const char*input,
                                       cx_session*src,const char*output)
{
	const cx_tensor*t=cx_session_output(src,output);
	if (!t) {
		return -1;
	}
	/* If source and destination are the same session, free() of previous
	   input cannot invalidate t unless the output name equals the input. */
	if (src==dst && cx_is(input,output)) {
		return 0;
	}
	return cx_session_feed_tensor(dst,input,t);
}


#endif /* CATS_ONNX_H */

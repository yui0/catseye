//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#define _debug(...)	{ printf("%s(%d):", __func__, __LINE__); printf(__VA_ARGS__); }

#if defined(CATS_OPENGL) || defined(CATS_OPENCL)
#ifndef CATS_USE_FLOAT
#define CATS_USE_FLOAT
#endif
#endif

#ifdef CATS_USE_FIXED
#define real		short
#elif defined CATS_USE_FLOAT
#define real		float
#define sqrt		sqrtf
#define pow		powf
#define exp		expf
#define log		logf
#define fabs		fabsf
#define sin		sinf
#define cos		cosf
#define tan		tanf
#define tanh		tanhf
#else
#define real		double
#warning "using double!!"
#endif

#if defined(CATS_OPENGL)
#define GL_DEBUG
#include "sgemm_gl1.h"
#define sgemm_init(s)				sgemm_gl_init(s, s, s)
#define sgemm_finish()				sgemm_gl_finish()
#define gemm_rnn(m, n, k, alpha, a, b, beta, c)	sgemm_gl(GEMM1_RNN, m, n, k, alpha, a, b, beta, c)
#define gemm_rnt(m, n, k, alpha, a, b, beta, c)	sgemm_gl(GEMM1_RNT, m, n, k, alpha, a, b, beta, c)
#define gemm_rtn(m, n, k, alpha, a, b, beta, c)	sgemm_gl(GEMM1_RTN, m, n, k, alpha, a, b, beta, c)
#elif defined(CATS_OPENCL)
#include "sgemm_ocl2.h"
#define sgemm_init(s)				sgemm_ocl_init(0, 0, (s))
#define sgemm_finish()				sgemm_ocl_finish()
#define gemm_rnn(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('N', 'N', m, n, k, alpha, a, b, beta, c)
#define gemm_rnt(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('N', 'T', m, n, k, alpha, a, b, beta, c)
#define gemm_rtn(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('T', 'N', m, n, k, alpha, a, b, beta, c)
#else
//#include "gemm_cpu.h"
inline void gemm_rnn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
	#pragma omp parallel for
	for (int m=0; m<M; ++m) { // fast
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m*K+k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}
inline void gemm_rnt(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
	#pragma omp parallel for
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			register real sum = 0;
			for (int k=0; k<K; ++k) {
				sum += A[m*K+k] * B[k+K*n];
			}
			C[m*N+n] += alpha * sum;
		}
	}
}
inline void gemm_rtn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
	#pragma omp parallel for
	for (int m=0; m<M; ++m) {
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m+M*k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}
#define sgemm_init(s)
#define sgemm_finish()
#endif

// http://xorshift.di.unimi.it/xorshift128plus.c
// https://github.com/AndreasMadsen/xorshift/blob/master/reference.c
// http://ogawa-sankinkoutai.seesaa.net/category/5784373-1.html
#define XOR128_MAX	18446744073709551615.0
#if __WORDSIZE == 64
typedef unsigned long int	uint64_t;
#else
__extension__
typedef unsigned long long int	uint64_t;
#endif
// The state must be seeded so that it is not everywhere zero.
uint64_t seed[2];
void xor128_init(unsigned int s)
{
	for (int i=1; i<=2; i++) {
		seed[i-1] = s = 1812433253U * ( s ^ ( s >> 30 ) ) + i;
	}
}
static inline uint64_t xor128()
{
	uint64_t s1 = seed[0];
	const uint64_t s0 = seed[1];
	seed[0] = s0;
	s1 ^= s1 << 23;
	return ( seed[1] = ( s1 ^ s0 ^ ( s1 >> 17 ) ^ ( s0 >> 26 ) ) ) + s0;
}
#define frand()			( xor128() / (XOR128_MAX +1.0) )
#define _rand(max)		(int)( xor128() / (XOR128_MAX +1.0) * max)
#define random(min, max)	( xor128() / (XOR128_MAX +1.0) * (max -min) +min )
// https://omitakahiro.github.io/random/random_variables_generation.html
static inline real rand_normal(real mu, real sigma)
{
	real z = sqrt(-2.0*log(frand())) * sin(2.0*M_PI*frand());
	return mu + sigma*z;
}
static inline int binomial(/*int n, */real p)
{
//	if (p<0 || p>1) return 0;
	int c = 0;
//	for (int i=0; i<n; i++) {
		real r = frand();
		if (r < p) c++;
//	}
	return c;
}

struct __CatsEye;
typedef struct __CatsEye_layer {
	int inputs;		// input size
	int type;		// layer type
	real eta;		// learning rate

	int ksize;		// CNN
	int stride;		// CNN
	int padding;		// CNN
	int px, py, pz;		// CNN (AUTO: padding)
	int ich, ch;		// CNN
	int sx, sy, ox, oy;	// CNN (AUTO: i/o size)

	int r;			// Pixel Shuffler

	int fix;		// training / no training

	int hiddens;		// RNN
	int truncatedTime;	// RNN

	// auto config
	int outputs;		// output size
	real *x;		// input
	real *z;		// output
	real *bias;		// bias
	real *w, *dw, *g;	// weight
	real *dOut, *dIn;	// gradient
	real *workspace;	// for im2col, col2im
/*	real *Wi, *dOuti;	// RNN [hidden * time](input -> hidden) = W, dOut
	real *Wr, *dOutr;	// RNN [hidden * hidden]
	real *Wo, *dOuto;	// RNN [output * hidden](hidden -> output)
//	real *U, *dU;		// RNN [hidden * time](input -> hidden)
//	real *V, *dV;		// RNN [output * hidden](hidden -> output)
	real *eh;		// RNN [time * hidden]
	real *s;		// RNN [time * hidden]
	real *u;		// RNN [time * hidden]
//	real *y;		// RNN [time * output]
	real *v;		// RNN [time * output]*/
//	real (*act2)(real x);	// RNN
//	real (*dact2)(real x);	// RNN

	real momentum;		// SGD with momentum
	real mu, var;		// Batch Normalization [average, variance]
	real gamma, beta;	// Batch Normalization
	real alpha, min, max;	// Leaky ReLU, RReLU

	void (*forward)(struct __CatsEye_layer*);
	void (*backward)(struct __CatsEye_layer*);
	void (*update)(struct __CatsEye_layer*);
	struct __CatsEye *p;	// point to struct CatsEye for loss function

	char *name;		// layer name
} CatsEye_layer;

typedef struct __CatsEye {
	// number of each layer
	int layers, *u;
	CatsEye_layer *layer;
	int start, stop, end;

	// train parameter
	int epoch;
	int batch;
	int slide;
	// label
	int16_t *clasify;
	real *label;

	// output layers [o = f(z)]
	real **z, **o, *odata;
	int osize;
	// error value
	real **d, *ddata;
	int dsize;
	// weights
	real **w, *wdata;
	int *ws, wsize;
	// working memory
	real *mem;
} CatsEye;

// https://qiita.com/omiita/items/1735c1d048fe5f611f80
#ifdef CATS_USE_MOMENTUM_SGD
 #define CatsEye_solver	CatsEye_solver_MomentumSGD
#elif defined CATS_USE_ADAGRAD
 #define CatsEye_solver	CatsEye_solver_adagrad
#elif defined CATS_USE_RMSPROP
 #define RMSPROP_RHO	0.9
 #define SOLVER(gemm, m, n, k, alpha, a, b, beta, c)\
{\
	gemm(m, n, k, 1, a, b, 0, l->dw);\
	for (int i=0; i<m*n; i++) {\
		l->g[i] = /*l->mu*/RMSPROP_RHO * l->g[i] + (1 - /*l->mu*/RMSPROP_RHO) * l->dw[i] * l->dw[i];\
		c[i] -= l->eta * l->dw[i] / (sqrt(l->g[i] +1e-12));\
	}\
}
#else
 #define SOLVER(gemm, m, n, k, alpha, a, b, beta, c) gemm(m, n, k, alpha, a, b, beta, c)
#endif
static inline void CatsEye_solver_MomentumSGD(CatsEye_layer *l, char mj, char ta, char tb, int m, int n, int k, real *a, int lda, real *b, int ldb, int ldc)
{
	// MomentumSGD [ dw = u * dw - n * g ]
//	gemm(mj, ta, tb, m, n, k, -l->eta * (1 - l->momentum), a, lda, b, ldb, l->momentum, l->dw, ldc);
	gemm_rtn(l->outputs, l->inputs, l->p->batch, -l->eta * (1 - l->momentum), l->dOut, l->x, l->momentum, l->dw);
	for (int i=0; i<m*n; i++) {
		l->w[i] += l->dw[i];
	}
}
static inline void CatsEye_solver_adagrad(CatsEye_layer *l, char mj, char ta, char tb, int m, int n, int k, real *a, int lda, real *b, int ldb, int ldc)
{
	// adagrad [ g2[i] += g * g; w[i] -= eta * g / sqrt(g2[i]); ]
//	gemm(mj, ta, tb, m, n, k, 1, a, lda, b, ldb, 0, l->dw, ldc);
	gemm_rtn(l->outputs, l->inputs, l->p->batch, 1, l->dOut, l->x, 0, l->dw);
	for (int i=0; i<m*n; i++) {
		l->g[i] += l->dw[i] * l->dw[i];
		l->w[i] -= l->eta * l->dw[i] / (sqrt(l->g[i] +1e-8));
	}
}

// Fully connected [ z^l = ( w^l * a^l-1 + b^l ) ]
static void CatsEye_linear_forward(CatsEye_layer *l)
{
#ifdef CATS_TEST
	real *o = l->z;
	real *w = l->w;
	for (int i=l->outputs; i>0; i--) {
		real *x = l->x;
		register real a = 0;
		for (int n=0; n<l->inputs; n++) {
			a += (*x++) * (*w++);
		}
//		*o++ = a;
		*o++ = a + *w++;	// bias!!
	}
#else
	// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
	// output(m,n) := input(m=1,k) * weightsT(k,n)
	gemm_rnt(l->p->batch, l->outputs, l->inputs, 1, l->x, l->w, 0, l->z);
//	for (int i=0; i<l->outputs; i++) l->z[i] += l->w[l->inputs*l->outputs +i]; // bias!!
	for (int n=0; n<l->p->batch; n++) {
//		gemm_rnt(1, l->outputs, l->inputs, 1, l->x+l->inputs*n, l->w, 0, l->z+l->outputs*n);
		for (int i=0; i<l->outputs; i++) l->z[n*l->outputs +i] += l->w[l->inputs*l->outputs +i]; // bias!!
	}
//	for (int i=l->inputs-10; i<l->inputs; i++) printf("%f ", l->x[i]);
//	printf("\n");

	// https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
	// output := weightsT * input
//	gemm_rtn(l->outputs, l->p->batch, l->inputs, 1, l->w, l->x, 0, l->z);
#endif
}
static void CatsEye_linear_backward(CatsEye_layer *l)
{
#ifdef CATS_TEST
	real *d = l->dIn;
	real *w = l->w;
//	for (int i=0; i<l->inputs; i++) {
	for (int i=0; i<=l->inputs; i++) {	// bias!!
		real *dw = l->dOut;
		real *ww = w++;
		register real a = 0;
		for (int n=0; n<l->outputs; n++) {
			a += (*dw++) * (*ww);
//			ww += l->inputs;
			ww += l->inputs+1;	// bias!!
		}
		*d++ = a;
	}
#else
	// dIn := dOut * W
	gemm_rnn(l->p->batch, l->inputs, l->outputs, 1, l->dOut, l->w, 0, l->dIn);
//	for (int i=0; i<l->outputs; i++) l->dIn[l->inputs] += l->dOut[i] * l->w[l->inputs*l->outputs +i]; // bias!!
/*	for (int n=0; n<l->p->batch; n++) {
//		gemm_rnn(1, l->inputs, l->outputs, 1, l->dOut+l->outputs*n, l->w, 0, l->dIn+l->inputs*n);
		for (int i=0; i<l->outputs; i++) l->dIn[n*l->inputs +l->inputs] += l->dOut[n*l->outputs +i] * l->w[l->inputs*l->outputs +i]; // bias!!
	}*/

/*	printf("gemm_rnn: ");
	for (int i=l->inputs-10; i<l->inputs; i++) printf("%f ", l->dIn[i]);
	printf("\n");
	printf("GEMM1_RNN: ");
	sgemm_gl(GEMM1_RNN, l->p->batch, l->inputs, l->outputs, 1, l->dOut, l->w, 0, l->dIn);
	for (int i=l->inputs-10; i<l->inputs; i++) printf("%f ", l->dIn[i]);
	printf("\n");*/

	// gradients(input) := weights * gradients(output)
//	gemm_rnn(l->inputs, l->p->batch, l->outputs, 1, l->w, l->dOut, 0, l->dIn);
#endif
}
static void CatsEye_linear_update(CatsEye_layer *l)
{
#ifdef CATS_TEST
	real *w = l->w;
	real *d = l->dOut;
	for (int i=l->outputs; i>0; i--) {
		real *x = l->x;
		register real a = -l->eta * (*d++);
		for (int n=0; n<l->inputs; n++) {
			*w++ += (*x++) * a;
		}
		*w++ += a;	// bias!!
	}
#else
	// W := W - eta * dOutT * x
	//gemm_rtn(l->outputs, l->inputs, l->p->batch, -l->eta, l->dOut, l->x, 1, l->w);
//	for (int i=0; i<l->outputs; i++) l->w[l->inputs*l->outputs +i] += -l->eta * l->dOut[i]; // bias!!
	SOLVER(gemm_rtn, l->outputs, l->inputs, l->p->batch, -l->eta, l->dOut, l->x, 1, l->w);
	for (int n=0; n<l->p->batch; n++) {
//		gemm_rtn(l->outputs, l->inputs, 1, -l->eta, l->dOut+l->outputs*n, l->x+l->inputs*n, 1, l->w);
		for (int i=0; i<l->outputs; i++) l->w[l->inputs*l->outputs +i] -= l->eta * l->dOut[n*l->outputs +i]; // bias!!
	}
//	for (int i=0; i<10; i++) printf("%f ", l->dOut[i]);
/*	for (int i=0; i<10; i++) printf("%f ", l->w[i]);
	printf("\n");*/

	// weights := weights - eta * input * gradientsT(output)
//	gemm_rnt(l->inputs, l->outputs, l->p->batch, -l->eta, l->x, l->dOut, 1, l->w);
#endif
}

#if 0
static void CatsEye_bias_forward(CatsEye_layer *l)
{
	for (int n=0; n<l->p->batch; n++) {
//		gemm_rnt(1, l->outputs, l->inputs, 1, l->x+l->inputs*n, l->w, 0, l->z+l->outputs*n);
		for (int i=0; i<l->outputs; i++) l->z[n*l->outputs +i] += l->w[i];
	}
}
/*static void CatsEye_bias_backward(CatsEye_layer *l)
{
	for (int n=0; n<l->p->batch; n++) {
//		gemm_rnn(1, l->inputs, l->outputs, 1, l->dOut+l->outputs*n, l->w, 0, l->dIn+l->inputs*n);
		for (int i=0; i<l->outputs; i++) l->dIn[n*l->inputs] += l->dOut[n*l->outputs +i] * l->w[i];
	}
}*/
static void CatsEye_bias_update(CatsEye_layer *l)
{
	for (int n=0; n<l->p->batch; n++) {
//		gemm_rtn(l->outputs, l->inputs, 1, -l->eta, l->dOut+l->outputs*n, l->x+l->inputs*n, 1, l->w);
		for (int i=0; i<l->outputs; i++) l->w[i] -= l->eta * l->dOut[n*l->outputs +i];
	}
}
#endif

// convolution [https://github.com/hiroyam/dnn-im2col, https://github.com/pjreddie/darknet]
static inline void im2col(const real *im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, real *col)
{
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;

	for (int c=0; c<channels_col; c++) {
		int w_offset = c % kernel_w;
		int h_offset = (c / kernel_w) % kernel_h;
		int c_im = c / kernel_h / kernel_w;
		for (int h=0; h<height_col; h++) {
			for (int w=0; w<width_col; w++) {
				int h_pad = h * stride_h - pad_h + h_offset;
				int w_pad = w * stride_w - pad_w + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					col[(c * height_col + h) * width_col + w] =
						im[(c_im * height + h_pad) * width + w_pad];
				else
					col[(c * height_col + h) * width_col + w] = 0;
			}
		}
	}
}
static inline void col2im(const real *col, const int channels,
	const int height, const int width, const int patch_h, const int patch_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, real *im)
{
	memset(im, 0, sizeof(real)*height * width * channels);
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int channels_col = channels * patch_h * patch_w;

	for (int c = 0; c < channels_col; ++c) {
		int w_offset = c % patch_w;
		int h_offset = (c / patch_w) % patch_h;
		int c_im = c / patch_h / patch_w;
		for (int h = 0; h < height_col; ++h) {
			for (int w = 0; w < width_col; ++w) {
				int h_pad = h * stride_h - pad_h + h_offset;
				int w_pad = w * stride_w - pad_w + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					im[(c_im * height + h_pad) * width + w_pad] +=
						col[(c * height_col + h) * width_col + w];
			}
		}
	}
}
static void CatsEye_convolutional_forward(CatsEye_layer *l)
{
#if 1
	for (int i=0; i<l->p->batch; i++) {
		real *workspace;
		if (l->ksize==1) {
			workspace = l->x +l->inputs*i;
		} else {
			workspace = l->workspace +l->ox*l->oy*l->ksize*l->ksize*l->ich *i;
			im2col(l->x +l->inputs*i, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace);
		}
		// z = W * x [A(m,k) B(k,n) C(m,n)], cnhw
//		gemm('R', 'N', 'N', l->ch, l->ox*l->oy*1, l->ksize*l->ksize*l->ich, 1, l->w, l->ksize*l->ksize*l->ich, workspace, l->ox*l->oy, 0, l->z +l->outputs*i, l->ox*l->oy);
		gemm_rnn(l->ch, l->ox*l->oy*1, l->ksize*l->ksize*l->ich, 1, l->w, workspace, 0, l->z +l->outputs*i);

		// https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
		// output := workspace * weights
//		gemm_rnn(l->ox*l->oy*1, l->ch, l->ksize*l->ksize*l->ich, 1, workspace, l->w, 0, l->z +l->outputs*i);
	}
#else
	real *workspace;
	if (l->ksize==1) {
		workspace = l->x;
	} else {
		workspace = l->workspace;
		for (int i=0; i<l->p->batch; i++) {
			im2col(l->x +l->inputs*i, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace +l->ksize*l->ksize*l->ich *l->ox*l->oy*i);
		}
	}
	gemm_rnn(l->ch, l->ox*l->oy*l->p->batch, l->ksize*l->ksize*l->ich, 1, l->w, workspace, 0, l->z);
#endif
}
static void CatsEye_convolutional_backward(CatsEye_layer *l)
{
#if 1
	for (int i=0; i<l->p->batch; i++) {
//		real *workspace = l->ksize!=1 ? col : l->dIn +l->inputs*i;
		real *workspace = l->ksize!=1 ? l->p->mem : l->dIn +l->inputs*i;
		// dIn = W**T * dOut [A(m,k) B(k,n) C(m,n)]
//		gemm('R', 'T', 'N', l->ksize*l->ksize*l->ich, l->ox*l->oy*1, l->ch, 1, l->w, l->ksize*l->ksize*l->ich, l->dOut +l->outputs*i, l->ox*l->oy, 0, workspace, l->ox*l->oy);
		gemm_rtn(l->ksize*l->ksize*l->ich, l->ox*l->oy*1, l->ch, 1, l->w, l->dOut +l->outputs*i, 0, workspace);
/*		printf("gemm_rtn: ");
		for (int i=0; i<10; i++) printf("%f ", workspace[i]);
		printf("\n");*/
		if (l->ksize!=1) {
			col2im(workspace, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->dIn +l->inputs*i);
		}
	}
#else
	real *workspace = l->ksize!=1 ? l->p->mem : l->dIn;
	//gemm_rtn(l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, l->ch, 1, l->w, l->dOut, 0, workspace);
	SOLVER(gemm_rtn, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, l->ch, 1, l->w, l->dOut, 0, workspace);
	if (l->ksize!=1) {
		for (int i=0; i<l->p->batch; i++) {
			col2im(workspace +l->ksize*l->ksize*l->ich *l->ox*l->oy*i, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->dIn +l->inputs*i);
		}
	}
#endif
}
static void CatsEye_convolutional_update(CatsEye_layer *l)
{
#if 1
	for (int i=0; i<l->p->batch; i++) {
		real *workspace = l->ksize!=1 ? l->workspace +l->ox*l->oy*l->ksize*l->ksize*l->ich *i : l->x +l->inputs*i;
		// W = W - eta * dOut * x**T [A(m,k) B(k,n) C(m,n)]
//		gemm('R', 'N', 'T', l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, -l->eta/l->p->batch, l->dOut, l->ox*l->oy, workspace, l->ox*l->oy, 1, l->w, l->ksize*l->ksize*l->ich);
//		CatsEye_solver(l, 'R', 'N', 'T', l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy, l->dOut +l->outputs*i, l->ox*l->oy, workspace, l->ox*l->oy, l->ksize*l->ksize*l->ich);
		gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*1, -l->eta, l->dOut, workspace, 1, l->w);
	}
#else
	real *workspace = l->ksize!=1 ? l->workspace : l->x;
	gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, -l->eta, l->dOut, workspace, 1, l->w);
#endif
}

static void CatsEye_deconvolutional_forward(CatsEye_layer *l)
{
	gemm_rtn(l->ksize*l->ksize*l->ich, l->sx*l->sy*l->p->batch, l->ch, 1, l->w, l->x, 0, l->workspace);
	for (int i=0; i<l->p->batch; i++) {
		col2im(l->workspace +l->ksize*l->ksize*l->ich*l->sx*l->sy*i, l->ch, l->oy, l->ox, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->z +l->outputs*i);
	}
}
static void CatsEye_deconvolutional_backward(CatsEye_layer *l)
{
	real *workspace = l->ksize!=1 ? l->workspace : l->dOut;
	for (int i=0; i<l->p->batch; i++) {
		im2col(l->dOut +l->outputs*i, l->ch, l->oy, l->ox, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace +l->ksize*l->ksize*l->ich *l->ox*l->oy*i);
	}
	gemm_rnn(l->ch, l->ox*l->oy*l->p->batch, l->ksize*l->ksize*l->ich, 1, l->w, workspace, 0, l->dIn);
}
static void CatsEye_deconvolutional_update(CatsEye_layer *l)
{
	real *workspace = l->ksize!=1 ? l->workspace : l->x;
	gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, -l->eta, l->dOut, workspace, 1, l->w);
}

// calculate forward propagation
static void CatsEye_maxpooling_forward(CatsEye_layer *l)
{
	int step = l->sx -l->ksize;
	for (int i=0; i<l->p->batch; i++) {
		int *max = (int*)l->workspace +l->outputs*i; // temp
		real *o = l->z +l->outputs*i;

		for (int c=0; c<l->ch; c++) { // in/out
			for (int y=0; y<l->oy; y++) {
				int ix = c*l->sx*l->sy + y*l->stride*l->sx +l->inputs*i;
				for (int x=0; x<l->ox; x++) {
					int n = ix+x*l->stride;
					real a = l->x[n];
					*max = n;
					for (int wy=l->ksize; wy>0; wy--) {
						for (int wx=l->ksize; wx>0; wx--) {
							if (a<l->x[n]) {
								a = l->x[n];
								*max = n;
							}
							n++;
						}
						n += step;
					}
					max++;
					*o++ = a;
				}
			}
		}
	}
}
// calculate back propagation
static void CatsEye_maxpooling_backward(CatsEye_layer *l)
{
	int *max = (int*)l->workspace; // temp
	real *delta = l->dOut;
	real *d = l->dIn;
	memset(d, 0, sizeof(real)*l->inputs *l->p->batch);
	for (int i=0; i<l->outputs *l->p->batch; i++) {
		d[*max++] += *delta++;
	}
}

static void CatsEye_avgpooling_forward(CatsEye_layer *l)
{
	int step = l->sx -l->ksize;
	real n = l->ksize * l->ksize;
	for (int i=0; i<l->p->batch; i++) {
		real *o = l->z +l->outputs*i;

		for (int c=0; c<l->ch; c++) { // in/out
			for (int y=0; y<l->oy; y++) {
				int ix = c*l->sx*l->sy + y*l->stride*l->sx +l->inputs*i;
				for (int x=0; x<l->ox; x++) {
					real *u = l->x + ix+x*l->stride;
					real a = 0;
					for (int wy=l->ksize; wy>0; wy--) {
						for (int wx=l->ksize; wx>0; wx--) {
							a += *u++;
						}
						u += step;
					}
					*o++ = a / n;
				}
			}
		}
	}
}
static void CatsEye_avgpooling_backward(CatsEye_layer *l)
{
	int step = l->sx -l->ksize;
	real n = l->ksize * l->ksize;
	memset(l->dIn, 0, sizeof(real)*l->inputs *l->p->batch);
	for (int i=0; i<l->p->batch; i++) {
		real *delta = l->dOut +l->outputs*i;

		for (int c=0; c<l->ch; c++) { // in/out
			for (int y=0; y<l->oy; y++) {
				int ix = c*l->sx*l->sy + y*l->stride*l->sx +l->inputs*i;
				for (int x=0; x<l->ox; x++) {
					real *d = l->dIn + ix+x*l->stride;
					real a = *delta++ / n;
					for (int wy=l->ksize; wy>0; wy--) {
						for (int wx=l->ksize; wx>0; wx--) {
							*d++ += a;
						}
						d += step;
					}
				}
			}
		}
	}
}
static void CatsEye_global_avgpooling_forward(CatsEye_layer *l)
{
	real *o = l->z;
	real *u = l->x;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ich; c++) { // in/out
			real a = 0;
			for (int n=l->sx*l->sy; n>0; n--) {
				a += *u++;
			}
			*o++ = a / (l->sx*l->sy);
		}
	}
}
static void CatsEye_global_avgpooling_backward(CatsEye_layer *l)
{
//	memset(l->dIn, 0, sizeof(real)*l->inputs *l->p->batch);

	real *delta = l->dOut;
	real *d = l->dIn;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ich; c++) { // in/out
			real a = (*delta++) / (l->sx*l->sy);
			for (int n=l->sx*l->sy; n>0; n--) {
				*d++ = a;
			}
		}
	}
}

// Sub-Pixel Convolution
static void CatsEye_PixelShuffler_forward(CatsEye_layer *l)
{
	int ch = l->ich / l->ch;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // out
			real *xx = l->x + c*ch*l->sx*l->sy +l->inputs*i;
			real *o = l->z + c*l->ox*l->oy +l->outputs*i;

			for (int cc=0; cc<ch; cc++) { // in
				real *x = xx + cc*l->sx*l->sy;
				int px = cc%l->r;
				int py = cc/l->r;
				for (int n=0; n<l->sy; n++) {
					for (int m=0; m<l->sx; m++) {
						o[m*l->r+px +(n*l->r+py)*l->ox] = *x++;
					}
				}
			}
		}
	}
}
static void CatsEye_PixelShuffler_backward(CatsEye_layer *l)
{
	int ch = l->ich / l->ch;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // out
			real *d = l->dIn + c*ch*l->sx*l->sy +l->inputs*i;
			real *delta = l->dOut + c*l->ox*l->oy +l->outputs*i;

			for (int cc=0; cc<ch; cc++) { // in
				real *x = d + cc*l->sx*l->sy;
				int px = cc%l->r;
				int py = cc/l->r;
				for (int n=0; n<l->sy; n++) {
					for (int m=0; m<l->sx; m++) {
						*x++ = delta[m*l->r+px +(n*l->r+py)*l->ox];
					}
				}
			}
		}
	}
}

// FIXME: obsolete
static void CatsEye_padding_forward(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // in/out
			real *x = l->x +c*l->sx*l->sy +l->inputs*i;
			real *o = l->z +c*l->ox*l->oy +l->ox*l->padding +l->padding +l->outputs*i;
			for (int n=0; n<l->sy; n++) {
				memcpy(o, x, sizeof(real)*l->sx);
				x += l->sx;
				o += l->ox;
			}
		}
	}
}
static void CatsEye_padding_backward(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // in/out
			real *d = l->dIn +c*l->sx*l->sy +l->inputs*i;
			real *delta = l->dOut +c*l->ox*l->oy +l->ox*l->padding +l->padding +l->outputs*i;
			for (int n=0; n<l->sy; n++) {
				memcpy(d, delta, sizeof(real)*l->sx);
				d += l->sx;
				delta += l->ox;
			}
		}
	}
}

// https://deepnotes.io/batchnorm
static void CatsEye_BatchNormalization_forward(CatsEye_layer *l)
{
	int len = l->inputs *l->p->batch;

	// average
	real *x = l->x;
	real avg = 0;
	for (int i=0; i<len; i++) avg += *x++;
	avg /= len;

	// variance
	x = l->x;
	real var = 0;
	real *z = l->z;
	for (int i=0; i<len; i++) {
		*z = *x++ -avg;
		var += (*z) * (*z);
		z++;
	}
	var /= len;
	real sigma = sqrt(var + 1e-8);

	z = l->z;
	for (int i=0; i<len; i++) {
		*z = (*z / sigma) * l->gamma + l->beta;
		z++;
	}

	l->mu = avg;
	l->var = var;
}
static void CatsEye_BatchNormalization_backward(CatsEye_layer *l)
{
	int len = l->inputs *l->p->batch;
	real *x = l->x;
	real *delta = l->dOut;
	real *d = l->dIn;
	real sigma = sqrt(l->var + 1e-8);
//	real dbeta = 0;
	real dvar = 0;
	real dmu = 0;
	real dmu2 = 0;
	for (int i=0; i<len; i++) {
//		dbeta += delta[i];

		real X_mu = *x++ - l->mu;
		real dX_norm = *delta++ * l->gamma;
		*d = dX_norm / sigma;
		dvar += dX_norm * X_mu;
		dmu += - *d++; //dX_norm / -sigma;
		dmu2 += -2.0 * X_mu;
	}
	dvar *= -0.5 * pow(l->var + 1e-8, -3.0/2.0);
	dmu += dvar / len * dmu2;

	x = l->x;
	d = l->dIn;
	real a = dmu / len;
	real b = dvar * 2.0 / len;
	for (int i=0; i<len; i++) {
		*d++ += a + b * (*x++ - l->mu);
	}
}

#define CATS_ACT_ARRAY(type)	\
static void CatsEye_act_##type(CatsEye_layer *l)\
{\
	real *x = l->x;\
	real *z = l->z;\
	for (int i=l->outputs *l->p->batch; i>0; i--) {\
		*z++ = CATS_ACT_##type(*x, l);\
		x++;\
	}\
}
#define CATS_DACT_ARRAY(type)	\
static void CatsEye_dact_##type(CatsEye_layer *l)\
{\
	real *z = l->z;\
	real *d = l->dIn;\
	real *dOut = l->dOut;\
	for (int i=0; i<l->inputs *l->p->batch; i++) {\
		*d++ = (*dOut++) * CATS_DACT_##type(*z, l);\
		z++;\
	}\
}

#define sigmoid_gain			1//0.1
#define CATS_ACT_sigmoid(x, l)		(1.0 / (1.0 + exp(-(x) * sigmoid_gain)))
#define CATS_DACT_sigmoid(y, l)		((1.0-(y))*(y) * sigmoid_gain)
CATS_ACT_ARRAY(sigmoid);
CATS_DACT_ARRAY(sigmoid);

// softmax with loss function (output only)
static void CatsEye_act_softmax(CatsEye_layer *l)
{
	real *x = l->x;
	real *z = l->z;
	for (int i=0; i<l->p->batch; i++) {
		real alpha = x[0];
//		for (int i=1; i<l->inputs; i++) alpha = alpha>x[i] ? alpha : x[i]; // FIXME
		for (int i=1; i<l->inputs; i++) alpha = alpha<x[i] ? alpha : x[i];
		real denom = 0.0;
		for (int i=0; i<l->inputs; i++) denom += exp(x[i] - alpha);

		for (int i=l->inputs/*out*/; i>0; i--) {
			real numer = exp(*x++ - alpha);
			*z++ = (numer / denom);
		}

		x += l->inputs;
	}
}
#define CATS_DACT_softmax(y, l)		((y) * (1.0 - (y)))
CATS_DACT_ARRAY(softmax);

// tanh function
// https://github.com/nyanp/tiny-cnn/blob/master/tiny_cnn/activations/activation_function.h
//#ifdef CATS_NORMAL_TANH
#if 1
#define CATS_ACT_tanh(x, l)		(tanh(x))
CATS_ACT_ARRAY(tanh);
#else
static void CatsEye_act_tanh(CatsEye_layer *l)
{
	real *x = l->x;
	real *z = l->z;
	for (int i=l->outputs *l->p->batch; i>0; i--) {
/*		real ep = exp(x);
		real em = exp(-x);
		return (ep-em) / (ep+em);*/
		// fast approximation of tanh (improve 2-3% speed in LeNet-5)
		real x1 = *x;
		real x2 = x1 * x1;
		x1 *= 1.0 + x2 * (0.1653 + x2 * 0.0097);

		*z++ = x1 / sqrt(1.0 + x2);
		x++;
	}
}
#endif
#define CATS_DACT_tanh(y, l)		(1.0-(y)*(y))	// (1.0-tanh(x)*tanh(x))
CATS_DACT_ARRAY(tanh);

// rectified linear unit function
#define CATS_ACT_ReLU(x, l)		((x)>0 ? (x) : 0.0)
#define CATS_DACT_ReLU(y, l)		((y)>0 ? 1.0 : 0.0)
CATS_ACT_ARRAY(ReLU);
CATS_DACT_ARRAY(ReLU);
// leaky rectified linear unit function
//#define leaky_alpha	0.01		// 0 - 1
#define CATS_ACT_LeakyReLU(x, l)	((x)>0 ? (x) : (x)*l->alpha)
#define CATS_DACT_LeakyReLU(y, l)	((y)>0 ? 1.0 : l->alpha)
CATS_ACT_ARRAY(LeakyReLU);
CATS_DACT_ARRAY(LeakyReLU);
// exponential rectified linear unit function
// http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/elu.html
#define CATS_ACT_ELU(x, l)		((x)>0 ? (x) : exp(x)-1.0)
#define CATS_DACT_ELU(y, l)		((y)>0 ? 1.0 : 1.0+y)
CATS_ACT_ARRAY(ELU);
CATS_DACT_ARRAY(ELU);
// Randomized ReLU
// https://qiita.com/Nezura/items/f52fdc483e5e7eceb6b9
#define CATS_ACT_RReLU(x, l)		((x)>0 ? (x) : (x)*l->alpha)
#define CATS_DACT_RReLU(y, l)		((y)>0 ? 1.0 : l->alpha)
static void CatsEye_act_RReLU(CatsEye_layer *l)
{
	l->alpha = random(l->min, l->max);
	real *x = l->x;
	real *z = l->z;
	for (int i=l->outputs *l->p->batch; i>0; i--) {
		*z++ = CATS_ACT_RReLU(*x, l);
		x++;
	}
}
CATS_DACT_ARRAY(RReLU);

// calculate the error of output layer
static void CatsEye_loss_delivative_0_1(CatsEye_layer *l)
{
	// y - t [ t <- one hot ]
	memcpy(l->dIn, l->x, sizeof(real)*l->inputs *l->p->batch);
//	for (int n=0; n<l->inputs; n++) printf(" %f", l->dIn[n]);
//	printf("\n");
//	l->dIn[a] -= 1;
	int16_t *a = l->p->clasify;
	for (int i=0; i<l->p->batch; i++) {
		l->dIn[l->inputs*i + *a++] -= 1;
		/*printf("batch %d:", i);
		for (int n=0; n<l->inputs; n++) printf(" %f", l->dIn[l->inputs*i + n]);
		printf("\n");*/
	}
}
// loss function for mse with identity and cross entropy with sigmoid
static void CatsEye_loss_mse(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *y = l->x;
//	real *o = l->z;
	real mse = 0;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
//		*o++ = (*y - *t) * (*y - *t);
		mse += (*y - *t) * (*y - *t);
		y++;
		t++;
	}
	mse /= l->inputs *l->p->batch;
	l->z[0] = mse;
}
static void CatsEye_loss_delivative_mse(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
		// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
//		printf("(%f-%f=%f) ", *y, *t, *y-*t);
		*d++ = *y++ - *t++;	// (y - t) * (y - t) / 2  ->  y - t
	}
}
// cross-entropy loss function for (multiple independent) binary classifications
// https://qiita.com/kenta1984/items/59a9ef1788e6934fd962
static void CatsEye_loss_delivative_cross_entropy(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		*d++ = (*y - *t++) / (*y * (1 - *y));	// -t * log(y) - (1 - t) * log(1 - y);
		y++;
	}
}
// cross-entropy loss function for multi-class classification
static void CatsEye_loss_delivative_cross_entropy_multiclass(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		// https://www.yukisako.xyz/entry/napier-number
		*d++ = -(*t++) / *y++;	// -t * log(y);
	}
}

static void CatsEye_none(CatsEye_layer *l)
{
}
static void (*_CatsEye_layer_forward[])(CatsEye_layer *l) = {
	CatsEye_linear_forward,
	CatsEye_convolutional_forward,
	CatsEye_deconvolutional_forward,
	CatsEye_maxpooling_forward,
	CatsEye_avgpooling_forward,
	CatsEye_global_avgpooling_forward,
	CatsEye_PixelShuffler_forward,
//	CatsEye_rnn_forward,

	CatsEye_padding_forward,
	CatsEye_BatchNormalization_forward,

	// activation
	CatsEye_act_sigmoid,
	CatsEye_act_softmax,
	CatsEye_act_tanh,
	CatsEye_act_ReLU,
	CatsEye_act_LeakyReLU,
	CatsEye_act_ELU,
	CatsEye_act_RReLU,

	// loss
	CatsEye_none,		// 0-1 loss
	CatsEye_none,		// mse loss
	CatsEye_none,		// cross-entropy loss
	CatsEye_none,		// cross-entropy multiclass loss

//	CatsEye_none,		// binary classification
//	CatsEye_none,		// multi-value classification
//	CatsEye_none,		// regression
	CatsEye_loss_mse,	// identity with mse loss
	CatsEye_none,		// sigmoid with cross-entropy loss
	CatsEye_none,		// softmax with multi cross-entropy loss
};
static void (*_CatsEye_layer_backward[])(CatsEye_layer *l) = {
	CatsEye_linear_backward,
	CatsEye_convolutional_backward,
	CatsEye_deconvolutional_backward,
	CatsEye_maxpooling_backward,
	CatsEye_avgpooling_backward,
	CatsEye_global_avgpooling_backward,
	CatsEye_PixelShuffler_backward,
//	CatsEye_rnn_backward,

	CatsEye_padding_backward,
	CatsEye_BatchNormalization_backward,

	// activation
	CatsEye_dact_sigmoid,
	CatsEye_dact_softmax,
	CatsEye_dact_tanh,
	CatsEye_dact_ReLU,
	CatsEye_dact_LeakyReLU,
	CatsEye_dact_ELU,
	CatsEye_dact_RReLU,

	// loss
	CatsEye_loss_delivative_0_1,
	CatsEye_loss_delivative_mse,
	CatsEye_loss_delivative_cross_entropy,
	CatsEye_loss_delivative_cross_entropy_multiclass,

//	CatsEye_none,		// binary classification
//	CatsEye_none,		// multi-value classification
//	CatsEye_none,		// regression
	CatsEye_loss_delivative_mse,	// (y-t) identity with mse loss
	CatsEye_loss_delivative_mse,	// (y-t) sigmoid with cross-entropy loss
	CatsEye_loss_delivative_0_1,	// (y-t) softmax with multi cross-entropy loss
};
static void (*_CatsEye_layer_update[])(CatsEye_layer *l) = {
	CatsEye_linear_update,
	CatsEye_convolutional_update,
	CatsEye_deconvolutional_update,
	CatsEye_none,	// maxpool
	CatsEye_none,	// avgpool
	CatsEye_none,	// global avgpool
	CatsEye_none,	// Pixel Shuffler
//	CatsEye_rnn_update,

	CatsEye_none,	// padding
	CatsEye_none,	// Batch Normalization

	// activation
	CatsEye_none,	// sigmoid
	CatsEye_none,	// softmax
	CatsEye_none,	// tanh
	CatsEye_none,	// ReLU
	CatsEye_none,	// LeakyReLU
	CatsEye_none,	// ELU
	CatsEye_none,	// RReLU

	// loss
	CatsEye_none,	// 0-1 loss
	CatsEye_none,	// mse loss
	CatsEye_none,	// cross-entropy loss
	CatsEye_none,	// cross-entropy multiclass loss

//	CatsEye_none,	// binary classification
//	CatsEye_none,	// multi-value classification
//	CatsEye_none,	// regression
	CatsEye_none,	// identity with mse loss
	CatsEye_none,	// sigmoid with cross-entropy loss
	CatsEye_none,	// softmax with multi cross-entropy loss
};
typedef enum {
	CATS_LINEAR, CATS_CONV, CATS_DECONV, CATS_MAXPOOL, CATS_AVGPOOL, CATS_GAP,
	CATS_PIXELSHUFFLER, /*CATS_RECURRENT,*/
	CATS_PADDING, CATS_BATCHNORMAL,

	CATS_ACT_SIGMOID, CATS_ACT_SOFTMAX, CATS_ACT_TANH,
	CATS_ACT_RELU, CATS_ACT_LEAKY_RELU, CATS_ACT_ELU, CATS_ACT_RRELU,

	CATS_LOSS_0_1, CATS_LOSS_MSE, CATS_LOSS_BCE, CATS_LOSS_CE,

	// y-t
//	CATS_LOSS_BC, CATS_LOSS_MC, CATS_LOSS_REGRESSION,
	CATS_LOSS_IDENTITY_MSE, CATS_SIGMOID_BCE, CATS_SOFTMAX_CE
} CATS_LAYER_TYPE;
char CatsEye_string[][16] = {
	"dense", "conv", "deconv", "max", "avg", "gap",
	"subpixel", /*"rnn",*/
	"pad", "bn",

	"sigmoid", "softmax", "tanh",
	"relu", "leaky", "elu", "rrelu",

	"binary", "mse", "bce", "ce",

//	"binary class", "multi class", "regression",
	"identity/mse", "sigmoid/bce", "softmax/ce",
};

#define CATS_INIT			{ .batch=1 }
#define CatsEye__construct(t, p)	_CatsEye__construct(t, p, sizeof(p)/sizeof(CatsEye_layer))
void _CatsEye__construct(CatsEye *this, CatsEye_layer *layer, int layers)
{
	//this->batch = 1;
	this->u = 0;
	this->z = 0;
	this->layers = layers;
	this->layer = calloc(this->layers, sizeof(CatsEye_layer));
	memcpy(this->layer, layer, this->layers*sizeof(CatsEye_layer));

	// calculate parameters
	int dsize[this->layers], wsize[this->layers];
	this->o = malloc(sizeof(real*)*(this->layers));	// outputs
	this->osize = 0;
	this->d = malloc(sizeof(real*)*(this->layers/*-1*/));	// errors
	this->dsize = 0;
	this->w = malloc(sizeof(real*)*(this->layers/*-1*/));	// weights
	this->ws = malloc(sizeof(int)*(this->layers/*-1*/));
	this->wsize = 0;

	int n[this->layers], m[this->layers], b[this->layers];
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		l->p = (void*)this;

		if (l->momentum==0.0) l->momentum = 0.9; // MomentumSGD

		l->forward = _CatsEye_layer_forward[l->type];
		l->backward = _CatsEye_layer_backward[l->type];
		if (!i /*&& l->type!=CATS_RECURRENT*/) l->backward = CatsEye_none;
		l->update = _CatsEye_layer_update[l->type];

		if (i>=this->layers-1) { // last layer
			if (!l->outputs) l->outputs = 0;//1; // FIXME
		}
		//osize[i] = 0;
		if (i>0) { // NOT first layer
			if (!l->ich) l->ich = (l-1)->ch;
			if (!l->inputs) {
				l->inputs = (l-1)->outputs;
			} else if (l->inputs<0) { // select the layer
				l->inputs = (l+l->inputs)->outputs;
				//osize[i] = osize[i+l->inputs];
			}
//			if (l->inputs<=0) l->inputs = (l-1+l->inputs)->outputs;
		} else { // first layer
			if (!l->ch) l->ch = 1;
			if (!l->ich) l->ich = 1;
		}
		if (!l->sx && l->ich) l->sx = l->sy = (int)sqrt(l->inputs/l->ich);

		n[i] = m[i] = b[i] = 0;
		switch (l->type) {
		case CATS_CONV:
			l->ox = l->px = (l->sx +2*l->padding -l->ksize) /l->stride +1;
			l->oy = l->py = (l->sy +2*l->padding -l->ksize) /l->stride +1;
			l->px -= l->padding*2;
			l->py -= l->padding*2;
			l->pz = l->padding +l->padding*l->ox;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ksize * l->ksize;	// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->ksize*l->ksize*l->ich *this->batch);
			printf("%3d %-8s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) printf("\t↑ warning: stride is strange!\n");
			break;
		case CATS_DECONV: // https://blog.shikoan.com/pytorch-convtranspose2d/
			l->ox = l->px = (l->sx-1) *l->stride -2*l->padding +l->ksize;
			l->oy = l->py = (l->sy-1) *l->stride -2*l->padding +l->ksize;
			l->px -= l->padding*2;
			l->py -= l->padding*2;
			l->pz = l->padding +l->padding*l->ox;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ksize * l->ksize;	// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->ksize*l->ksize*l->ich *this->batch);
			printf("%3d %-8s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_AVGPOOL:
		case CATS_MAXPOOL:
			l->ch = l->ich;
			l->ox = (l->sx +2*l->padding -l->ksize) /l->stride +1;
			l->oy = (l->sy +2*l->padding -l->ksize) /l->stride +1;
//			l->ox = (l->sx +2*l->padding -l->ksize +(l->stride-1)) /l->stride +1;
//			l->oy = (l->sy +2*l->padding -l->ksize +(l->stride-1)) /l->stride +1;
			l->outputs = l->ch * l->ox * l->oy;
			if (l->type == CATS_MAXPOOL) {
				l->workspace = malloc(sizeof(int)* l->outputs *this->batch);
			}
			printf("%3d %-8s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) printf("\t↑ warning: stride is strange!\n");
			break;
		case CATS_GAP:
			l->outputs = l->ch = l->ich;
			l->ox = 1;
			l->oy = 1;
			printf("%3d %-8s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_PIXELSHUFFLER:
			if (l->ch * l->r*l->r != l->ich) {
				l->sx = l->sy = (int)sqrt(l->inputs/ (l->ch * l->r*l->r));
			}
			l->ich = l->ch * l->r*l->r;
			l->ox = l->sx * l->r;
			l->oy = l->sy * l->r;
			l->outputs = l->ch * l->ox * l->oy;
			printf("%3d %-8s %10d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		/*case CATS_RECURRENT:
			l->wi = calloc(l->inputs * l->hiddens, sizeof(real));
//			l->wi = l->w;
			l->wr = calloc(l->hiddens * l->hiddens, sizeof(real));
			l->wo = calloc(l->hiddens * l->outputs, sizeof(real));
			l->dOuti = calloc(l->inputs * l->hiddens, sizeof(real));
			l->dOutr = calloc(l->hiddens * l->hiddens, sizeof(real));
			l->dOuto = calloc(l->hiddens * l->outputs, sizeof(real));
			l->eh = calloc(l->inputs * l->hiddens, sizeof(real));
			l->s = calloc(l->inputs * l->hiddens, sizeof(real));
			l->u = calloc(l->inputs * l->hiddens, sizeof(real));
			l->v = calloc(l->inputs * l->outputs, sizeof(real));
//			l->y = calloc(l->inputs * l->outputs, sizeof(real));
//			l->y = l->z;
			//l->act2 = CatsEye_act[CATS_ACT_IDENTITY];	// FIXME
			//l->dact2 = CatsEye_dact[CATS_ACT_IDENTITY];	// FIXME
//			l->act2 = CatsEye_act[CATS_ACT_SIGMOID];
//			l->dact2 = CatsEye_dact[CATS_ACT_SIGMOID];
//			n[i] = l->hiddens;
			n[i] = l->inputs;
			m[i] = l->hiddens;
			printf("%3d %-8s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;*/

		case CATS_PADDING:
			l->ch = l->ich;
			l->ox = l->sx + l->padding*2;
			l->oy = l->sy + l->padding*2;
			l->outputs = l->ch * l->ox * l->oy;
			printf("%3d %-8s %4d     %d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->ch, l->padding, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_BATCHNORMAL:
			l->gamma = 1.0;
			l->beta = 0;
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			printf("%3d %-8s %10d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_ACT_RRELU:
			l->min = 0;
			l->max = 0.05;
		case CATS_ACT_LEAKY_RELU:
			if (l->alpha==0.0) l->alpha = 0.01;
		case CATS_ACT_SIGMOID:
		case CATS_ACT_SOFTMAX:
		case CATS_ACT_TANH:
		case CATS_ACT_RELU:
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			printf("%3d %-8s %10d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_LINEAR:
		case CATS_LOSS_0_1:
		case CATS_LOSS_MSE:
		default:
			if (i<this->layers-1 && !l->outputs) {
				if ((l+1)->inputs>0) l->outputs = (l+1)->inputs;
				else { l->outputs = l->inputs; printf("\t↓ warning: out:%d\n", l->outputs); }
			}
			if (l->type == CATS_LINEAR) {
				n[i] = l->inputs;
				m[i] = l->outputs;
				b[i] = 1; // bias
				l->ox = l->oy = 1;
				l->ch = l->outputs;
			}
			printf("%3d %-8s %10d %4d x%4d x%4d -> %4d x%4d x%4d\n", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
//			printf("%3d %-8s %10d %4d x%4d x%4d -> loss\n", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich);
		}
		if (!l->inputs) {
			printf("\t↑ warning: input is strange!\n");
		}
		wsize[i] = this->wsize;
		this->ws[i] = (n[i]+b[i])*m[i];
		this->wsize += this->ws[i];

/*		if (!osize[i]) { // NOT select the layer
			osize[i] = this->osize;
			this->osize += l->inputs;//+1; // FIXME: bias
		}*/

		dsize[i] = this->dsize;
		this->dsize += l->outputs;//+1; // FIXME: bias
	}
	this->osize = this->dsize +this->layer[0].inputs +this->layer[this->layers-1].inputs; // input+output
	this->odata = calloc(this->osize*this->batch, sizeof(real));
	this->ddata = calloc(this->dsize*this->batch, sizeof(real));
	this->wdata = calloc(this->wsize*3, sizeof(real)); // w, dw and g
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];

		if (!i) l->x = this->odata;
		else l->x = this->odata +(this->layer[0].inputs +dsize[i-1])*this->batch;
		l->z = this->odata +(this->layer[0].inputs +dsize[i])*this->batch;

		if (i>0) l->dIn = (l-1)->dOut;
		l->dOut = this->ddata + dsize[i]*this->batch;
		l->bias = this->wdata + wsize[i] +n[i]*m[i]; // FIXME: for bias
		l->w = this->wdata + wsize[i];
		l->dw = this->wdata + this->wsize + wsize[i];
		l->g = this->wdata + this->wsize*2 + wsize[i];

//		this->o[i] = this->odata + osize[i];	// input
//		this->d[i] = this->ddata + dsize[i];
		this->o[i] = l->x;
		this->d[i] = l->dOut;
		this->w[i] = this->wdata + wsize[i];

		if (i==this->layers-1) {
			if ((l->type==CATS_SOFTMAX_CE && (l-1)->type==CATS_ACT_SOFTMAX) ||
				(l->type==CATS_SIGMOID_BCE && (l-1)->type==CATS_ACT_SIGMOID)) {
				// FIXME: for loss function
				// https://www.renom.jp/ja/notebooks/tutorial/basic_algorithm/lossfunction/notebook.html
				(l-1)->backward = CatsEye_none;
				(l-2)->dOut = l->dIn;
			}
			break;
		}

//		l->eta /= this->batch;
//		this->o[i][l->outputs] = 1;	// FIXME: bias

		// initialize weights, range depends on the research of Y. Bengio et al. (2010)
		// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
		xor128_init(time(0));
		real range = sqrt(6)/sqrt(n[i]+m[i]+2);
		if (l->type == CATS_LINEAR) {
			range = sqrt(2)/sqrt(n[i]+m[i]);
		}
		for (int j=0; j<this->ws[i]; j++) {
#ifdef CATS_WEIGHT_UNIFORM
			this->w[i][j] = 2.0*range*frand()-range; // uniform
#else
			this->w[i][j] = rand_normal(0, range); // normal
#endif
		}
//		memcpy(&this->wdata[this->wsize], this->wdata, this->wsize*sizeof(real));	// for debug
	}
	this->clasify = (int16_t*)this->layer[this->layers-1].z;
	this->label = this->layer[this->layers-1].z;

	uint64_t max = 0;
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		printf("L%02d in:%d out:%d (x:%ld-z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->x-this->odata, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);

		uint64_t s = l->ox*l->oy*l->ksize*l->ksize*l->ich;
		if (max < s) max = s;
	}
	this->mem = calloc(max*this->batch, sizeof(real));
	uint64_t wmem = this->osize*this->batch+this->dsize*this->batch+this->wsize*3;
	printf("Memory: %.1f MiB [%d B], Working Memory: %.1f MiB [%lu B]\n\n", this->wsize/1024/1024., this->wsize, wmem/1024/1024., wmem);

	this->start = this->stop = 0;
	this->end = this->layers-1;
	this->slide = this->layer[0].inputs;

//	sgemm_init(wmem);
	//sgemm_init(128*128*sizeof(real));
	sgemm_init(1024*1024*1024);
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	sgemm_finish();

	// delete arrays
//	printf("%x %x %x %x %x %x %x %x %x\n",this->z,this->odata,this->o,this->ddata,this->d,this->ws,this->wdata,this->w,this->u);
	free(this->mem);
	for (int i=0; i<this->layers-1; i++) {
		CatsEye_layer *l = &this->layer[i];
		if (l->workspace) free(l->workspace);
	}
	free(this->layer);
//	for (int i=0; i<this->layers-1; i++) free(this->z[i]);
	if (this->z) free(this->z);
	free(this->odata);
	free(this->o);
	free(this->ddata);
	free(this->d);
/*	free(this->e2);
	free(this->e3);
	free(this->m2);
	free(this->m3);
	free(this->v2);
	free(this->v3);*/
	free(this->ws);
	free(this->wdata);
	free(this->w);
	if (this->u) free(this->u);
}

int CatsEye_getLayer(CatsEye *this, char *name)
{
	CatsEye_layer *l = this->layer;
	for (int i=0; i<this->layers; i++) {
		if (l->name && !strcmp(l->name, name)) return i;
		l++;
	}
	return -1;
}

void CatsEye_forward(CatsEye *this, real *x)
{
	int b = this->batch; // FIXME
	this->batch = 1;

	CatsEye_layer *l = &this->layer[this->start];
	memcpy(l->x, x, l->inputs*sizeof(real));

//	this->layer[0].x = x; // FIXME
#ifdef CATS_DENOISING_AUTOENCODER // FIXME
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<this->layer[0].inputs; i++) {
		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
	}
#endif
	for (int i=this->start; i<this->end; i++) {
		l->x[l->inputs] = 1;	// for bias
		l->forward(l);
		l++;
	}

	this->batch = b; // FIXME
}

// return most probable label to the input x
int16_t CatsEye_predict(CatsEye *this, real *x)
{
	// forward propagation
	CatsEye_forward(this, x);

	// biggest output means most probable label
	CatsEye_layer *l = &this->layer[this->end];
	real max = l->x[0];
	int ans = 0;
	for (int i=1; i<l->inputs; i++) {
		if (l->x[i] > max) {
			max = l->x[i];
			ans = i;
		}
	}
	return ans;
}

int CatsEye_accuracy(CatsEye *this, real *x, int16_t *t, int verify)
{
	int r = 0;
	for (int i=0; i<verify; i++) {
		int16_t p = CatsEye_predict(this, x+i*this->slide);
		if (p==t[i]) r++;
		//else printf("%d/%d ",p,t[i]);
	}
	return r;
}

int CatsEye_train(CatsEye *this, real *x, void *t, int N, int epoch, int random, int verify)
{
	int a = this->end;	// layers-1
	this->layer[a].z = t;	// FIXME
	int lsize = this->layer[a].inputs *sizeof(real);
	if (this->layer[a].type==CATS_LOSS_0_1 ||
		this->layer[a].type==CATS_SOFTMAX_CE /*|| this->layer[a].type==CATS_SIGMOID_BCE*/) {
		lsize = sizeof(int16_t);
	}

	if (verify) N -= verify;	// for test
	int repeat = N;			// for random
	if (random) repeat = random;
	if (this->batch>1) {
//		repeat = N/this->batch;
		repeat /= this->batch;
		printf(" repeat: %d\n", repeat);
	}

	printf("epoch    loss     elapsed time\n");

	struct timeval start, stop;
	gettimeofday(&start, NULL);
	for (int times=0; times<epoch; times++) {
		for (int n=0; n<repeat; n++) {
			CatsEye_layer *l = &this->layer[this->start];

			// create data
			for (int b=0; b<this->batch; b++) {
				int sample = random ? (int)(frand()*N) : n;
				memcpy(l->x +l->inputs*b, x+sample*this->slide, l->inputs*sizeof(real));
				memcpy((int8_t*)this->label +lsize*b, (int8_t*)t+lsize*sample, lsize);
//				memcpy((int8_t*)this->label +lsize*b, (int8_t*)this->layer[a].z+lsize*sample, lsize);
//printf("label %d: %d\n", sample, *((int8_t*)t+lsize*sample));
//printf("label %d: %x %x %x %x\n", sample, this->layer[a].z, &this->layer[a].z[sample], t, ((int8_t*)t+lsize*sample));
//printf("d:%f label %d: %f %f\n", l->x[3], sample, this->layer[a].z[sample], *(real*)((int8_t*)t+lsize*sample));
//printf("d:%f label: %f %f\n", l->x[3], this->label[0], *((real*)t+sample));
			}

			// forward propagation
			for (int i=this->start; i<this->end; i++) {
				l->forward(l);
				l++;
			}
//			CatsEye_forward(this, x+sample*this->slide);

			// calculate the error and update the weights
#if 0
			int i = this->end;
			l = &this->layer[i--];
			l->backward(l);
/*{
	for (int i=0; i<l->inputs; i++) {
		for (int n=0; n<this->batch; n++) {
			l->dIn[i] += l->dIn[l->inputs*n+i];
		}
		l->dIn[i] /= this->batch;
	}
}
int batch = this->batch;
this->batch = 1;*/
			l--;
			for (; i>=this->start; i--) {
#endif
			for (int i = this->end; i>=this->start; i--) {
				if (/*!(l->fix&2)*/i>=this->stop) l->backward(l);
				if (!(l->fix&1)) l->update(l);
				l--;
			}
//this->batch = batch;
		}

		real err = 0;
		{
			// calculate the mean squared error
			real mse = 0;
			CatsEye_layer *l = &this->layer[a];
//			for (int i=0; i<l->inputs; i++) { // FIXME batch
			for (int i=0; i<this->batch * l->inputs; i++) {
				mse += l->dIn[i] * l->dIn[i];
			}
			mse /= this->batch;
			err = 0.5 * (err + mse);
		}
/*		CatsEye_layer *l = &this->layer[a];
		for (int i=0; i<this->batch * l->inputs; i++) err += l->dIn[i] * l->dIn[i];
		err /= (this->batch * l->inputs);*/

		gettimeofday(&stop, NULL);
		printf("%7d, %f [%.2fs]", times, err, (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);

		if (verify) {
			int r = CatsEye_accuracy(this, x+this->layer[0].inputs*N, (int16_t*)t+N, verify);
			printf(" %.1f%%", (float)r/verify*100.0);
		}
		printf("\n");
		if (isnan(err) || isinf(err)) {
			return 0;
		}
	}
	return 1;
}

int CatsEye_saveCats(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp==NULL) return -1;

	printf("Saving #%d weight... %lu bytes\n", this->epoch, this->wsize*sizeof(real));
	fwrite(this->wdata, this->wsize*sizeof(real), 1, fp);
	fwrite(&this->epoch, sizeof(int), 1, fp);

	fclose(fp);
	return 0;
}
int CatsEye_loadCats(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "rb");
	if (fp==NULL) return -1;

	fread(this->wdata, this->wsize*sizeof(real), 1, fp);
	fread(&this->epoch, sizeof(int), 1, fp);
	printf("Loading #%d weight... %lu bytes\n", this->epoch++, this->wsize*sizeof(real));

	fclose(fp);
	return 0;
}

// save weights to json file
int CatsEye_saveJson(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (fp==NULL) return -1;

	CatsEye_layer *l = this->layer;
	for (int n=0; n<this->layers; n++) {
//		fprintf(fp, "var u%d = [%d,%d,%d,%d,%d,%d,%d,%d];\n", n, l->type, l->activation,
//			l->ch, l->inputs, l->sx, l->sy, l->ksize, l->stride);
		fprintf(fp, "var u%d = [%d,%d,%d,%d,%d,%d,%d];\n", n, l->type,
			l->ch, l->inputs, l->sx, l->sy, l->ksize, l->stride);
		l++;
	}
	fprintf(fp, "var u = [u0");
	for (int n=1; n<this->layers; n++) {
		fprintf(fp, ",u%d", n);
	}
	fprintf(fp, "];\n");

	l = this->layer;
	for (int n=0; n<this->layers-1; n++) {
		fprintf(fp, "var w%d = [", n+1);
//		if (l->type != CATS_MAXPOOL) {
			int i;
			for (i=0; i<this->ws[n]; i++) {
				fprintf(fp, "%lf,", this->w[n][i]);
			}
			fprintf(fp, "%lf", this->w[n][i]);
//		}
		fprintf(fp, "];\n");
		l++;
	}
	fprintf(fp, "var w = [w1");
	for (int n=1; n<this->layers-1; n++) {
		fprintf(fp, ",w%d", n+1);
	}
	fprintf(fp, "];\n");

	fclose(fp);
	return 0;
}

// visualize
void CatsEye_visualize(real *o, int n, int sx, uint8_t *p, int width, int ch)
{
	real max = o[0];
	real min = o[0];
	for (int i=1; i<n*ch; i++) {
		if (max < o[i]) max = o[i];
		if (min > o[i]) min = o[i];
	}
	for (int c=0; c<ch; c++) {
		for (int i=0; i<n; i++) {
			p[((i/sx)*width + i%sx)*ch +c] = (uint8_t)((o[i+c*n] - min) / (max - min) * 255.0);
		}
	}
}

// https://www.cs.toronto.edu/~kriz/cifar.html
real *CatsEye_loadCifar(char *name, int size, int lsize, int sample, int16_t **label)
{
	uint8_t *data = malloc((size+lsize)*sample);	// +1 for label
	if (!data) { printf("Can't open %s\n", name); return 0; }
	int16_t *t = malloc(sizeof(int16_t)*sample);
	if (!t) { printf("Can't open %s\n", name); return 0; }
//	real *x = malloc(sizeof(real)*(size+1)*(sample+1));	// +1 for bias
	real *x = malloc(sizeof(real)*size*sample);
	if (!x) { printf("Can't open %s\n", name); return 0; }

	FILE *fp = fopen(name, "rb");
	if (!fp) { printf("Can't open %s\n", name); return 0; }
	fread(data, (size+lsize)*sample, 1, fp);
	for (int n=0; n<sample; n++) {
		if (lsize==2) {
			int16_t *p = (int16_t*)&data[n*(size+2)];
			t[n] = *p;
		} else {
			t[n] = data[n*(size+lsize)];
		}
		for (int i=0; i<size; i++) x[n*size+i] = data[n*(size+lsize)+lsize+i] * (1.0/255.0);
	}
	// shuffle
	/*for (int n=0; n<sample; n++) {
		int a = (int)(frand()*sample);
		real d[size];
		memcpy(d, &x[n*size], sizeof(real)*size);
		memcpy(&x[n*size], &x[a*size], sizeof(real)*size);
		memcpy(&x[a*size], d, sizeof(real)*size);
		int16_t tt = t[n];
		t[n] = t[a];
		t[a] = tt;
	}*/
	fclose(fp);
	free(data);

	*label = t;
	return x;
}

// save weights to csv file
/*int CatsEye_save(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (fp==NULL) return -1;

	fprintf(fp, "%d %d %d\n", SIZE(0), SIZE(1), SIZE(2));

	int i;
	for (i=0; i<(SIZE(0)+1)*SIZE(1)-1; i++) {
		fprintf(fp, "%lf ", this->w[0][i]);
	}
	fprintf(fp, "%lf\n", this->w[0][i]);

	for (i=0; i<(SIZE(1)+1)*SIZE(2)-1; i++) {
		fprintf(fp, "%lf ", this->w[1][i]);
	}
	fprintf(fp, "%lf\n", this->w[1][i]);

	fclose(fp);
	return 0;
}*/

// save weights to binary file
/*int CatsEye_saveBin(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp==NULL) return -1;

	fwrite(&SIZE(0), sizeof(int), 1, fp);
	fwrite(&SIZE(1), sizeof(int), 1, fp);
	fwrite(&SIZE(2), sizeof(int), 1, fp);

	//fwrite(this->w[0], sizeof(real)*(SIZE(0)+1)*SIZE(1), 1, fp);
	//fwrite(this->w[1], sizeof(real)*(SIZE(1)+1)*SIZE(2), 1, fp);
	for (int i=0; i<(SIZE(0)+1)*SIZE(1); i++) {
		float a = this->w[0][i];
		fwrite(&a, sizeof(float), 1, fp);
	}
	for (int i=0; i<(SIZE(1)+1)*SIZE(2); i++) {
		float a = this->w[1][i];
		fwrite(&a, sizeof(float), 1, fp);
	}

	fclose(fp);
	return 0;
}*/

// visualize weights [w1]
/*void CatsEye_visualizeWeights(CatsEye *this, int n, int size, uint8_t *p, int width)
{
	real *w = &this->w[0][n];
	real max = w[0];
	real min = w[0];
	for (int i=1; i<SIZE(0); i++) {
		if (max < w[i * SIZE(1)]) max = w[i * SIZE(1)];
		if (min > w[i * SIZE(1)]) min = w[i * SIZE(1)];
	}
	for (int i=0; i<SIZE(0); i++) {
		p[(i/size)*width + i%size] = (uint8_t)((w[i * SIZE(1)] - min) / (max - min) * 255.0);
	}
}*/

real *CatsEye_loadMnist(char *name, char *name2, int sample, int **label)
{
	int size = 784;
	uint8_t *data = malloc((size+1)*sample);		// +1 for label
	if (!data) return 0;
	int *t = malloc(sizeof(int)*sample);
	if (!t) return 0;
	real *x = malloc(sizeof(real)*(size+1)*(sample+1));	// +1 for bias
	if (!x) return 0;

	FILE *fp = fopen(name, "rb");
	if (!fp) return 0;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	fp = fopen(name2, "rb");
	if (!fp) return 0;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);

	*label = t;
	return x;
}

#undef TYPE
#undef SIZE
#undef CH
#undef RANDOM

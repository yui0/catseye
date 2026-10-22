//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2023 Yuichiro Nakada
//---------------------------------------------------------

#ifndef CATSEYE_H
#define CATSEYE_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif

// CATS_QUIET: silence the network summary / training log
#ifdef CATS_QUIET
#define CATS_LOG(...)	((void)0)
#else
#define CATS_LOG(...)	printf(__VA_ARGS__)
#endif

#include <sys/time.h>
static inline float time_diff(struct timespec *start, struct timespec *end)
{
	return (end->tv_sec - start->tv_sec) + 1e-9*(end->tv_nsec - start->tv_nsec);
}

//#pragma GCC optimize("Ofast")
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#define _debug(...)	do { printf("%s(%d):", __func__, __LINE__); printf(__VA_ARGS__); } while (0)

// 16-byte aligned allocation (SIMD); calloc zeroes and checks the n*size overflow
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
static inline void* cats_malloc(size_t size) { return _aligned_malloc(size, 16); }
static inline void cats_free(void *p) { _aligned_free(p); }
#else
static inline void* cats_malloc(size_t size)
{
	void *p;
	return posix_memalign(&p, 16, size ? size : 1)==0 ? p : NULL;
}
static inline void cats_free(void *p) { (free)(p); }
#endif
static inline void* cats_calloc(size_t n, size_t size)
{
	if (size && n > (size_t)-1 / size) return NULL;
	void *p = cats_malloc(n * size);
	if (p) memset(p, 0, n * size);
	return p;
}
#ifndef CATS_NO_ALLOC_MACROS
#define malloc(size)	cats_malloc(size)
#define free(p)		cats_free(p)
#define calloc(n, size)	cats_calloc(n, size)
#endif

#ifndef CATS_INFER_ONLY

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
#define sgemm_init(s)				sgemm_ocl_init(0, 0, /*(s*3)*/0)
#define sgemm_finish()				sgemm_ocl_finish()
#define gemm_rnn(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('N', 'N', m, n, k, alpha, a, b, beta, c)
#define gemm_rnt(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('N', 'T', m, n, k, alpha, a, b, beta, c)
#define gemm_rtn(m, n, k, alpha, a, b, beta, c)	sgemm_ocl('T', 'N', m, n, k, alpha, a, b, beta, c)
#else
//#include "gemm_cpu.h"
static void gemm_rnn(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
static void gemm_rnt(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
static void gemm_rtn(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
// https://ogawa-sankinkoutai.seesaa.net/article/108848981.html
#define XOR128_MAX	18446744073709551615.0
// The state must be seeded so that it is not everywhere zero.
static int cats_seeded = 0;	// set once the user (or the constructor) seeded the generators
uint64_t seed[2];
void xor128_init(uint64_t s)
{
	cats_seeded = 1;
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

// xoroshiro generator taken from http://vigna.di.unimi.it/xorshift/xoroshiro128plus.c
uint64_t xoroshiro_s[2] = {
    0X922AC4EB35B502D9L,
    0XDA3AA4832B8F1D27L
};
void xoroshiro128plus_init(uint64_t s)
{
	cats_seeded = 1;
	for (int i=1; i<=2; i++) {
		xoroshiro_s[i-1] = s = 1812433253U * ( s ^ ( s >> 30 ) ) + i;
	}
}
static inline uint64_t rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}
uint64_t xoroshiro128plus()
{
	const uint64_t s0 = xoroshiro_s[0];
	uint64_t s1 = xoroshiro_s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	xoroshiro_s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	xoroshiro_s[1] = rotl(s1, 36); // c
	return result;
}

// taken from https://github.com/svaarala/duktape/blob/master/misc/splitmix64.c
uint64_t splitmix64_x; /* The state can be seeded with any value. */
uint64_t splitmix64_next()
{
	uint64_t z = (splitmix64_x += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

//#define CATS_USE_XOR128
#ifdef CATS_USE_XOR128
#define xrand()			xor128()
#define frand()			( (xor128() >> 11) * (1.0/9007199254740992.0) )	/* [0,1) */
#else
#define xrand()			xoroshiro128plus()
#define frand()			( (xoroshiro128plus() >> 11) * (1.0/9007199254740992.0) )	/* [0,1) */
#endif
#define _rand(max)		(int)( frand() * (max))
#define random(min, max)	( frand() * ((max) -(min)) +(min) )
#define irand(min, max)		( (int)(xrand() % (uint64_t)((max) -(min) +1)) +(min) )
// http://www.natural-science.or.jp/article/20110404234734.php (mt19937ar.h)
// https://omitakahiro.github.io/random/random_variables_generation.html
static inline real rand_normal(real mu, real sigma)
{
	real z = sqrt(-2.0*log(1.0 -frand())) * sin(2.0*M_PI*frand());	// 1-frand() is in (0,1]: no log(0)
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
	int fix;		// training / no training

	int wtype;		// weight init type
	real wrange;		// weight init

	int ksize, kw, kh;	// CNN
	int stride;		// CNN
	int padding, pw, ph;	// CNN
	int px, py, pz;		// CNN (AUTO: padding)
	int ich, ch;		// CNN
	int sx, sy, ox, oy;	// CNN (AUTO: i/o size)

	int r;			// Pixel Shuffler

	char *layer;		// concat
	struct __CatsEye_layer* l; // concat
	int offset;		// concat
//	int size;		// concat
	int order;		// concat

//	int hiddens;		// RNN
//	int truncatedTime;	// RNN

	union {
		real momentum;	// SGD with momentum
		real rho;	// RMSProp RHO
	} u;
	real beta1, beta2;	// Adam
	real *mu, *var;		// Batch Normalization [average, variance]
//	real *gamma, *beta;	// Batch Normalization
	real gamma, beta;	// Batch Normalization
	real alpha, min, max;	// Leaky ReLU, RReLU

	// auto config
	int outputs;		// output size
	int wsize;		// weight size
	real *x;		// input
	real *z;		// output
	real *bias;		// bias
	real *w, *dw, *g, *s;	// weight
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

	// research
	real forward_time;
	real backward_time;
//	real update_time;

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

	int da;			// use data augmentation
//	void (*set_data)(struct __CatsEye *this, CatsEye_layer *l, void *data, void *label, void *x, void *t);
	void (*set_data)(struct __CatsEye *this, CatsEye_layer *l, int n, int b);

	// train parameter
	int epoch;
	int batch;
	int slide;
	// label
	int16_t *clasify;
	real *label;

	real lr;		// learning rate
	real lambda;		// weight decay (L2 norm)

	int label_size;		// internal use
	void *label_data;	// internal use
	real *learning_data;	// internal use
	int data_num;		// internal use
	int *shuffle_buffer;	// internal use
	int shuffle_base;	// internal use

	real loss;
	int infer;		// 1 while CatsEye_forward() runs: batch normalization uses its running statistics

	// output layers [o = f(z)]
	real **z, **o, *odata;
	int osize;
	// gradient value
	real **d, *ddata;
	int dsize;
	// weights
	real **w, *wdata;
	int *ws, wsize;

	// working memory
	real *mem;
} CatsEye;

#ifndef RMSPROP_RHO
#define RMSPROP_RHO	0.9
#endif
#ifndef ADAM_BETA1
#define ADAM_BETA1	0.9
#define ADAM_BETA2	0.999
#endif

// https://qiita.com/omiita/items/1735c1d048fe5f611f80
#ifdef CATS_USE_MOMENTUM_SGD
 // MomentumSGD [ dw = u * dw - n * g ]
#define SOLVER(l)	CatsEye_optimizer_momentumSGD(l)
static void CatsEye_optimizer_momentumSGD(CatsEye_layer *l)
{
	for (int i=l->wsize-1; i>=0; i--) {
//		l->g[i] = l->u.momentum * l->g[i] -l->p->lr * l->dw[i];
		l->g[i] = l->u.momentum * l->g[i] -l->eta * l->dw[i];
		l->w[i] += l->g[i];
	}
}

#elif defined CATS_USE_ADAGRAD
 // adagrad [ g2[i] += g * g; w[i] -= eta * g / sqrt(g2[i]); ]
#define SOLVER(l)	CatsEye_optimizer_adagrad(l)
static void CatsEye_optimizer_adagrad(CatsEye_layer *l)
{
	for (int i=l->wsize-1; i>=0; i--) {
		l->s[i] += l->dw[i] * l->dw[i];
//		l->w[i] -= l->p->lr * l->dw[i] / (sqrt(l->s[i]) +1e-12);
		l->w[i] -= l->eta * l->dw[i] / (sqrt(l->s[i]) +1e-12);
	}
}

#elif defined CATS_USE_RMSPROP
#define SOLVER(l)	CatsEye_optimizer_RMSprop(l)
static void CatsEye_optimizer_RMSprop(CatsEye_layer *l)
{
	for (int i=l->wsize-1; i>=0; i--) {
		l->g[i] = l->u.rho * l->g[i] + (1 - l->u.rho) * l->dw[i] * l->dw[i];
//		l->w[i] -= l->p->lr * l->dw[i] / (sqrt(l->g[i] +1e-12));
		l->w[i] -= l->eta * l->dw[i] / (sqrt(l->g[i] +1e-12));
	}
}

#elif defined CATS_USE_ADAM
#define SOLVER(l)	CatsEye_optimizer_Adam(l)
static void CatsEye_optimizer_Adam(CatsEye_layer *l)
{
	for (int i=l->wsize-1; i>=0; i--) {
		l->g[i] = l->beta1 * l->g[i] + (1 - l->beta1) * l->dw[i];
		l->s[i] = l->beta2 * l->s[i] + (1 - l->beta2) * l->dw[i] * l->dw[i];
//		l->w[i] -= l->p->lr * l->g[i] / (sqrt(l->s[i] +1e-12));
		l->w[i] -= l->eta * l->g[i] / (sqrt(l->s[i] +1e-12));

//		l->w[i] -= l->eta * l->g[i]/(1 - l->beta1) / (sqrt(l->s[i]/(1 - l->beta2) +1e-12));
	}
}

#else // SGD
// https://tech-lab.sios.jp/archives/21823
// https://github.com/tiny-dnn/tiny-dnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
#define SOLVER(l)	CatsEye_optimizer_SGD(l)
static void CatsEye_optimizer_SGD(CatsEye_layer *l)
{
	for (int i=l->wsize-1; i>=0; i--) {
//		l->w[i] -= l->p->lr * (l->dw[i] +l->p->lambda *l->w[i]); // L2 norm
		l->w[i] -= l->eta * (l->dw[i] +l->p->lambda *l->w[i]); // L2 norm
	}
}
#endif

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
	for (int n=0; n<l->p->batch; n++) {
		for (int i=0; i<l->outputs; i++) l->z[n*l->outputs +i] += l->w[l->inputs*l->outputs +i]; // bias!!
	}
//	for (int i=l->inputs-10; i<l->inputs; i++) CATS_LOG("%f ", l->x[i]);
//	CATS_LOG("\n");

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

/*	CATS_LOG("gemm_rnn: ");
	for (int i=l->inputs-10; i<l->inputs; i++) CATS_LOG("%f ", l->dIn[i]);
	CATS_LOG("\n");
	CATS_LOG("GEMM1_RNN: ");
	sgemm_gl(GEMM1_RNN, l->p->batch, l->inputs, l->outputs, 1, l->dOut, l->w, 0, l->dIn);
	for (int i=l->inputs-10; i<l->inputs; i++) CATS_LOG("%f ", l->dIn[i]);
	CATS_LOG("\n");*/

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
	// dW = dOutT * x / batch: every layer accumulates the mean gradient over the batch in dw,
	// the optimizer (SOLVER) applies it to the weights and to the bias, which follows the weights
	gemm_rtn(l->outputs, l->inputs, l->p->batch, 1.0/l->p->batch, l->dOut, l->x, 1, l->dw);
	real *db = l->dw +l->inputs*l->outputs;	// bias!!
	for (int n=0; n<l->p->batch; n++) {
		for (int i=0; i<l->outputs; i++) db[i] += l->dOut[n*l->outputs +i] / l->p->batch;
	}
//	for (int i=0; i<10; i++) CATS_LOG("%f ", l->dOut[i]);
/*	for (int i=0; i<10; i++) CATS_LOG("%f ", l->w[i]);
	CATS_LOG("\n");*/

	// weights := weights - eta * input * gradientsT(output)
//	gemm_rnt(l->inputs, l->outputs, l->p->batch, -l->eta, l->x, l->dOut, 1, l->w);
#endif
}

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

	for (int c=0; c<channels_col; ++c) {
		int w_offset = c % patch_w;
		int h_offset = (c / patch_w) % patch_h;
		int c_im = c / patch_h / patch_w;
		for (int h=0; h<height_col; ++h) {
			for (int w=0; w<width_col; ++w) {
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
			workspace = l->workspace +l->ox*l->oy*l->kw*l->kh*l->ich *i;
			im2col(l->x +l->inputs*i, l->ich, l->sy, l->sx, l->kh, l->kw, l->ph, l->pw, l->stride, l->stride, workspace);
		}
		// z = W * x [A(m,k) B(k,n) C(m,n)], cnhw
		gemm_rnn(l->ch, l->ox*l->oy*1, l->kw*l->kh*l->ich, 1, l->w, workspace, 0, l->z +l->outputs*i);

		// https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
		// output := workspace * weights
//		gemm_rnn(l->ox*l->oy*1, l->ch, l->ksize*l->ksize*l->ich, 1, workspace, l->w, 0, l->z +l->outputs*i);
	}
#else
	// https://qiita.com/t-tkd3a/items/6b17f296d61d14e12953
	real *workspace;
	if (l->ksize==1) {
		workspace = l->x;
	} else {
		workspace = l->workspace;
		for (int i=0; i<l->p->batch; i++) {
			im2col(l->x +l->inputs*i, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace +l->ksize*l->ksize*l->ich *l->ox*l->oy*i);
		}
	}
	gemm_rnn(l->ch, l->ox*l->oy*l->p->batch, l->ksize*l->ksize*l->ich, 1, l->w, workspace, 0, l->z); // !!z: [ch][ox*sy*batch]
#endif
}
static void CatsEye_convolutional_backward(CatsEye_layer *l)
{
#if 1
	for (int i=0; i<l->p->batch; i++) {
		real *workspace = l->ksize!=1 ? l->p->mem : l->dIn +l->inputs*i;
		// dIn = W**T * dOut [A(m,k) B(k,n) C(m,n)]
		gemm_rtn(l->kw*l->kh*l->ich, l->ox*l->oy*1, l->ch, 1, l->w, l->dOut +l->outputs*i, 0, workspace);
		/*CATS_LOG("gemm_rtn: ");
		if (i==l->p->batch-1) for (int i=0; i<10; i++) CATS_LOG("%f ", workspace[i]);
		CATS_LOG("\n");*/
		if (l->ksize!=1) {
			col2im(workspace, l->ich, l->sy, l->sx, l->kh, l->kw, l->ph, l->pw, l->stride, l->stride, l->dIn +l->inputs*i);
//			for (int n=0; n<10; n++) CATS_LOG("%f ", l->dIn[l->inputs*i + l->inputs-1-n]);
//			CATS_LOG("[%d]\n", l->wsize);
		}
	}
#else
	real *workspace = l->ksize!=1 ? l->p->mem : l->dIn;
	gemm_rtn(l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, l->ch, 1, l->w, l->dOut, 0, workspace);
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
		real *workspace = l->ksize!=1 ? l->workspace +l->ox*l->oy*l->kw*l->kh*l->ich *i : l->x +l->inputs*i;
		// W = W - eta * dOut * x**T [A(m,k) B(k,n) C(m,n)]
//		gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*1, 1, l->dOut +l->outputs*i, workspace, 1, l->dw);
		gemm_rnt(l->ch, l->kw*l->kh*l->ich, l->ox*l->oy*1, 1.0/l->p->batch, l->dOut +l->outputs*i, workspace, 1, l->dw);
//		for (int i=0; i<100; i++) CATS_LOG("%f ", l->dw[l->wsize-1-i]);
//		CATS_LOG("\n");
	}
//	for (int i=0; i<l->wsize; i++) l->dw[i] /= l->p->batch; // avg
//	for (int i=0; i<10; i++) CATS_LOG("%f ", l->dOut[i +l->outputs*(l->p->batch-1)]);
//	for (int i=0; i<10; i++) CATS_LOG("%f ", l->dw[i]);
//	CATS_LOG("w:%d %d\n", l->wsize, l->ch*l->ksize*l->ksize*l->ich);
//	SOLVER(l);
#else
	real *workspace = l->ksize!=1 ? l->workspace : l->x;
	gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, -l->eta, l->dOut, workspace, 0, l->dw);
#endif
}

// https://github.com/pjreddie/darknet/blob/master/src/deconvolutional_layer.c
static void CatsEye_deconvolutional_forward(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		gemm_rtn(l->ksize*l->ksize*l->ch, l->sx*l->sy*1, l->ich, 1, l->w, l->x +l->inputs*i, 0, l->p->mem);
		col2im(l->p->mem, l->ch, l->oy, l->ox, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->z +l->outputs*i);
	}
}
static void CatsEye_deconvolutional_backward(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		real *workspace = l->workspace +l->sx*l->sy*l->kw*l->kh*l->ch *i;
		im2col(l->dOut +l->outputs*i, l->ch, l->oy, l->ox, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace);
		gemm_rnn(l->ich, l->sx*l->sy*1, l->ksize*l->ksize*l->ch, 1, l->w, workspace, 0, l->dIn +i*l->inputs);
	}
}
static void CatsEye_deconvolutional_update(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		real *workspace = l->workspace +l->sx*l->sy*l->kw*l->kh*l->ch *i;
		gemm_rnt(l->ich, l->ksize*l->ksize*l->ch, l->sx*l->sy*1, 1.0/l->p->batch, l->x +l->inputs*i, workspace, 1, l->dw);	// avg
	}
//	SOLVER(l);
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
	real *x = l->x;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // out
			real *o = l->z + c*l->ox*l->oy +l->outputs*i;

			for (int cc=0; cc<ch; cc++) { // in
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
	real *x = l->dIn;
	for (int i=0; i<l->p->batch; i++) {
		for (int c=0; c<l->ch; c++) { // out
			real *delta = l->dOut + c*l->ox*l->oy +l->outputs*i;

			for (int cc=0; cc<ch; cc++) { // in
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

// https://qiita.com/t-tkd3a/items/14950dbf55f7a3095600
// https://qiita.com/omiita/items/01855ff13cc6d3720ea4
// x is laid out [batch][ch][h*w] and l->r groups (channels) are normalized separately, each over the whole
// batch and all of its positions. Weight layout: w[0..r) running mean, w[r..2r) running variance,
// w[2r] gamma, w[2r+1] beta (so they are saved and loaded with the other weights).
// Training uses (and remembers in l->workspace) the batch statistics and updates the running ones;
// CatsEye_forward() (p->infer) uses the running statistics, so a single sample is normalized like in training.
#ifndef CATS_BN_MOMENTUM
#define CATS_BN_MOMENTUM	0.9
#endif
#define CATS_BN_GAMMA(l)	((l)->w[2*(l)->r])
#define CATS_BN_BETA(l)		((l)->w[2*(l)->r +1])
static void CatsEye_BatchNormalization_forward(CatsEye_layer *l)
{
	const int hw = l->inputs /l->r/*ch*/;
	const int batch = l->p->batch;
	const real gamma = CATS_BN_GAMMA(l), beta = CATS_BN_BETA(l);
	real *bmu = l->workspace, *bvar = l->workspace +l->r;	// batch statistics (for the backward pass)

	for (int n=0; n<l->r; n++) {
		real avg, var;
		if (l->p->infer) {
			avg = l->mu[n];
			var = l->var[n];
		} else {
			avg = 0;
			for (int b=0; b<batch; b++) {
				const real *x = l->x +(size_t)b*l->inputs +(size_t)n*hw;
				for (int i=0; i<hw; i++) avg += x[i];
			}
			avg /= (real)hw *batch;
			var = 0;
			for (int b=0; b<batch; b++) {
				const real *x = l->x +(size_t)b*l->inputs +(size_t)n*hw;
				for (int i=0; i<hw; i++) var += (x[i] -avg) * (x[i] -avg);
			}
			var /= (real)hw *batch;
			bmu[n] = avg;
			bvar[n] = var;
			l->mu[n] = CATS_BN_MOMENTUM * l->mu[n] + (1 -CATS_BN_MOMENTUM) * avg;
			l->var[n] = CATS_BN_MOMENTUM * l->var[n] + (1 -CATS_BN_MOMENTUM) * var;
		}
		const real sigma = sqrt(var + 1e-8);
		for (int b=0; b<batch; b++) {
			const real *x = l->x +(size_t)b*l->inputs +(size_t)n*hw;
			real *z = l->z +(size_t)b*l->inputs +(size_t)n*hw;
			for (int i=0; i<hw; i++) z[i] = (x[i] -avg) / sigma * gamma + beta;
		}
	}
}
// https://deepnotes.io/batchnorm
static void CatsEye_BatchNormalization_backward(CatsEye_layer *l)
{
	const int hw = l->inputs /l->r;
	const int batch = l->p->batch;
	const real len = (real)hw *batch;
	const real gamma = CATS_BN_GAMMA(l);
	real *bmu = l->workspace, *bvar = l->workspace +l->r;
	real dgamma = 0;
	real dbeta = 0;

	for (int n=0; n<l->r; n++) {
		const real sigma = sqrt(bvar[n] + 1e-8);
		real dvar = 0;
		real dmu = 0;
		real dmu2 = 0;

		for (int b=0; b<batch; b++) {
			const real *x = l->x +(size_t)b*l->inputs +(size_t)n*hw;
			const real *delta = l->dOut +(size_t)b*l->inputs +(size_t)n*hw;
			real *d = l->dIn +(size_t)b*l->inputs +(size_t)n*hw;
			for (int i=0; i<hw; i++) {
				real X_mu = x[i] - bmu[n];

				real dX_norm = delta[i] * gamma;
				dvar += dX_norm * X_mu;
				d[i] = dX_norm / sigma;
				dmu += - d[i]; //dX_norm / -sigma;
				dmu2 += -2.0 * X_mu;

				dbeta += delta[i];
				dgamma += delta[i] * X_mu / sigma;
			}
		}
		dvar *= -0.5 * pow(bvar[n] + 1e-8, -3.0/2.0);
		dmu += dvar * dmu2/len;

		real a = dmu / len;
		real c = dvar * 2.0 / len;
		for (int b=0; b<batch; b++) {
			const real *x = l->x +(size_t)b*l->inputs +(size_t)n*hw;
			real *d = l->dIn +(size_t)b*l->inputs +(size_t)n*hw;
			for (int i=0; i<hw; i++) d[i] += a + c * (x[i] - bmu[n]);
		}
	}
	// gamma/beta are shared by all channels: the summed gradients are applied by the optimizer
	if (!(l->fix&1)) {
		l->dw[2*l->r] += dgamma / batch;
		l->dw[2*l->r +1] += dbeta / batch;
	}
}

static void CatsEye_concat_forward(CatsEye_layer *l)
{
	real *x = l->x;
	real *mix = l->l->x +l->offset;
	real *z = l->z;
	for (int i=0; i<l->p->batch; i++) {
		if (l->order) {
			memcpy(z+(l->outputs - l->inputs), x, l->inputs*sizeof(real));
			memcpy(z, mix, (l->outputs - l->inputs)*sizeof(real));
		} else {
			memcpy(z, x, l->inputs*sizeof(real));
			memcpy(z+l->inputs, mix, (l->outputs - l->inputs)*sizeof(real));
		}
		x += l->inputs;
		mix += l->l->inputs;
		z += l->outputs;
	}
}
static void CatsEye_concat_backward(CatsEye_layer *l)
{
	real *d = l->dIn;
	real *delta = l->dOut;
	for (int i=0; i<l->p->batch; i++) {
		if (l->order) {
			memcpy(d, delta+(l->outputs - l->inputs), l->inputs*sizeof(real));
		} else {
			memcpy(d, delta, l->inputs*sizeof(real));
		}
//		memcpy(l->l->dIn+l->offset, l->dOut+l->inputs, (l->outputs - l->inputs)*sizeof(real));
		d += l->inputs;
		delta += l->outputs;
	}
}
static void CatsEye_shortcut_forward(CatsEye_layer *l)
{
	real *x = l->x;
	real *a = l->l->x +l->offset;
	real *z = l->z;
	for (int i=0; i<l->p->batch; i++) {
		memcpy(z, x, l->inputs*sizeof(real));
		if (l->r) { // *r
			int len = l->l->inputs / l->l->ich;
			for (int ic=0; ic<l->l->ich; ic++) {
				for (int c=0; c<l->r; c++) {
					for (int n=0; n<len; n++) z[(ic*l->r +c)*len +n] += a[ic*len +n];
				}
			}
		} else {
			for (int n=0; n<l->inputs; n++) z[n] += a[n];
		}
		x += l->inputs;
		a += l->l->inputs;
		z += l->outputs;
	}
}
static void CatsEye_shortcut_backward(CatsEye_layer *l)
{
	real *d1 = l->dIn;
	real *d2 = l->l->dIn;
	real *delta = l->dOut;
	for (int i=0; i<l->p->batch; i++) {
		memcpy(d1, delta, l->inputs*sizeof(real));
		if (d2) {
			if (l->r) { // *r
				memset(d2, 0, l->l->inputs*sizeof(real));
				int len = l->l->inputs / l->l->ich;
				for (int ic=0; ic<l->l->ich; ic++) {
					for (int c=0; c<l->r; c++) {
						for (int n=0; n<len; n++) d2[ic*len +n] += delta[(ic*l->r +c)*len +n];
					}
				}
			} else {
				memcpy(d2, delta, l->inputs*sizeof(real));
			}
			d2 += l->r ? l->l->inputs : l->inputs;
		}
		d1 += l->inputs;
		delta += l->outputs;
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
	real *y = l->z;\
	real *d = l->dIn;\
	real *dOut = l->dOut;\
	for (int i=0; i<l->inputs *l->p->batch; i++) {\
		*d++ = (*dOut++) * CATS_DACT_##type(*y, l);\
		y++;\
	}\
}
//	for (int b=0; b<l->p->batch; b++) for (int n=0; n<10; n++) CATS_LOG("%f[%f] ", l->dIn[b*l->inputs +l->inputs-1-n], l->z[b*l->inputs +l->inputs-1-n]);\
//	CATS_LOG("%d\n", l->wsize);\

#define sigmoid_gain			1//0.1
#define CATS_ACT_sigmoid(x, l)		(1.0 / (1.0 + exp(-(x) * sigmoid_gain)))
#define CATS_DACT_sigmoid(y, l)		((1.0-(y))*(y) * sigmoid_gain)
CATS_ACT_ARRAY(sigmoid);
CATS_DACT_ARRAY(sigmoid);

/*#define CATS_ACT_swish(x, l)		((x) / (1.0 + exp(-(x))))
#define CATS_DACT_swish(y, l)		((y) + (1.0-(y))*(y))
CATS_ACT_ARRAY(swish);
CATS_DACT_ARRAY(swish);*/

#define CATS_ACT_log(x, l)		(log(x))
#define CATS_DACT_log(y, l)		(exp(-(y)))	// y = log(x): d/dx = 1/x = exp(-y)
CATS_ACT_ARRAY(log);
CATS_DACT_ARRAY(log);

// softmax with loss function (output only)
static void CatsEye_act_softmax(CatsEye_layer *l)
{
	real *x = l->x;
	real *z = l->z;
	for (int n=0; n<l->p->batch; n++) {
#if 0
		real alpha = x[0];
//		for (int i=1; i<l->inputs; i++) alpha = alpha>x[i] ? alpha : x[i]; // FIXME
		for (int i=1; i<l->inputs; i++) alpha = alpha<x[i] ? alpha : x[i];
		real denom = 0.0;
		for (int i=0; i<l->inputs; i++) denom += exp(x[i] - alpha);

		for (int i=l->inputs/*out*/; i>0; i--) {
			real numer = exp(*x++ - alpha);
			*z++ = (numer / denom);
		}

//		x += l->inputs;
#endif
		real sum = 0;
		real max = x[0];
		for (int i=1; i<l->inputs; i++) max = max<x[i] ? x[i] : max;
		for (int i=0; i<l->inputs; i++) {
			real e = exp(*x++ - max); // calc diff to avoid overflow
			sum += e;
			z[i] = e;
		}
		for (int i=l->inputs/*out*/; i>0; i--) *z++ /= sum;
	}
}
// the Jacobian of softmax is not diagonal: dz_j = y_j * (dy_j - sum_k dy_k y_k)
// (softmax followed by CATS_SOFTMAX_CE skips this layer: the loss delivers y - t directly)
static void CatsEye_dact_softmax(CatsEye_layer *l)
{
	for (int n=0; n<l->p->batch; n++) {
		const real *y = l->z +l->inputs*n;
		const real *dOut = l->dOut +l->inputs*n;
		real *d = l->dIn +l->inputs*n;
		real dot = 0;
		for (int i=0; i<l->inputs; i++) dot += dOut[i] * y[i];
		for (int i=0; i<l->inputs; i++) d[i] = y[i] * (dOut[i] - dot);
	}
}

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
		real x1 = *x++;
		real x2 = x1 * x1;
		x1 *= 1.0 + x2 * (0.1653 + x2 * 0.0097);

		*z++ = x1 / sqrt(1.0 + x2);

//		real ep = exp(2* *x++);
//		*z++ = (ep-1) / (ep+1);
	}
}
#endif
#define CATS_DACT_tanh(y, l)		(1.0-(y)*(y))	// (1.0-tanh(x)*tanh(x))
CATS_DACT_ARRAY(tanh);

// rectified linear unit function
#define CATS_ACT_ReLU(x, l)		((x)>0 ? (x) : 0.0)
//#define CATS_DACT_ReLU(y, l)		((y)>0 ? 1.0 : 0.0)
#define CATS_DACT_ReLU(y, l)		((y)>0)
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

// L1, L2
// https://toeming.hatenablog.com/entry/2020/04/03/000925

// calculate the error of output layer
static void CatsEye_loss_delivative_0_1(CatsEye_layer *l)
{
/*	real wa = 0;
	real *w = l->p->wdata;
	for (int i=0; i<l->p->wsize; i++) wa += *w++; // L2 norm
	wa *= 0.01;
	for (int i=0; i<l->p->batch *l->inputs; i++) l->dIn[i] = wa;*/

	// y - t [ t <- one hot ]
	memcpy(l->dIn, l->x, sizeof(real)*l->inputs *l->p->batch);
//	for (int n=0; n<l->inputs; n++) CATS_LOG(" %f", l->dIn[n]);
//	CATS_LOG("\n");
//	l->dIn[a] -= 1;
	int16_t *a = l->p->clasify;
	for (int i=0; i<l->p->batch; i++) {
		l->dIn[l->inputs*i + *a++] -= 1;
/*		l->dIn[l->inputs*i + *a] -= 1;
		l->dIn[l->inputs*i + *a++] /= l->p->batch; // FIXME: right?*/
		/*CATS_LOG("batch %d:", i);
		for (int n=0; n<l->inputs; n++) CATS_LOG(" %f", l->dIn[l->inputs*i + n]);
		CATS_LOG("\n");*/
	}
}
// loss function for mse with identity and cross entropy with sigmoid
static void CatsEye_loss_mse(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *y = l->x;
	real mse = 0;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		mse += (*y - *t) * (*y - *t);
		y++;
		t++;
	}
	mse /= l->inputs *l->p->batch;
//	mse /= 2 *l->p->batch;
	l->p->loss = mse;
}
static void CatsEye_loss_delivative_mse(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
		// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
//		CATS_LOG("(%f-%f=%f) ", *y, *t, *y-*t);
		*d++ = *y++ - *t++;	// (y - t) * (y - t) / 2  ->  y - t
/*		*d = *y++ - *t++;	// (y - t) * (y - t) / 2  ->  y - t
		*d++ /= l->p->batch;	// FIXME: right?*/
	}
}
// cross-entropy loss function for (multiple independent) binary classifications
// https://qiita.com/kenta1984/items/59a9ef1788e6934fd962
static void CatsEye_loss_cross_entropy(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *y = l->x;
	real loss = 0;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		loss += - *t * log(*y +1e-12) - (1 - *t) * log(1 - *y +1e-12);
		y++;
		t++;
	}
	loss /= l->inputs *l->p->batch;
	l->p->loss = loss;
}
static void CatsEye_loss_delivative_cross_entropy(CatsEye_layer *l)
{
	real *t = l->p->label;
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		*d++ = (*y - *t++) / (*y * (1 - *y) +1e-12);	// -t * log(y) - (1 - t) * log(1 - y);
		y++;
	}
}
// cross-entropy loss function for multi-class classification
static void CatsEye_loss_cross_entropy_multiclass(CatsEye_layer *l)
{
	int16_t *a = l->p->clasify;
	real *y = l->x;
	real loss = 0;
	for (int i=0; i<l->p->batch; i++) {
		loss += - 1 * log(y[*a++] +1e-12);
		y += l->inputs;
	}
/*	real *t = l->p->label;
	real *y = l->x;
	real loss = 0;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		loss += - *t * log(*y);
		y++;
		t++;
	}*/
	loss /= l->p->batch;	// mean over the batch
	l->p->loss = loss;
}
static void CatsEye_loss_delivative_cross_entropy_multiclass(CatsEye_layer *l)
{
/*	real *t = l->p->label; // FIXME
	real *d = l->dIn;
	real *y = l->x;
	for (int i=0; i<l->inputs *l->p->batch; i++) {
		// https://www.yukisako.xyz/entry/napier-number
		*d++ = -(*t++) / *y++;	// -t * log(y);
	}*/
	// y - t [ t <- one hot ]
	memset(l->dIn, 0, sizeof(real)*l->inputs *l->p->batch);	// d(-log y_t)/dy_j is 0 except for the target
	int16_t *a = l->p->clasify;
	for (int i=0; i<l->p->batch; i++) {
		l->dIn[l->inputs*i + *a] = -1 / (l->x[l->inputs*i + *a] +1e-12);
//		l->dIn[l->inputs*i + *a] /= l->p->batch; // FIXME: right?
		a++;
	}
}

// ArcFace
/*static void CatsEye_loss_arcface(CatsEye_layer *l)
{
	real *x = l->x;
	// variance
	real l2 = 0;
	for (int i=0; i<l->inputs; i++) l2 += x[i] * x[i];
	l2 /= l->inputs;
	real sigma = sqrt(l2 + 1e-8);
	for (int i=0; i<l->inputs; i++) x[i] = x[i] / sigma;
}*/

static void CatsEye_none(CatsEye_layer *l)
{
}
static void (*_CatsEye_layer_forward[])(CatsEye_layer *l) = {
	CatsEye_linear_forward,
	CatsEye_convolutional_forward,
	CatsEye_convolutional_forward,		// conv1d
	CatsEye_deconvolutional_forward,
	CatsEye_maxpooling_forward,
	CatsEye_avgpooling_forward,
	CatsEye_global_avgpooling_forward,
	CatsEye_PixelShuffler_forward,
//	CatsEye_rnn_forward,

	CatsEye_padding_forward,
	CatsEye_BatchNormalization_forward,

	CatsEye_concat_forward,
	CatsEye_shortcut_forward,

	// activation
	CatsEye_act_sigmoid,
	CatsEye_act_softmax,
	CatsEye_act_tanh,
	CatsEye_act_ReLU,
	CatsEye_act_LeakyReLU,
	CatsEye_act_ELU,
	CatsEye_act_RReLU,
	CatsEye_act_log,

	// loss
	CatsEye_none,				// 0-1 loss
	CatsEye_loss_mse,			// mse loss
	CatsEye_loss_cross_entropy,		// cross-entropy loss
	CatsEye_loss_cross_entropy_multiclass,	// (0-1) cross-entropy multiclass loss

	CatsEye_loss_mse,			// identity with mse loss
	CatsEye_loss_cross_entropy,		// sigmoid with cross-entropy loss
	CatsEye_loss_cross_entropy_multiclass,	// softmax with multi cross-entropy loss
};
static void (*_CatsEye_layer_backward[])(CatsEye_layer *l) = {
	CatsEye_linear_backward,
	CatsEye_convolutional_backward,
	CatsEye_convolutional_backward,		// conv1d
	CatsEye_deconvolutional_backward,
	CatsEye_maxpooling_backward,
	CatsEye_avgpooling_backward,
	CatsEye_global_avgpooling_backward,
	CatsEye_PixelShuffler_backward,
//	CatsEye_rnn_backward,

	CatsEye_padding_backward,
	CatsEye_BatchNormalization_backward,

	CatsEye_concat_backward,
	CatsEye_shortcut_backward,

	// activation
	CatsEye_dact_sigmoid,
	CatsEye_dact_softmax,
	CatsEye_dact_tanh,
	CatsEye_dact_ReLU,
	CatsEye_dact_LeakyReLU,
	CatsEye_dact_ELU,
	CatsEye_dact_RReLU,
	CatsEye_dact_log,

	// loss
	CatsEye_loss_delivative_0_1,
	CatsEye_loss_delivative_mse,
	CatsEye_loss_delivative_cross_entropy,
	CatsEye_loss_delivative_cross_entropy_multiclass, // 0-1

	CatsEye_loss_delivative_mse,		// (y-t) identity with mse loss
	CatsEye_loss_delivative_mse,		// (y-t) sigmoid with cross-entropy loss
	CatsEye_loss_delivative_0_1,		// (y-t) softmax with multi cross-entropy loss
};
static void (*_CatsEye_layer_update[])(CatsEye_layer *l) = {
	CatsEye_linear_update,
	CatsEye_convolutional_update,
	CatsEye_convolutional_update,		// conv1d
	CatsEye_deconvolutional_update,
	CatsEye_none,	// maxpool
	CatsEye_none,	// avgpool
	CatsEye_none,	// global avgpool
	CatsEye_none,	// Pixel Shuffler
//	CatsEye_rnn_update,

	CatsEye_none,	// padding
	CatsEye_none,	// Batch Normalization

	CatsEye_none,	// mix / concat
	CatsEye_none,	// shortcut

	// activation
	CatsEye_none,	// sigmoid
	CatsEye_none,	// softmax
	CatsEye_none,	// tanh
	CatsEye_none,	// ReLU
	CatsEye_none,	// LeakyReLU
	CatsEye_none,	// ELU
	CatsEye_none,	// RReLU
	CatsEye_none,	// log

	// loss
	CatsEye_none,	// 0-1 loss
	CatsEye_none,	// mse loss
	CatsEye_none,	// cross-entropy loss
	CatsEye_none,	// cross-entropy multiclass loss

	CatsEye_none,	// identity with mse loss
	CatsEye_none,	// sigmoid with cross-entropy loss
	CatsEye_none,	// softmax with multi cross-entropy loss
};
typedef enum {
	CATS_LINEAR, CATS_CONV, CATS_CONV1D, CATS_DECONV, CATS_MAXPOOL, CATS_AVGPOOL, CATS_GAP,
	CATS_PIXELSHUFFLER, /*CATS_RECURRENT,*/
	CATS_PADDING, CATS_BATCHNORMAL,

	CATS_CONCAT,
	CATS_SHORTCUT,

	CATS_ACT_SIGMOID, CATS_ACT_SOFTMAX, CATS_ACT_TANH,
	CATS_ACT_RELU, CATS_ACT_LEAKY_RELU, CATS_ACT_ELU, CATS_ACT_RRELU,
	CATS_ACT_LOG,

	CATS_LOSS_0_1, CATS_LOSS_MSE, CATS_LOSS_BCE, CATS_LOSS_CE,

	// y-t
	CATS_LOSS_IDENTITY_MSE, CATS_SIGMOID_BCE, CATS_SOFTMAX_CE
} CATS_LAYER_TYPE;
// weight initialization (layer.wtype)
typedef enum {
	CATS_RAND, /*CATS_UNIFORM,*/ CATS_GLOROT_UNIFORM/*Xavier*/, CATS_GLOROT_NORMAL, CATS_HE_UNIFORM/*Kaiming*/, CATS_HE_NORMAL
} CATS_WEIGHT;
char CatsEye_string[][16] = {
	"dense", "conv", "conv1d", "deconv", "max", "avg", "gap",
	"subpixel", /*"rnn",*/
	"pad", "bn",

	"concat", "shortcut",

	"sigmoid", "softmax", "tanh",
	"relu", "leaky", "elu", "rrelu",
	"log",

	"binary", "mse", "bce", "ce",

	"identity/mse", "sigmoid/bce", "softmax/ce",
};

int CatsEye_getLayer(CatsEye *this, const char *name)
{
	if (!name) return -1;
	CatsEye_layer *l = this->layer;
	for (int i=0; i<this->layers; i++) {
		if (l->name && !strcmp(l->name, name)) return i;
		l++;
	}
	return -1;
}

#define CATS_INIT			{ .batch=1, .lambda=0 }
#define CatsEye__construct(t, p)	_CatsEye__construct(t, p, sizeof(p)/sizeof(CatsEye_layer))
void _CatsEye__construct(CatsEye *this, CatsEye_layer *layer, int layers)
{
	//this->batch = 1;
	if (!cats_seeded) { // random by default; call xoroshiro128plus_init(seed) beforehand to be reproducible
		xor128_init(time(0));
		xoroshiro128plus_init(time(0));
	}
	this->u = 0;
	this->z = 0;
	this->layers = layers;
	this->layer = calloc(this->layers, sizeof(CatsEye_layer));
	memcpy(this->layer, layer, this->layers*sizeof(CatsEye_layer));

	// calculate parameters
	int dsize[this->layers], wsize[this->layers];
	this->o = malloc(sizeof(real*)*(this->layers));	// outputs
	this->osize = 0;
	this->d = malloc(sizeof(real*)*(this->layers));	// errors
	this->dsize = 0;
	this->w = malloc(sizeof(real*)*(this->layers));	// weights
	this->ws = malloc(sizeof(int)*(this->layers));
	this->wsize = 0;

	int n[this->layers], m[this->layers], b[this->layers];
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		l->p = (void*)this;

		if (l->u.momentum==0.0) l->u.momentum = 0.9; // MomentumSGD
		//if (l->u.rho==0.0) l->u.rho = RMSPROP_RHO; // FIXME: RMSprop
		//if (l->beta1==0.0) l->beta1 = 0.9; // FIXME: Adam
		// Adam
		if (l->beta2==0.0) {
			l->beta1 = ADAM_BETA1;
			l->beta2 = ADAM_BETA2;
		}

		l->forward = _CatsEye_layer_forward[l->type];
		l->backward = _CatsEye_layer_backward[l->type];
		if (!i /*&& l->type!=CATS_RECURRENT*/) l->backward = CatsEye_none; // first layer
		l->update = _CatsEye_layer_update[l->type];

		if (i>=this->layers-1) { // last layer
			if (!l->outputs) l->outputs = 0;//1; // FIXME
		}
		if (i>0) { // NOT first layer
			if (!l->ich) l->ich = (l-1)->ch;
			if (!l->inputs) {
				l->inputs = (l-1)->outputs;
			} else if (l->inputs<0) { // select the layer
				l->inputs = (l+l->inputs)->outputs;
			}
//			if (l->inputs<=0) l->inputs = (l-1+l->inputs)->outputs;
		} else { // first layer
			if (!l->ch) l->ch = 1;
			if (!l->ich) l->ich = 1;
		}
		if (!l->sx && l->ich) l->sx = l->sy = (int)sqrt(l->inputs/l->ich);

		if (!l->kw) l->kw = l->kh = l->ksize; // conv2d, conv1d
		else l->ksize = l->kw;
		if (!l->pw) l->pw = l->ph = l->padding; // conv2d, conv1d
		else l->padding = l->pw;

		n[i] = m[i] = b[i] = 0;
		switch (l->type) {
		case CATS_CONV:
			l->ox = /*l->px =*/ (l->sx +2*l->pw -l->kw) /l->stride +1;
			l->oy = /*l->py =*/ (l->sy +2*l->ph -l->kh) /l->stride +1;
/*			l->px -= l->padding*2;
			l->py -= l->padding*2;
			l->pz = l->padding +l->padding*l->ox;*/
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->kw * l->kh;		// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->wsize = n[i] * m[i];
			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->kw*l->kh*l->ich *this->batch);
			CATS_LOG("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->kw, l->kh, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
//			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) CATS_LOG("\n\t↑ warning: stride is strange!\n");
			break;
		case CATS_CONV1D:
			l->sx = l->inputs / l->ich;
			l->ox = (l->sx +2*l->pw -l->ksize) /l->stride +1;
			l->oy = l->sy = l->kh = 1;
			l->ph = 0;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->kw;			// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->wsize = n[i] * m[i];
			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->kw*l->ich *this->batch);
			CATS_LOG("%3d %-12s %4d   %d/%d %4d       x%4d -> %4d       x%4d", i+1, CatsEye_string[l->type], l->ch, l->kw, l->stride, l->sx, l->ich, l->ox, l->ch);
			break;
		case CATS_DECONV: // https://blog.shikoan.com/pytorch-convtranspose2d/
			l->ox = (l->sx-1) *l->stride -2*l->padding +l->ksize;
			l->oy = (l->sy-1) *l->stride -2*l->padding +l->ksize;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ksize * l->ksize;	// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->wsize = n[i] * m[i];
//			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->kw*l->kh*l->ich *this->batch);
			l->workspace = malloc(sizeof(real)* l->sx*l->sy*l->kw*l->kh*l->ch *this->batch);
			CATS_LOG("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_AVGPOOL:
		case CATS_MAXPOOL:
			l->ch = l->ich;
//			l->ox = (l->sx +2*l->padding -l->ksize) /l->stride +1;
//			l->oy = (l->sy +2*l->padding -l->ksize) /l->stride +1;
			l->ox = (l->sx +2*l->padding -l->ksize +(l->stride-1)) /l->stride +1; // 32->16(k:3,s:2)
			l->oy = (l->sy +2*l->padding -l->ksize +(l->stride-1)) /l->stride +1;
			l->outputs = l->ch * l->ox * l->oy;
			if (l->type == CATS_MAXPOOL) {
				l->workspace = malloc(sizeof(int)* l->outputs *this->batch);
			}
			CATS_LOG("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) CATS_LOG("\n\t↑ warning: kernel or stride is strange!\n");
			break;
		case CATS_GAP:
			l->outputs = l->ch = l->ich;
			l->ox = 1;
			l->oy = 1;
			CATS_LOG("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_PIXELSHUFFLER:
			if (l->ch * l->r*l->r != l->ich) {
				l->sx = l->sy = (int)sqrt(l->inputs/ (l->ch * l->r*l->r));
			}
			l->ich = l->ch * l->r*l->r;
			l->ox = l->sx * l->r;
			l->oy = l->sy * l->r;
			l->outputs = l->ch * l->ox * l->oy;
			CATS_LOG("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
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
			CATS_LOG("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;*/

		case CATS_PADDING:
			l->ch = l->ich;
			l->ox = l->sx + l->padding*2;
			l->oy = l->sy + l->padding*2;
			l->outputs = l->ch * l->ox * l->oy;
			CATS_LOG("%3d %-12s %4d     %d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->padding, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_BATCHNORMAL:
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			l->r = l->sx==1? 1: l->ch;
			l->workspace = malloc(sizeof(real)* 2*l->r);	// batch mean, variance
			n[i] = l->r +1;
			m[i] = 2; // mean, variance (running) / gamma, beta
			l->wsize = n[i] * m[i];
			CATS_LOG("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_SHORTCUT:
			if (CatsEye_getLayer(this, l->layer)<0) { fprintf(stderr, "CatsEye: layer <%s> not found\n", l->layer ? l->layer : "(null)"); exit(1); }
			l->l = &this->layer[CatsEye_getLayer(this, l->layer)];
		case CATS_ACT_RRELU:
			l->min = 0;
			l->max = 0.05;
		case CATS_ACT_LEAKY_RELU:
			if (l->alpha==0.0) l->alpha = 0.01;
		case CATS_ACT_SIGMOID:
		case CATS_ACT_SOFTMAX:
		case CATS_ACT_TANH:
		case CATS_ACT_RELU:
		case CATS_ACT_ELU:	// element-wise: the output has the shape of the input
		case CATS_ACT_LOG:
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			CATS_LOG("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_CONCAT:
			if (CatsEye_getLayer(this, l->layer)<0) { fprintf(stderr, "CatsEye: layer <%s> not found\n", l->layer ? l->layer : "(null)"); exit(1); }
			l->l = &this->layer[CatsEye_getLayer(this, l->layer)];
		case CATS_LINEAR:
		case CATS_LOSS_0_1:
		case CATS_LOSS_MSE:
		default:
			if (i<this->layers-1 && !l->outputs) {
				if ((l+1)->inputs>0) l->outputs = (l+1)->inputs;
				else { l->outputs = l->inputs; CATS_LOG("\t↓ warning: out:%d\n", l->outputs); }
			}
			if (l->type==CATS_LINEAR) {
				l->wsize = (l->inputs +1) * l->outputs;	// weights and bias
				n[i] = l->inputs;
				m[i] = l->outputs;
				b[i] = 1; // bias
				l->ox = l->oy = 1;
				l->ch = l->outputs;
			}
			if (l->type==CATS_CONCAT) {
				l->ox = l->oy = 1;
				l->ch = l->outputs;
			}
			CATS_LOG("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
//			CATS_LOG("%3d %-12s %10d %4d x%4d x%4d -> loss", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich);
		}
		if (l->name) CATS_LOG(" <%s>\n", l->name);
		else if (l->layer) CATS_LOG(" +<%s:%d>\n", l->layer, l->r);
		else CATS_LOG("\n");
		if (!l->inputs) {
			CATS_LOG("\t↑ warning: input is strange!\n");
		}
		wsize[i] = this->wsize;
		this->ws[i] = (n[i]+b[i])*m[i];
		this->wsize += this->ws[i];

		dsize[i] = this->dsize;
		this->dsize += l->outputs;//+1; // FIXME: bias
	}
	this->osize = this->dsize +this->layer[0].inputs +this->layer[this->layers-1].inputs+ 1/*loss*/; // input+output
	this->odata = calloc(this->osize*this->batch, sizeof(real));
	this->ddata = calloc(this->dsize*this->batch, sizeof(real));
	this->wdata = calloc(this->wsize*4, sizeof(real)); // w, dw, g and s
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];

		if (!i) l->x = this->odata;
		else l->x = this->odata +(this->layer[0].inputs +dsize[i-1])*this->batch;
		l->z = this->odata +(this->layer[0].inputs +dsize[i])*this->batch;

		if (i>0) l->dIn = (l-1)->dOut;
		l->dOut = this->ddata + dsize[i]*this->batch;
		l->w = this->wdata +wsize[i];
		l->dw = this->wdata +this->wsize +wsize[i]; // stack +1
		l->g = this->wdata +this->wsize*2 +wsize[i];// stack +2
		l->s = this->wdata +this->wsize*3 +wsize[i];// stack +3
		l->bias = l->w +n[i]*m[i]; // FIXME: for bias

		l->mu = l->w;			// batch norm
		l->var = l->w + l->r;		// batch norm
//		l->gamma = l->w + l->inputs*2;	// batch norm
//		l->beta = l->w + l->inputs*3;	// batch norm

		this->o[i] = l->x;
		this->d[i] = l->dOut;
		this->w[i] = l->w; //this->wdata + wsize[i];

//		l->eta /= this->batch;
//		this->o[i][l->outputs] = 1;	// FIXME: bias

		if (i==this->layers-1) {
			if (i>=2 && ((l->type==CATS_SOFTMAX_CE && (l-1)->type==CATS_ACT_SOFTMAX) ||
				(l->type==CATS_SIGMOID_BCE && (l-1)->type==CATS_ACT_SIGMOID))) {
				// FIXME: for loss function
				// https://www.renom.jp/ja/notebooks/tutorial/basic_algorithm/lossfunction/notebook.html
				(l-1)->backward = CatsEye_none;
				(l-2)->dOut = l->dIn;
			}
			break;
		}
/*		if (l->type == CATS_BATCHNORMAL) {
			CATS_LOG("+!\n");
			for (int n=0; n<l->inputs; n++) {
			CATS_LOG("%d\n",n);
				l->gamma[n] = 1.0;
				l->beta[n] = 0;
			}
		}*/
		if (!l->wsize) { /*l->dw = 0;*/ continue; }
		if (l->type==CATS_BATCHNORMAL) { // mean 0, variance 1, gamma 1, beta 0
			for (int n=0; n<l->r; n++) { l->mu[n] = 0; l->var[n] = 1; }
			CATS_BN_GAMMA(l) = 1;
			CATS_BN_BETA(l) = 0;
			continue;
		}

		// initialize weights, range depends on the research of Y. Bengio et al. (2010)
		// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
		// https://ntacoffee.com/xavier-initialization/
		real range = sqrt(2)/sqrt(l->inputs+l->outputs);
		// https://pytorch.org/docs/stable/nn.init.html
		real gain = 1;
		int fin = l->ksize ? l->ksize*l->ksize*l->ich : l->inputs;
		if (i<this->layers-1) {
			int n = (l+1)->type==CATS_BATCHNORMAL ? 2 : 1;
			switch (i+n<this->layers ? (l+n)->type : -1) {
			case CATS_ACT_TANH:
				gain = 5.0/3.0;
				break;
			case CATS_ACT_RELU:
				gain = sqrt(2);
//				range = sqrt(3 / (fin));
				range = 1.0 / sqrt(fin);
				break;
			case CATS_ACT_RRELU:
			case CATS_ACT_LEAKY_RELU:
				gain = sqrt(2/(1 +l->alpha*l->alpha));
				range = 1.0 / sqrt(fin);
				break;
/*			case CATS_ACT_SELU:
				gain = 3.0/4.0;*/
			}
		}
		range *= gain;
		if (l->wrange!=0) range = l->wrange;
		switch (l->wtype) {
		case CATS_GLOROT_UNIFORM: // linear, sigmoid, tanh
			range = gain * sqrt(6)/sqrt(l->inputs+l->outputs);
			for (int n=0; n<l->wsize; n++) l->w[n] = 2.0*range*frand()-range; // uniform
			break;
		case CATS_GLOROT_NORMAL: // linear, sigmoid, tanh
			range = gain * sqrt(2)/sqrt(l->inputs+l->outputs);
			for (int n=0; n<l->wsize; n++) l->w[n] = rand_normal(0, range); // normal
			break;
		case CATS_HE_UNIFORM: // ReLU
			range = gain * sqrt(6.0 / fin);
//			range = gain * sqrt(3 / (l->inputs/l->ich));
			for (int n=0; n<l->wsize; n++) l->w[n] = 2.0*range*frand()-range; // uniform
			break;
		case CATS_HE_NORMAL: // ReLU
			range = gain * sqrt(2.0 / fin);
//			range = gain / sqrt(l->inputs/l->ich);
			for (int n=0; n<l->wsize; n++) l->w[n] = rand_normal(0, range); // normal
			break;
/*		case CATS_UNIFORM:
			range = sqrt(6)/sqrt(l->inputs+l->outputs);
			for (int n=0; n<l->wsize; n++) l->w[n] = 2.0*range*frand()-range; // uniform
			break;*/
		default: // CATS_RAND
			for (int n=0; n<l->wsize; n++) l->w[n] = rand_normal(0, range); // normal sigma:0.02
//			for (int n=0; n<l->wsize; n++) l->w[n] = 2.0*range*frand()-range; // uniform
		}
//		memcpy(&this->wdata[this->wsize], this->wdata, this->wsize*sizeof(real)); // for debug
		l->wrange = range;
		if (l->type==CATS_LINEAR) memset(l->w +l->inputs*l->outputs, 0, sizeof(real)*l->outputs);	// bias starts at 0
	}
	this->clasify = (int16_t*)this->layer[this->layers-1].z;
	this->label = this->layer[this->layers-1].z;
	CATS_LOG("\n");

	uint64_t max = 0;
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		CATS_LOG("L%02d in:%d out:%d weight:%d[%f] (z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->wsize, l->wrange, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
//		CATS_LOG("L%02d in:%d out:%d weight:%d (x:%ld-z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->wsize, l->x-this->odata, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
		//CATS_LOG("  (dIn:%ld-dOut:%ld)\n", l->dIn, l->dOut);

		uint64_t s = (uint64_t)l->ox*l->oy*l->ksize*l->ksize*l->ich; // col2im
		if (max < s) max = s;
		if (l->type==CATS_DECONV) { // gemm output before col2im
			s = (uint64_t)l->sx*l->sy*l->ksize*l->ksize*l->ch;
			if (max < s) max = s;
		}
	}
	this->mem = calloc(max/**this->batch*/, sizeof(real));
	uint64_t wmem = sizeof(real) * ((uint64_t)this->osize*this->batch+(uint64_t)this->dsize*this->batch+(uint64_t)this->wsize*3);
	CATS_LOG("Memory: %.1f MiB [%lu B], Working Memory: %.1f MiB [%lu B]\n\n", sizeof(real)*this->wsize/1024/1024., sizeof(real)*this->wsize, wmem/1024/1024., wmem);

	this->start = this->stop = 0;
	this->end = this->layers-1;
	this->slide = this->layer[0].inputs;
	this->shuffle_buffer = 0;

	sgemm_init(wmem);
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	sgemm_finish();

	// delete arrays
//	CATS_LOG("%x %x %x %x %x %x %x %x %x\n",this->z,this->odata,this->o,this->ddata,this->d,this->ws,this->wdata,this->w,this->u);
	if (this->shuffle_buffer) {
		free(this->shuffle_buffer);
		this->shuffle_buffer = 0;
	}
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

void CatsEye_forward(CatsEye *this, real *x)
{
	int b = this->batch; // FIXME
	this->batch = 1;
	this->infer = 1;

	CatsEye_layer *l = &this->layer[this->start];
	memcpy(l->x, x, l->inputs*sizeof(real));

	for (int i=this->start; i<this->end; i++) {
		l->x[l->inputs] = 1;	// for bias
		l->forward(l);
		l++;
	}

	this->batch = b; // FIXME
	this->infer = 0;
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
		//else CATS_LOG("%d/%d ",p,t[i]);
	}
	return r;
}

static inline void _CatsEye_DA_zoom(real *p, real *s, int w, int h, int ch, real r)
{
	if (r < 0.1) r = 0.1;	// r<1: zoom out (outside the image is 0), r>1: zoom in
	for (int c=0; c<ch; c++) {
		for (int y=0; y<h; y++) {
			int sy = (int)(y/r);
			for (int x=0; x<w; x++) {
				int sx = (int)(x/r);
				*p++ = (sx<w && sy<h) ? s[sy*w +sx] : 0;
			}
		}
		s += w*h;
	}
}
// shift by (px,py), the uncovered border is mirrored
static inline int _CatsEye_DA_mirror(int i, int n)
{
	if (i<0) i = -i;
	if (i>=n) i = 2*n -2 -i;
	return i<0 ? 0 : i>=n ? n-1 : i;
}
static inline void _CatsEye_DA_translationXY(real *p, real *s, int w, int h, int ch, int px, int py)
{
	for (int c=0; c<ch; c++) {
		for (int y=0; y<h; y++) {
			real *row = s + _CatsEye_DA_mirror(y-py, h)*w;
			for (int x=0; x<w; x++) *p++ = row[_CatsEye_DA_mirror(x-px, w)];
		}
		s += w*h;
	}
}
static inline void _CatsEye_DA_flip(real *p, real *s, int w, int h, int ch)
{
	for (int c=0; c<ch; c++) {
		for (int y=0; y<h; y++) {
			for (int x=0; x<w; x++) *p++ = s[y*w +w-1-x];
		}
		s += w*h;
	}
}
typedef enum {
	CATS_ORIGINAL = 1,
	CATS_NOISE = 2,
	CATS_ZOOM_IN = 4,
	CATS_ZOOM_OUT = 8,
	CATS_TRANSLATION = 16,
	CATS_TRANSLATION_X = 32,
	CATS_TRANSLATION_Y = 64,
	CATS_FLIP = 128,
} CATS_DA_TYPE;
static inline void _CatsEye_DataAugmentation(real *p, real *s, int sx, int sy, int ch, int f)
{
	int tbl[8], n=0;
	tbl[n] = CATS_ORIGINAL;
	if (f&CATS_ORIGINAL) tbl[n++] = CATS_ORIGINAL;
	if (f&CATS_NOISE) tbl[n++] = CATS_NOISE;
	if (f&CATS_ZOOM_IN) tbl[n++] = CATS_ZOOM_IN;
	if (f&CATS_ZOOM_OUT) tbl[n++] = CATS_ZOOM_OUT;
	if (f&CATS_TRANSLATION) tbl[n++] = CATS_TRANSLATION;
	if (f&CATS_TRANSLATION_X) tbl[n++] = CATS_TRANSLATION_X;
	if (f&CATS_TRANSLATION_Y) tbl[n++] = CATS_TRANSLATION_Y;
	if (f&CATS_FLIP) tbl[n++] = CATS_FLIP;

	int type = n ? irand(0, n-1) : 0;
	switch (tbl[type]) {
	case CATS_NOISE: // noise
		for (int i=0; i<sx*sy*ch; i++) {
			real sigma = 0.04;
			//real a = binomial(/*0.7(30%)*/0.9);
			p[i] = s[i] +rand_normal(0, sigma);	// zero-mean gaussian noise
		}
		break;
	case CATS_ZOOM_IN: // zoom in
		_CatsEye_DA_zoom(p, s, sx, sy, ch, random(1, 2));
		break;
	case CATS_ZOOM_OUT: // zoom out
		_CatsEye_DA_zoom(p, s, sx, sy, ch, random(0, 1));
		break;
	case CATS_TRANSLATION: // translation
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, irand(-(sx/4), sx/4), irand(-(sy/4), sy/4));
		break;
	case CATS_TRANSLATION_X: // translationX
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, irand(-(sx/4), sx/4), 0);
		break;
	case CATS_TRANSLATION_Y: // translationY
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, 0, irand(-(sy/4), sy/4));
		break;
	case CATS_FLIP:
		_CatsEye_DA_flip(p, s, sx, sy, ch);
		break;

	case CATS_ORIGINAL: // original
	default:
		memcpy(p, s, sx*sy*ch*sizeof(real));
	}
}
static inline void _CatsEye_shuffle(int *array, int n)
{
//	CATS_LOG("_CatsEye_shuffle: %d\n", n);
	for (int i=0; i<n-1; i++) {
//		int j = i + rand() / (RAND_MAX / (n - i) + 1);
		int j = i + (int)(xrand() % (uint64_t)(n-i));
		int t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
/*	for (int i=0; i<n; i++) {
		CATS_LOG(" %d", array[i]);
	}
	CATS_LOG("\n");*/
}
static inline void _CatsEye_data_transfer(CatsEye *this, real *x, void *l, int n)
{
	if (this->shuffle_buffer && this->data_num!=n) { // the data set changed size: reallocate
		free(this->shuffle_buffer);
		this->shuffle_buffer = 0;
	}
	if (!this->shuffle_buffer) this->shuffle_buffer = malloc(n * sizeof(int));
	for (int i=0; i<n; i++) this->shuffle_buffer[i] = i;
	_CatsEye_shuffle(this->shuffle_buffer, n);
	this->shuffle_base = 0;

	this->label_size = this->layer[this->end].inputs *sizeof(real);
	if (this->layer[this->end].type==CATS_LOSS_0_1 ||
		this->layer[this->end].type==CATS_LOSS_CE ||	// these losses read class indices (clasify)
		this->layer[this->end].type==CATS_SOFTMAX_CE /*|| this->layer[this->end].type==CATS_SIGMOID_BCE*/) {
		this->label_size = sizeof(int16_t);
	}

	this->data_num = n;
	this->learning_data = x;
	this->label_data = l;
}
void _CatsEye_set_data(CatsEye *this, CatsEye_layer *l, int n, int b)
{
	memcpy(l->x +l->inputs*b, this->learning_data+n*this->slide, l->inputs*sizeof(real));
	memcpy((int8_t*)this->label +this->label_size*b, (int8_t*)this->label_data +this->label_size*n, this->label_size/*bytes*/);
}
void _CatsEye_set_data_with_da(CatsEye *this, CatsEye_layer *l, int n, int b)
{
	_CatsEye_DataAugmentation(l->x +l->inputs*b, this->learning_data+n*this->slide, l->sx, l->sy, l->ich, this->da);
	memcpy((int8_t*)this->label +this->label_size*b, (int8_t*)this->label_data +this->label_size*n, this->label_size/*bytes*/);
}
#if 0
void _CatsEye_set_data(CatsEye *this, CatsEye_layer *l, void *data, void *label, void *x, void *t)
{
	memcpy(data, x, l->inputs*sizeof(real));
	memcpy(label, t, this->label_size/*bytes*/);
}
void _CatsEye_set_data_with_da(CatsEye *this, CatsEye_layer *l, void *data, void *label, void *x, void *t)
{
	_CatsEye_DataAugmentation(data, x, l->sx, l->sy, l->ich, this->da);
	memcpy(label, t, this->label_size/*bytes*/);
}
#endif
static inline void _CatsEye_forward(CatsEye *this)
{
	CatsEye_layer *l = &this->layer[this->start];

	if (!this->set_data) {
		if (this->da) {
			this->set_data = _CatsEye_set_data_with_da;
		} else {
			this->set_data = _CatsEye_set_data;
		}
	}

	// create data for every batch
	int lsize = this->label_size;
	int8_t *label = (int8_t*)this->label;
	int8_t *label_data = (int8_t*)this->label_data;
	for (int b=0; b<this->batch; b++) {
#if 0
		int sample = this->shuffle_buffer[this->shuffle_base];
		if (this->da) {
			_CatsEye_DataAugmentation(l->x +l->inputs*b, this->learning_data+sample*this->slide, l->sx, l->sy, l->ich, this->da);
			memcpy(label +lsize*b, label_data+lsize*sample, lsize/*bytes*/);
		} else {
			memcpy(l->x +l->inputs*b, this->learning_data+sample*this->slide, l->inputs*sizeof(real));
			memcpy(label +lsize*b, label_data+lsize*sample, lsize/*bytes*/);
		}
//		this->set_data(this, l, l->x +l->inputs*b, label +lsize*b, this->learning_data+sample*this->slide, label_data+lsize*sample);
#endif
		this->set_data(this, l, this->shuffle_buffer[this->shuffle_base], b);

		this->shuffle_base++;
		if (this->shuffle_base>=this->data_num) this->shuffle_base = 0;
	}

#ifndef NAME
#define NAME
#endif
#ifdef CATS_CHECK
	static int flag = 10;
	if (flag) {
		void CatsEye_visualize(real *o, int n, int sx, uint8_t *p, int width, int ch);
		uint8_t *pixels = calloc(3, 96*96*3*100);
		int n = this->batch<50 ? this->batch : 50;
		for (int i=0; i<n; i++) {
			CatsEye_visualize(l->x+l->inputs*i, l->sx*l->sy, l->sx, &pixels[(((i+50)/10)*96*96*10+((i+50)%10)*96)*3], 96*10, 3);
			CatsEye_visualize((real*)label+lsize*i, 96*96, 96, &pixels[((i/10)*96*96*10+(i%10)*96)*3], 96*10, 3);
		}
		char name[50];
		snprintf(name, 50, "/tmp/"NAME"_in%03d.png", flag);
		stbi_write_png(name, 96*10, 96*10, 3, pixels, 0);
		free(pixels);
		--flag;
	}
#endif

	for (int i=this->start; i<=this->end; i++) {
#ifdef CATS_CHECK
		struct timespec start, stop;
		clock_gettime(CLOCK_REALTIME, &start);
#endif
		l->forward(l);
#ifdef CATS_CHECK
		clock_gettime(CLOCK_REALTIME, &stop);
		l->forward_time = (l->forward_time + time_diff(&start, &stop)) * 0.5;

		int count[20];
		memset(count, 0, 20*sizeof(int));
		for (int n=0; n<l->outputs; n++) {
			int a = (int)(l->z[n]*10 +10);
			a = a<0 ? 0 : a>20 ? 19 : a;
			count[a]++;
		}
		int max = 0;
		max = count[0];
		for (int n=1; n<20; n++) if (max < count[n]) max = count[n];
		for (int n=0; n<20; n++) count[n] = (int)(count[n]*19.0 / max);
		char name[50];
		snprintf(name, 50, "/tmp/"NAME"%03d.png", i+1);
		uint8_t z[20*20];
		memset(z, 0, 20*20);
		for (int x=0; x<20; x++) {
			for (int y=0; y<count[x]; y++) z[x+(19-y)*20] = 255;
		}
		stbi_write_png(name, 20, 20, 1, z, 0);
#endif
		l++;
	}
}
static inline void _CatsEye_backward(CatsEye *this)
{
//	memset(this->wdata +this->wsize, 0, this->wsize*sizeof(real)); // clear all l->dw
	CatsEye_layer *l = &this->layer[this->end];
	for (int i=this->end; i>=this->start; i--) {
#ifdef CATS_CHECK
		struct timespec start, stop;
		clock_gettime(CLOCK_REALTIME, &start);
#endif
		if (/*!(l->fix&2)*/i>this->stop) l->backward(l);
		if (!(l->fix&1)) l->update(l);
//		if (l->wsize) SOLVER(l);
#ifdef CATS_CHECK
		clock_gettime(CLOCK_REALTIME, &stop);
		l->backward_time = (l->backward_time + time_diff(&start, &stop)) * 0.5;
#endif
		l--;
	}
}
static inline void _CatsEye_update(CatsEye *this)
{
	CatsEye_layer *l = &this->layer[this->end];
	for (int i=this->end; i>=this->start; i--) {
		if (l->wsize) SOLVER(l);
		l--;
	}
}
static inline void _CatsEye_zero_grad(CatsEye *this)
{
	//memset(this->ddata, 0, this->dsize*this->batch*sizeof(real)); // clear all l->d (zero_grad)
	memset(this->wdata +this->wsize, 0, this->wsize*sizeof(real)); // clear all l->dw
}
int CatsEye_train(CatsEye *this, real *x, void *t, int N, int epoch, int samples, int verify)
{
	int a = this->end;	// layers-1
	this->layer[a].z = t;	// FIXME

	if (verify) N -= verify;	// for test
	int repeat = N;			// for random
	if (samples) repeat = samples;	// samples per epoch (0: all)
	if (this->batch>1) {
		repeat /= this->batch;
		if (repeat<1) repeat = 1;
		CATS_LOG(" batch: %d, repeat: %d\n", this->batch, repeat);
	}
	if (N<=0) {
		fprintf(stderr, "sample is strange!! %d\n", N);
		return 0;
	}
	CATS_LOG("epoch    loss     elapsed time\n");

	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	for (int times=0; times<epoch; times++) {
		_CatsEye_data_transfer(this, x, t, N);
		for (int n=0; n<repeat; n++) {
			_CatsEye_forward(this);
			_CatsEye_zero_grad(this);
			_CatsEye_backward(this);
			_CatsEye_update(this);
		}

		real err = 0;
		{
			// calculate the mean squared error
			real mse = 0;
			CatsEye_layer *l = &this->layer[a];
			for (int i=0; i<this->batch * l->inputs; i++) {
				mse += l->dIn[i] * l->dIn[i];
			}
			mse /= this->batch;
			err = 0.5 * (err + mse);
		}
/*		CatsEye_layer *l = &this->layer[a];
		for (int i=0; i<this->batch * l->inputs; i++) err += l->dIn[i] * l->dIn[i];
		err /= (this->batch * l->inputs);*/

		clock_gettime(CLOCK_REALTIME, &stop);
		CATS_LOG("%7d, %f %f [%.2fs]", times, this->loss, err, time_diff(&start, &stop));

		if (verify) {
			int r = CatsEye_accuracy(this, x+this->layer[0].inputs*N, (int16_t*)t+N, verify);
			CATS_LOG(" %.1f%%", (float)r/verify*100.0);
		}
		CATS_LOG("\n");
#ifdef CATS_CHECK
		FILE *fp = fopen("/tmp/"NAME"_time.txt", "w");
		for (int i=this->start; i<=this->end; i++) {
			CatsEye_layer *l = &this->layer[i];
			fprintf(fp, "#%d %.8f ms F\n", i+1, l->forward_time*1000);
			fprintf(fp, "#%d %.8f ms B\n", i+1, l->backward_time*1000);
		}
		fclose(fp);
#endif
		if (isnan(err) || isinf(err)) { // Ofast or O3
			CATS_LOG("\nnan or inf error!\n");
			CatsEye_layer *l = &this->layer[this->start];
			for (int i=this->start; i<=this->end; i++) {
				for (int n=0; n<this->ws[i]; n++) {
					if (isnan(l->w[n]) || isinf(l->w[n])) {
						CATS_LOG(" layer[%d] weight[%d]:%f\n", i, n, l->w[n]);
						break;
					}
				}
				for (int n=0; n<l->outputs; n++) {
					if (isnan(l->z[n]) || isinf(l->z[n])) {
						CATS_LOG(" layer[%d] z[%d]:%f\n", i, n, l->z[n]);
						break;
					}
				}
				for (int n=0; n<l->inputs; n++) {
					if (isnan(l->dIn[n]) || isinf(l->dIn[n])) {
						CATS_LOG(" layer[%d] d[%d]\n", i, n);
						break;
					}
				}
				l++;
			}
			return 0;
		}
	}
/*	if (this->shuffle_buffer) {
		free(this->shuffle_buffer);
		this->shuffle_buffer = 0;
	}*/
	return 1;
}

int CatsEye_saveCats(CatsEye *this, const char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp==NULL) return -1;

	CATS_LOG("Saving #%d weight... %lu bytes\n", this->epoch, (unsigned long)(this->wsize*sizeof(real)));
	int ok = fwrite(this->wdata, this->wsize*sizeof(real), 1, fp)==1 &&
		fwrite(&this->epoch, sizeof(int), 1, fp)==1;

	if (fclose(fp)!=0) ok = 0;
	return ok ? 0 : -1;
}
int CatsEye_loadCats(CatsEye *this, const char *filename)
{
	FILE *fp = fopen(filename, "rb");
	if (fp==NULL) return -1;

	if (fread(this->wdata, this->wsize*sizeof(real), 1, fp)!=1 || fread(&this->epoch, sizeof(int), 1, fp)!=1) {
		fclose(fp);
		return -2;	// file is too short: the network does not match
	}
	CATS_LOG("Loading #%d weight... %lu bytes\n", this->epoch++, (unsigned long)(this->wsize*sizeof(real)));

	fclose(fp);
	return 0;
}

// save weights to json file
int CatsEye_saveJson(CatsEye *this, const char *filename)
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
			for (int i=0; i<this->ws[n]; i++) {
				fprintf(fp, i ? ",%lf" : "%lf", (double)this->w[n][i]);
			}
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
/*#define DEF_OR_ARG(value,...) value
#define DEF_OR_ARG2(v1, v2, ...) v1,v2
#define CatsEye_visualize(o, n, sx, p, width, ch, ...) _CatsEye_visualize(o, n, sx, p, width, ch, DEF_OR_ARG2(__VA_ARGS__ __VA_OPT__(,) 0,0))*/
void CatsEye_visualize(real *o, int n, int sx, uint8_t *p, int width, int ch)
{
	real max = o[0];
	real min = o[0];
	for (int i=1; i<n*ch; i++) {
		if (max < o[i]) max = o[i];
		if (min > o[i]) min = o[i];
	}
	real range = max - min;
	if (range==0) range = 1;	// flat input: avoid 0/0
	for (int c=0; c<ch; c++) {
		for (int i=0; i<n; i++) {
			p[((i/sx)*width + i%sx)*ch +c] = (uint8_t)((o[i+c*n] - min) / range * 255.0);
		}
	}
}
void _CatsEye_visualize(real *o, int n, int sx, uint8_t *p, int width, int ch, real min, real max)
{
	if (min!=max) {
		/*real*/ max = o[0];
		/*real*/ min = o[0];
		for (int i=1; i<n*ch; i++) {
			if (max < o[i]) max = o[i];
			if (min > o[i]) min = o[i];
		}
	}
	real range = max - min;
	if (range==0) range = 1;
	for (int c=0; c<ch; c++) {
		for (int i=0; i<n; i++) {
			p[((i/sx)*width + i%sx)*ch +c] = (uint8_t)((o[i+c*n] - min) / range * 255.0);
		}
	}
}
void CatsEye_visualizeYUV(real *s, int n, int sx, uint8_t *p, int width)
{
	for (int i=0; i<n; i++) {
		int r = (int)(s[i]*255.0                     +1.402  *s[i+n*2]*255);
		int g = (int)(s[i]*255.0 -0.34414*s[i+n]*255 -0.71414*s[i+n*2]*255);
		int b = (int)(s[i]*255.0 +1.772  *s[i+n]*255);
		p[((i/sx)*width + i%sx)*3 +0] = (uint8_t)(r>255 ? 255 : r<0 ? 0 : r);
		p[((i/sx)*width + i%sx)*3 +1] = (uint8_t)(g>255 ? 255 : g<0 ? 0 : g);
		p[((i/sx)*width + i%sx)*3 +2] = (uint8_t)(b>255 ? 255 : b<0 ? 0 : b);
	}
}

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

// https://www.cs.toronto.edu/~kriz/cifar.html
real *CatsEye_loadCifar(const char *name, int size, int lsize, int sample, int16_t **label)
{
	FILE *fp = fopen(name, "rb");
	if (!fp) { fprintf(stderr, "Can't open %s\n", name); return 0; }
	uint8_t *data = malloc((size_t)(size+lsize)*sample);	// +lsize for label
	int16_t *t = malloc(sizeof(int16_t)*sample);
	real *x = malloc(sizeof(real)*size*sample);
	if (!data || !t || !x || fread(data, (size_t)(size+lsize)*sample, 1, fp)!=1) {
		fprintf(stderr, "Can't read %s\n", name);
		fclose(fp);
		free(data); free(t); free(x);
		return 0;
	}
	fclose(fp);
	for (int n=0; n<sample; n++) {
		if (lsize==2) {
			int16_t p;
			memcpy(&p, &data[n*(size+2)], sizeof(p));
			t[n] = p;
		} else {
			t[n] = data[n*(size+lsize)];
		}
		for (int i=0; i<size; i++) x[n*size+i] = data[n*(size+lsize)+lsize+i] * (1.0/255.0);
	}
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

real *CatsEye_loadMnist(const char *name, const char *name2, int sample, int **label)
{
	int size = 784;
	uint8_t *data = malloc((size_t)(size+1)*sample + 16);		// +1 for label, 16 for the header
	int *t = malloc(sizeof(int)*sample);
	real *x = calloc((size_t)(size+1)*(sample+1), sizeof(real));	// +1 for bias
	FILE *fp = fopen(name, "rb");
	FILE *fp2 = fopen(name2, "rb");
	if (!data || !t || !x || !fp || !fp2 ||
		fread(data, 16, 1, fp)!=1 || fread(data, size, sample, fp)!=(size_t)sample ||	// header, data
		fread(data +(size_t)size*sample, 8, 1, fp2)!=1 || fread(data +(size_t)size*sample, 1, sample, fp2)!=(size_t)sample) {	// header, labels
		fprintf(stderr, "Can't read %s / %s\n", name, name2);
		if (fp) fclose(fp);
		if (fp2) fclose(fp2);
		free(data); free(t); free(x);
		return 0;
	}
	fclose(fp);
	fclose(fp2);
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	for (int i=0; i<sample; i++) t[i] = data[(size_t)size*sample +i];
	free(data);

	*label = t;
	return x;
}

#undef SOLVER

#endif /* !CATS_INFER_ONLY */

// ---------------------------------------------------------------------------
// Inference primitives: CPU feature detection, SIMD dot / GEMM, fp16 / bf16, norms and
// activations. Float-only and independent of the training macros (`real`, sqrt -> sqrtf ...).
// CATS_INFER_ONLY (define before including) skips the training code above.
// ---------------------------------------------------------------------------
#include <stdatomic.h>
#include <limits.h>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define CATS_SIMD_X86 1
#include <immintrin.h>
#define CATS_AVX_FN __attribute__((target("avx")))
#define CATS_AVX2_FN __attribute__((target("avx2,fma")))
#define CATS_AVX512_FN __attribute__((target("avx512f,avx2,fma")))
#else
#define CATS_SIMD_X86 0
#endif

/* Environment override: CATS_SIMD=scalar|avx|avx2|avx512|auto.
   Unsupported requests safely use the highest supported lower level. */
static inline int cats_cpu_simd_level(void)
{
	static atomic_int cached=-1;
	int old=atomic_load_explicit(&cached,memory_order_relaxed);
	if (old>=0) {
		return old;
	}
	int maxlevel=0;
#if CATS_SIMD_X86
	__builtin_cpu_init();
	if (__builtin_cpu_supports("avx")) {
		maxlevel=1;
	}
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		maxlevel=2;
	}
	if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
		maxlevel=3;
	}
#endif
	const char*env=getenv("CATS_SIMD");
	if (env && (!strcmp(env,"scalar")||!strcmp(env,"off")||!strcmp(env,"0"))) {
		maxlevel=0;
	} else if (env && !strcmp(env,"avx") && maxlevel>1) {
		maxlevel=1;
	} else if (env && !strcmp(env,"avx2") && maxlevel>2) {
		maxlevel=2;
	}
	/* A requested ISA is never used if the current CPU lacks it. */
	atomic_store_explicit(&cached,maxlevel,memory_order_relaxed);
	return maxlevel;
}
static inline const char*cats_cpu_simd_name(void)
{
	int x=cats_cpu_simd_level();
	return x==3?"AVX-512F/FMA":x==2?"AVX2/FMA":x==1?"AVX":"scalar";
}


#ifdef CATS_GPU
#include "cats_gpu.h"   // optional GPU GEMM; the CPU kernels remain the fallback
#endif
/* Stage 31: dependency-free SIMD matrix-vector kernel for audio tower.
 * Runtime dispatch honors existing CATS_SIMD override. FP32 FMA may
 * differ slightly from the former scalar FP64 accumulation path.
 * No ISA-specific instructions are emitted on non-x86 targets. */
#if CATS_SIMD_X86
CATS_AVX2_FN static float cats_dot_fma(const float*x,const float*w,size_t n)
{
	__m256 a=_mm256_setzero_ps(),b=_mm256_setzero_ps(),
	                             c=_mm256_setzero_ps(),d=_mm256_setzero_ps();
	size_t i=0;
	for (; i+32<=n; i+=32) {
		a=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(w+i),a);
		b=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_loadu_ps(w+i+8),b);
		c=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_loadu_ps(w+i+16),c);
		d=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_loadu_ps(w+i+24),d);
	}
	a=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
	float parts[8];
	_mm256_storeu_ps(parts,a);
	double sum=0;
	for (int j=0; j<8; j++) {
		sum+=parts[j];
	}
	for (; i<n; i++) {
		sum+=(double)x[i]*w[i];
	}
	return (float)sum;
}
#endif
static inline float cats_dot_f32(const float*x,const float*w,size_t n)
{
#if CATS_SIMD_X86
	if (n>=32&&cats_cpu_simd_level()>=2) {
		return cats_dot_fma(x,w,n);
	}
#endif
	double sum=0;
	for (size_t j=0; j<n; j++) {
		sum+=(double)x[j]*w[j];
	}
	return (float)sum;
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
#ifndef CATS_GEMM_OB
#define CATS_GEMM_OB 16
#endif
#ifndef CATS_GEMM_TB
#define CATS_GEMM_TB 256
#endif
#ifndef CATS_GEMM_KC
#define CATS_GEMM_KC 512
#endif
typedef void (*cats_tile_fn)(float*,size_t,const float*,size_t,const float*,
                               size_t,size_t,size_t,size_t,int,const float*);
static void cats_tile_generic(float*c,size_t ldc,const float*x,size_t ldx,
                                const float*w,size_t ldw,size_t klen,size_t nt,size_t no,int acc,const float*bias)
{
	for (size_t t=0; t<nt; t++)for (size_t o=0; o<no; o++) {
			double sum=0;
			for (size_t k=0; k<klen; k++) {
				sum+=(double)x[t*ldx+k]*w[o*ldw+k];
			}
			float r=(float)sum;
			if (acc) {
				r+=c[t*ldc+o];
			} else if (bias) {
				r+=bias[o];
			}
			c[t*ldc+o]=r;
		}
}
#if CATS_SIMD_X86
CATS_AVX_FN static inline float cats_hsum8(__m256 v)
{
	__m128 s=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
	s=_mm_hadd_ps(s,s);
	s=_mm_hadd_ps(s,s);
	return _mm_cvtss_f32(s);
}
#define CATS_MAC_AVX(a,b,c) _mm256_add_ps((c),_mm256_mul_ps((a),(b)))
#define CATS_MAC_FMA(a,b,c) _mm256_fmadd_ps((a),(b),(c))
#define CATS_DEF_TILE(NAME,ATTR,MAC) \
ATTR static void NAME(float*c,size_t ldc,const float*x,size_t ldx,const float*w, \
        size_t ldw,size_t klen,size_t nt,size_t no,int acc,const float*bias){ \
    if(nt!=4||no!=2){cats_tile_generic(c,ldc,x,ldx,w,ldw,klen,nt,no,acc,bias);return;} \
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
    float r[4][2]={{cats_hsum8(a00),cats_hsum8(a01)},{cats_hsum8(a10),cats_hsum8(a11)}, \
                   {cats_hsum8(a20),cats_hsum8(a21)},{cats_hsum8(a30),cats_hsum8(a31)}}; \
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
CATS_DEF_TILE(cats_tile_avx,CATS_AVX_FN,CATS_MAC_AVX)
CATS_DEF_TILE(cats_tile_avx2,CATS_AVX2_FN,CATS_MAC_FMA)
#endif
static inline void cats_gemm_nt_cpu(float*c,const float*x,const float*w,size_t n,
                                      size_t in,size_t out,const float*bias);
static inline void cats_gemm_nt(float*c,const float*x,const float*w,size_t n,
                                  size_t in,size_t out,const float*bias)
{
	if (!n||!in||!out) {
		return;
	}
#ifdef CATS_GPU
	/* Weights are cached on the GPU by pointer: whoever frees or rewrites W
	 * must call cs_gpu_forget_weights() (CS_GPU_FORGET) first.  The rows are
	 * shared out: the GPU takes whole 128-row tiles (about CS_GPU_SHARE percent
	 * of them), the CPU computes the remaining rows while the GPU runs. */
	if (cs_gpu_ready()&&n>=32&&in%16==0&&(double)n*in*out>=2.0e7) {
		size_t ng=((n*cs_gpu_share()/100+CS_GPU_TILE-1)/CS_GPU_TILE)*CS_GPU_TILE;
		if (ng>n||n-ng<16) {
			ng=n;
		}
		if (!cs_gpu_gemm_begin(x,w,ng,in,out,bias)) {
			if (ng<n) {
				cats_gemm_nt_cpu(c+ng*out,x+ng*in,w,n-ng,in,out,bias);
			}
			if (!cs_gpu_gemm_end(c)) {
				return;
			}
			cats_gemm_nt_cpu(c,x,w,ng,in,out,bias);   /* GPU failed: redo its rows */
			return;
		}
	}
#endif
	cats_gemm_nt_cpu(c,x,w,n,in,out,bias);
}
static inline void cats_gemm_nt_cpu(float*c,const float*x,const float*w,size_t n,
                                      size_t in,size_t out,const float*bias)
{
	if (!n||!in||!out) {
		return;
	}
	cats_tile_fn tile=cats_tile_generic;
#if CATS_SIMD_X86
	{
		int level=cats_cpu_simd_level();
		if (level>=2) {
			tile=cats_tile_avx2;
		} else if (level==1) {
			tile=cats_tile_avx;
		}
	}
#endif
	const long ntb=(long)((n+CATS_GEMM_TB-1)/CATS_GEMM_TB);
	const long nob=(long)((out+CATS_GEMM_OB-1)/CATS_GEMM_OB);
	const int big=(double)n*(double)in*(double)out>1.0e6;
	(void)big;
	#pragma omp parallel for collapse(2) schedule(dynamic,1) if(big)
	for (long tb=0; tb<ntb; tb++)for (long ob=0; ob<nob; ob++) {
			size_t t0=(size_t)tb*CATS_GEMM_TB,t1=t0+CATS_GEMM_TB;
			if (t1>n) {
				t1=n;
			}
			size_t o0=(size_t)ob*CATS_GEMM_OB,o1=o0+CATS_GEMM_OB;
			if (o1>out) {
				o1=out;
			}
			for (size_t k0=0; k0<in; k0+=CATS_GEMM_KC) {
				size_t klen=in-k0<CATS_GEMM_KC?in-k0:CATS_GEMM_KC;
				int acc=k0!=0;
				for (size_t t=t0; t<t1; t+=4) {
					size_t nt=t1-t<4?t1-t:4;
					for (size_t o=o0; o<o1; o+=2) {
						size_t no=o1-o<2?o1-o:2;
						tile(c+t*out+o,out,x+t*in+k0,in,w+o*in+k0,in,klen,nt,no,acc,
						     bias?bias+o:NULL);
					}
				}
			}
		}
}
static inline float cats_half_to_float(unsigned h)
{
#ifdef __F16C__
	return _cvtsh_ss((unsigned short)h);
#else
	uint32_t sign=(uint32_t)(h&0x8000u)<<16,exp=(h>>10)&0x1fu,man=h&0x3ffu,bits;
	if (!exp) {
		if (!man) {
			bits=sign;
		} else {
			int e=-1;
			do {
				e++;
				man<<=1;
			} while (!(man&0x400u));
			man&=0x3ffu;
			bits=sign|((uint32_t)(112-e)<<23)|(man<<13);
		}
	} else if (exp==31) {
		bits=sign|0x7f800000u|(man<<13);
	} else {
		bits=sign|((exp+112u)<<23)|(man<<13);
	}
	float f;
	memcpy(&f,&bits,4);
	return f;
#endif
}

// bfloat16 is the upper half of an IEEE float
static inline float cats_bf16_to_float(unsigned h)
{
	uint32_t bits = (uint32_t)(h & 0xffffu) << 16;
	float f;
	memcpy(&f, &bits, 4);
	return f;
}
static inline float cats_silu(float x)
{
	return x/(1.0f+expf(-x));
}
static inline float cats_softplus(float x)
{
	return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x)));
}
static inline float cats_clampf(float x,float low,float high)
{
	return fminf(high,fmaxf(low,x));
}
static inline void cats_rmsnorm(float*dst,const float*src,const float*weight,unsigned d)
{
	double sum=0;
	for (unsigned i=0; i<d; i++) {
		sum+=(double)src[i]*src[i];
	}
	float scale=1.0f/sqrtf((float)(sum/d)+1.e-6f);
	for (unsigned i=0; i<d; i++) {
		dst[i]=src[i]*scale*weight[i];
	}
}

// tanh approximation of GELU (gelu_pytorch_tanh)
static inline float cats_gelu_tanh(float x)
{
	return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
// Softmax of row[lo..hi) after multiplying by `scale`; everything outside [lo, hi) of the n entries becomes 0.
// The AVX path evaluates exp 8 lanes at a time (Cephes polynomial, relative error ~1e-7, comparable to expf).
#if CATS_SIMD_X86
CATS_AVX_FN static inline __m256 cats_exp_ps_avx(__m256 x)
{
	x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
	__m256 fx = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _mm256_set1_ps(0.5f)));
	__m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
	r = _mm256_sub_ps(r, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
	__m256 y = _mm256_set1_ps(1.9875691500e-4f);
	y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(1.3981999507e-3f));
	y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(8.3334519073e-3f));
	y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(4.1665795894e-2f));
	y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(1.6666665459e-1f));
	y = _mm256_add_ps(_mm256_mul_ps(y, r), _mm256_set1_ps(5.0000001201e-1f));
	y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y, _mm256_mul_ps(r, r)), r), _mm256_set1_ps(1.0f));
	// 2^n built with 128-bit integer ops (AVX has no 256-bit integer shifts)
	__m256i n = _mm256_cvtps_epi32(fx);
	__m128i lo = _mm_slli_epi32(_mm_add_epi32(_mm256_castsi256_si128(n), _mm_set1_epi32(127)), 23);
	__m128i hi = _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256(n, 1), _mm_set1_epi32(127)), 23);
	__m256 pow2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(lo)), _mm_castsi128_ps(hi), 1);
	return _mm256_mul_ps(y, pow2);
}
CATS_AVX_FN static void cats_softmax_scaled_avx(float *row, size_t lo, size_t hi, float scale)
{
	const __m256 sc = _mm256_set1_ps(scale);
	size_t j = lo;
	__m256 vmax = _mm256_set1_ps(-FLT_MAX);
	for (; j + 8 <= hi; j += 8) {
		__m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + j), sc);
		_mm256_storeu_ps(row + j, v);
		vmax = _mm256_max_ps(vmax, v);
	}
	float tmp[8], mx = -FLT_MAX;
	_mm256_storeu_ps(tmp, vmax);
	for (int i = 0; i < 8; i++) if (tmp[i] > mx) mx = tmp[i];
	for (; j < hi; j++) { row[j] *= scale; if (row[j] > mx) mx = row[j]; }
	const __m256 vm = _mm256_set1_ps(mx);
	__m256 vsum = _mm256_setzero_ps();
	for (j = lo; j + 8 <= hi; j += 8) {
		__m256 e = cats_exp_ps_avx(_mm256_sub_ps(_mm256_loadu_ps(row + j), vm));
		_mm256_storeu_ps(row + j, e);
		vsum = _mm256_add_ps(vsum, e);
	}
	_mm256_storeu_ps(tmp, vsum);
	double sum = 0;
	for (int i = 0; i < 8; i++) sum += tmp[i];
	for (; j < hi; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
	const __m256 inv = _mm256_set1_ps((float)(1.0 / sum));
	for (j = lo; j + 8 <= hi; j += 8) _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
	for (; j < hi; j++) row[j] *= (float)(1.0 / sum);
}
#endif
static inline void cats_softmax_scaled(float *row, size_t n, size_t lo, size_t hi, float scale)
{
	if (lo >= hi) { memset(row, 0, n * sizeof(float)); return; }
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) {
		cats_softmax_scaled_avx(row, lo, hi, scale);
		memset(row, 0, lo * sizeof(float));
		memset(row + hi, 0, (n - hi) * sizeof(float));
		return;
	}
#endif
	float mx = -FLT_MAX;
	for (size_t j = lo; j < hi; j++) { row[j] *= scale; if (row[j] > mx) mx = row[j]; }
	double sum = 0;
	for (size_t j = lo; j < hi; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
	const float inv = (float)(1.0 / sum);
	memset(row, 0, lo * sizeof(float));
	for (size_t j = lo; j < hi; j++) row[j] *= inv;
	memset(row + hi, 0, (n - hi) * sizeof(float));
}
// in-place softmax over n values
static inline void cats_softmax_inplace(float *x, size_t n)
{
	cats_softmax_scaled(x, n, 0, n, 1.0f);
}

#ifdef _OPENMP
#include <omp.h>
#endif
// Threads to use when the user gave neither -j nor OMP_NUM_THREADS: one per physical core.
// Hyper-threads share the FP units, so they add no GEMM throughput and only add barrier cost.
// Returns 0 = leave the OpenMP default.
static inline int cats_physical_threads(void)
{
#ifdef _OPENMP
	FILE*f=fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list","r");
	if (!f) {
		return 0;
	}
	char line[256];
	int siblings=0;
	if (fgets(line,sizeof(line),f)) {
		for (const char*p=line; *p&&*p!='\n';) {
			char*end;
			long a=strtol(p,&end,10);
			if (end==p) {
				break;
			}
			long b=a;
			if (*end=='-') {
				p=end+1;
				b=strtol(p,&end,10);
			}
			siblings+=(int)(b-a+1);
			p=*end==','?end+1:end;
			if (*end!=','&&*end!='-') {
				break;
			}
		}
	}
	fclose(f);
	int procs=omp_get_num_procs();
	return siblings>1&&procs>=siblings?procs/siblings:0;
#else
	return 0;
#endif
}
// Applies `requested` threads (> 0), else one per physical core unless OMP_NUM_THREADS is set.
static inline void cats_set_threads(int requested)
{
#ifdef _OPENMP
	if (requested > 0) omp_set_num_threads(requested);
	else if (!getenv("OMP_NUM_THREADS") && cats_physical_threads() > 0) omp_set_num_threads(cats_physical_threads());
#else
	(void)requested;
#endif
}

// ---------------------------------------------------------------------------
// safetensors reader (mmap). Tensors are looked up by name and widened to float32 on demand,
// so bf16 / fp16 checkpoints from Hugging Face can be used without conversion.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Q4_0: 4-bit weights in blocks of 32 along the input dimension (the GGML layout, 18 bytes per block):
//   fp16 scale d, then 16 bytes where byte j holds element j (low nibble) and j+16 (high nibble).
//   value = (nibble - 8) * d.  A row of n weights (n % 32 == 0) takes n / 32 * 18 bytes.
// ---------------------------------------------------------------------------
#define CATS_Q4_0_BLOCK	32
#define CATS_Q4_0_BYTES	18
static inline size_t cats_q4_0_row_bytes(size_t n) { return n / CATS_Q4_0_BLOCK * CATS_Q4_0_BYTES; }

// float -> IEEE half, round to nearest even (subnormals and overflow handled)
static inline uint16_t cats_float_to_half(float f)
{
#ifdef __F16C__
	return (uint16_t)_cvtss_sh(f, 0);
#else
	uint32_t x;
	memcpy(&x, &f, 4);
	uint32_t sign = (x >> 16) & 0x8000u, e = (x >> 23) & 0xffu, m = x & 0x7fffffu;
	if (e == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0));
	int ne = (int)e - 127 + 15;
	if (ne >= 31) return (uint16_t)(sign | 0x7c00u);
	if (ne <= 0) {
		if (ne < -10) return (uint16_t)sign;
		m |= 0x800000u;
		int shift = 14 - ne;
		uint32_t half = m >> shift, rem = m & ((1u << shift) - 1), mid = 1u << (shift - 1);
		if (rem > mid || (rem == mid && (half & 1))) half++;
		return (uint16_t)(sign | half);
	}
	uint32_t half = ((uint32_t)ne << 10) | (m >> 13), rem = m & 0x1fffu;
	if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;	// a carry into the exponent is correct
	return (uint16_t)(sign | half);
#endif
}
static inline uint16_t cats_load_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// Quantizes n floats (n % 32 == 0) into n/32*18 bytes. The scale is chosen as max/-8 on the element of
// largest magnitude (so that element is exact), then refined by a small search that minimises the
// block's squared error; refine = 0 gives plain GGML Q4_0.
static inline void cats_q4_0_quantize_row(const float *x, size_t n, uint8_t *out, int refine)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, x += CATS_Q4_0_BLOCK, out += CATS_Q4_0_BYTES) {
		float amax = 0, mx = 0;
		for (int j = 0; j < CATS_Q4_0_BLOCK; j++) if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); mx = x[j]; }
		float best_d = mx / -8.0f, best_err = FLT_MAX;
		const int tries = refine ? 17 : 1;
		const float base = best_d;
		for (int t = 0; t < tries; t++) {
			float d = base * (1.0f + 0.0125f * (t - (tries - 1) / 2));	// +-10 % around max/-8
			uint16_t dh = cats_float_to_half(d);
			float dq = cats_half_to_float(dh), id = dq ? 1.0f / dq : 0.0f, err = 0;
			for (int j = 0; j < CATS_Q4_0_BLOCK; j++) {
				int q = (int)(x[j] * id + 8.5f);
				q = q < 0 ? 0 : q > 15 ? 15 : q;
				float e = x[j] - (q - 8) * dq;
				err += e * e;
			}
			if (err < best_err) { best_err = err; best_d = d; }
			if (!refine) break;
		}
		uint16_t dh = cats_float_to_half(best_d);
		float dq = cats_half_to_float(dh), id = dq ? 1.0f / dq : 0.0f;
		out[0] = (uint8_t)(dh & 0xff); out[1] = (uint8_t)(dh >> 8);
		for (int j = 0; j < 16; j++) {
			int q0 = (int)(x[j] * id + 8.5f), q1 = (int)(x[j + 16] * id + 8.5f);
			q0 = q0 < 0 ? 0 : q0 > 15 ? 15 : q0;
			q1 = q1 < 0 ? 0 : q1 > 15 ? 15 : q1;
			out[2 + j] = (uint8_t)(q0 | (q1 << 4));
		}
	}
}
static inline void cats_q4_0_dequantize_row(const uint8_t *in, size_t n, float *out)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, in += CATS_Q4_0_BYTES, out += CATS_Q4_0_BLOCK) {
		float d = cats_half_to_float(cats_load_u16(in));
		for (int j = 0; j < 16; j++) {
			out[j]      = ((in[2 + j] & 0x0f) - 8) * d;
			out[j + 16] = ((in[2 + j] >> 4) - 8) * d;
		}
	}
}
// ---- Q8_0 activations (36 bytes per 32 values: 32 int8, then the float scale) and the Q4_0 x Q8_0 dot ----
#define CATS_Q8_0_BYTES	36
static inline size_t cats_q8_0_row_bytes(size_t n) { return n / CATS_Q4_0_BLOCK * CATS_Q8_0_BYTES; }

#if CATS_SIMD_X86
// horizontally add 8 floats (hadd order)
CATS_AVX_FN static inline float cats_hsum_ps(__m256 acc)
{
	acc = _mm256_hadd_ps(acc, acc);
	acc = _mm256_hadd_ps(acc, acc);
	return _mm_cvtss_f32(_mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc)));
}
// sum of the 16 products of two int8 vectors, as 4 int32 lanes (x is used for its sign and magnitude)
CATS_AVX_FN static inline __m128i cats_mul_sum_i8_pairs(const __m128i x, const __m128i y)
{
	const __m128i ax = _mm_sign_epi8(x, x);
	const __m128i sy = _mm_sign_epi8(y, x);
	const __m128i dot = _mm_maddubs_epi16(ax, sy);
	return _mm_madd_epi16(_mm_set1_epi16(1), dot);
}
CATS_AVX_FN static float cats_dot_q4_0_q8_0_avx(const uint8_t *x, const int8_t *y, int n)
{
	const __m128i lowMask = _mm_set1_epi8(0xF), off = _mm_set1_epi8(8);
	int nb = n >> 5;
	__m256 acc = _mm256_setzero_ps();
	do {
		float delta = cats_half_to_float(cats_load_u16(x));
		const __m128i tmp = _mm_loadu_si128((const __m128i*)(x + 2));
		x += CATS_Q4_0_BYTES;
		__m128i bx = _mm_sub_epi8(_mm_and_si128(lowMask, tmp), off);
		const __m128i i32_0 = cats_mul_sum_i8_pairs(bx, _mm_loadu_si128((const __m128i*)y));
		bx = _mm_sub_epi8(_mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4)), off);
		const __m128i i32_1 = cats_mul_sum_i8_pairs(bx, _mm_loadu_si128((const __m128i*)(y + 16)));
		const __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));
		float ds;
		memcpy(&ds, y + 32, 4);
		y += CATS_Q8_0_BYTES;
		acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(delta * ds), p));
	} while (--nb > 0);
	return cats_hsum_ps(acc);
}
// Quantizes n floats (n % 32 == 0) to Q8_0 blocks: scale = max|x| / 127, round to nearest.
CATS_AVX_FN static void cats_q8_0_quantize_avx(int8_t *p, const float *x, int n)
{
	int nb = n >> 5;
	do {
		__m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
		x += 32;
		const __m256 signBit = _mm256_set1_ps(-0.0f);
		__m256 maxAbs = _mm256_andnot_ps(signBit, v0);
		maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
		maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
		maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));
		__m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
		max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
		max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
		const float maxScalar = _mm_cvtss_f32(max4);
		const float d = maxScalar / 127.0f;
		const float id = maxScalar != 0.0f ? 127.0f / maxScalar : 0.0f;
		const __m256 mul = _mm256_set1_ps(id);
		v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		__m256i i0 = _mm256_cvtps_epi32(v0), i1 = _mm256_cvtps_epi32(v1), i2 = _mm256_cvtps_epi32(v2), i3 = _mm256_cvtps_epi32(v3);
		__m128i ni0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extractf128_si256(i0, 1));
		__m128i ni2 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extractf128_si256(i1, 1));
		__m128i ni4 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extractf128_si256(i2, 1));
		__m128i ni6 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extractf128_si256(i3, 1));
		_mm_storeu_si128((__m128i*)(p), _mm_packs_epi16(ni0, ni2));
		_mm_storeu_si128((__m128i*)(p + 16), _mm_packs_epi16(ni4, ni6));
		memcpy(p + 32, &d, 4);
		p += CATS_Q8_0_BYTES;
	} while (--nb > 0);
}
#endif
static float cats_dot_q4_0_q8_0_scalar(const uint8_t *a, const int8_t *b, int n)
{
	float val = 0.0f;
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q4_0_BYTES, b += CATS_Q8_0_BYTES) {
		float delta = cats_half_to_float(cats_load_u16(a)), ds;
		int32_t ival = 0;
		for (int j = 0; j < 16; j++) {
			ival += (int32_t)((a[2 + j] & 0x0F) - 8) * b[j];
			ival += (int32_t)((a[2 + j] >> 4) - 8) * b[j + 16];
		}
		memcpy(&ds, b + 32, 4);
		val += ival * (delta * ds);
	}
	return val;
}
static void cats_q8_0_quantize_scalar(int8_t *q, const float *x, int n)
{
	for (int g = 0; g < n / CATS_Q4_0_BLOCK; g++, x += CATS_Q4_0_BLOCK, q += CATS_Q8_0_BYTES) {
		float amax = 0.0f;
		for (int i = 0; i < CATS_Q4_0_BLOCK; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
		const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
		for (int i = 0; i < CATS_Q4_0_BLOCK; i++) q[i] = (int8_t)roundf(x[i] * id);
		memcpy(q + 32, &d, 4);
	}
}
// Dispatching versions (AVX when the CPU has it, see cats_cpu_simd_level)
static inline float cats_dot_q4_0_q8_0(const uint8_t *w, const int8_t *a, int n)
{
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) return cats_dot_q4_0_q8_0_avx(w, a, n);
#endif
	return cats_dot_q4_0_q8_0_scalar(w, a, n);
}
static inline void cats_q8_0_quantize(int8_t *out, const float *x, int n)
{
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) { cats_q8_0_quantize_avx(out, x, n); return; }
#endif
	cats_q8_0_quantize_scalar(out, x, n);
}

// C[n][out] = X[n][in] . W^T (+ bias) with W stored as Q4_0 rows.
//   1..3 tokens: activations are quantized to Q8_0 and every output is an integer dot product, so the weights
//     stay 4-bit in memory (a vector-matrix product is bound by reading the weights);
//   4 or more tokens: the weights are expanded once into a scratch matrix (kept between calls, so this is not
//     thread-safe) and the blocked float GEMM runs. Measured on an AVX-only Xeon this is 1.5x faster at n = 4 and
//     about 4x at n = 100 and above; the threshold is a compile-time knob for machines with a faster int8 path.
#ifndef CATS_Q4_0_DEQUANT_MIN
#define CATS_Q4_0_DEQUANT_MIN	4
#endif
static inline void cats_gemm_nt_q4_0(float *c, const float *x, const uint8_t *w, size_t n, size_t in, size_t out,
				     const float *bias)
{
	if (!n || !in || !out || in % CATS_Q4_0_BLOCK) return;
	const size_t arow = cats_q8_0_row_bytes(in), wrow = cats_q4_0_row_bytes(in);
	if (n >= CATS_Q4_0_DEQUANT_MIN) {
		static float *scratch;
		static size_t cap;
		if (cap < in * out) {
			(free)(scratch);
			scratch = (float*)(malloc)(in * out * sizeof(float));
			cap = scratch ? in * out : 0;
			if (!scratch) return;
		}
		#pragma omp parallel for schedule(static)
		for (long r = 0; r < (long)out; r++) cats_q4_0_dequantize_row(w + (size_t)r * wrow, in, scratch + (size_t)r * in);
		cats_gemm_nt(c, x, scratch, n, in, out, bias);
		return;
	}
	int8_t *xq = (int8_t*)(malloc)(n * arow);
	if (!xq) return;
	#pragma omp parallel for schedule(static)
	for (long t = 0; t < (long)n; t++) cats_q8_0_quantize(xq + (size_t)t * arow, x + (size_t)t * in, (int)in);
	#pragma omp parallel for schedule(static)
	for (long r = 0; r < (long)out; r++) {
		const uint8_t *wr = w + (size_t)r * wrow;
		const float b = bias ? bias[r] : 0.0f;
		for (size_t t = 0; t < n; t++) c[t * out + r] = cats_dot_q4_0_q8_0(wr, xq + t * arow, (int)in) + b;
	}
	(free)(xq);
}

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#define CATS_ST_F32	0
#define CATS_ST_F16	1
#define CATS_ST_BF16	2
#define CATS_ST_Q4_0	3		// shape = logical [rows, cols]; rows are cats_q4_0 encoded
typedef struct {
	char name[192];
	int dtype;			// CATS_ST_*, -1 = unsupported
	int ndim;
	int64_t shape[4];
	uint64_t begin, end;		// byte range inside the data section
} cats_st_tensor;
typedef struct {
	uint8_t *map;
	size_t size;
	const uint8_t *data;		// start of the data section
	cats_st_tensor *tensor;
	int count;
} cats_st;

static inline void cats_st_close(cats_st *s)
{
#if !defined(_WIN32)
	if (s->map) munmap(s->map, s->size);
#endif
	(free)(s->tensor);
	memset(s, 0, sizeof(*s));
}
// Reads the JSON header: {"name":{"dtype":"BF16","shape":[..],"data_offsets":[a,b]}, ...}
static inline int cats_st_parse(cats_st *s, const char *j, size_t n)
{
	size_t cap = 64;
	s->tensor = (cats_st_tensor*)(malloc)(cap * sizeof(cats_st_tensor));
	if (!s->tensor) return -1;
	size_t i = 0;
	while (i < n && j[i] != '{') i++;
	i++;
	while (i < n) {
		while (i < n && j[i] != '"' && j[i] != '}') i++;
		if (i >= n || j[i] == '}') break;
		size_t k0 = ++i;
		while (i < n && j[i] != '"') i++;
		size_t klen = i - k0;
		i++;
		while (i < n && j[i] != '{') i++;
		if (i >= n) return -1;
		size_t o0 = i;
		int depth = 0;
		do {	// matching brace of this entry
			if (j[i] == '{') depth++;
			else if (j[i] == '}') depth--;
			i++;
		} while (i < n && depth > 0);
		if (klen == 12 && !strncmp(j + k0, "__metadata__", 12)) continue;
		if (klen >= sizeof(((cats_st_tensor*)0)->name)) return -1;
		if (s->count == (int)cap) {
			cap *= 2;
			cats_st_tensor *t = (cats_st_tensor*)(realloc)(s->tensor, cap * sizeof(cats_st_tensor));
			if (!t) return -1;
			s->tensor = t;
		}
		cats_st_tensor *t = &s->tensor[s->count++];
		memset(t, 0, sizeof(*t));
		memcpy(t->name, j + k0, klen);
		t->dtype = -1;
		for (size_t p = o0; p < i; p++) {
			if (!strncmp(j + p, "\"dtype\"", 7)) {
				const char *d = (const char*)memchr(j + p + 7, '"', i - (p + 7));
				if (!d) return -1;
				d++;
				t->dtype = !strncmp(d, "F32", 3) ? CATS_ST_F32 : !strncmp(d, "BF16", 4) ? CATS_ST_BF16 :
					   !strncmp(d, "F16", 3) ? CATS_ST_F16 : !strncmp(d, "Q4_0", 4) ? CATS_ST_Q4_0 : -1;
			} else if (!strncmp(j + p, "\"shape\"", 7)) {
				const char *d = (const char*)memchr(j + p + 7, '[', i - (p + 7));
				if (!d) return -1;
				d++;
				while (d < j + i && *d != ']' && t->ndim < 4) {
					t->shape[t->ndim++] = strtoll(d, (char**)&d, 10);
					while (*d == ',' || *d == ' ') d++;
				}
			} else if (!strncmp(j + p, "\"data_offsets\"", 14)) {
				const char *d = (const char*)memchr(j + p + 14, '[', i - (p + 14));
				if (!d) return -1;
				d++;
				t->begin = strtoull(d, (char**)&d, 10);
				d = (const char*)memchr(d, ',', j + i - d);
				if (!d) return -1;
				t->end = strtoull(d + 1, NULL, 10);
			}
		}
	}
	return 0;
}
static inline int cats_st_open(cats_st *s, const char *path)
{
	memset(s, 0, sizeof(*s));
#if defined(_WIN32)
	(void)path;
	return -1;
#else
	int fd = open(path, O_RDONLY);
	if (fd < 0) return -1;
	struct stat st;
	if (fstat(fd, &st) || st.st_size < 8) { close(fd); return -1; }
	s->size = (size_t)st.st_size;
	s->map = (uint8_t*)mmap(NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (s->map == MAP_FAILED) { s->map = NULL; return -1; }
	uint64_t hl;
	memcpy(&hl, s->map, 8);
	if (hl > s->size - 8) { cats_st_close(s); return -1; }
	s->data = s->map + 8 + hl;
	if (cats_st_parse(s, (const char*)s->map + 8, (size_t)hl)) { cats_st_close(s); return -1; }
	const uint64_t avail = s->size - 8 - hl;	// every tensor must lie inside the data section
	for (int i = 0; i < s->count; i++) {
		if (s->tensor[i].begin > s->tensor[i].end || s->tensor[i].end > avail) { cats_st_close(s); return -1; }
	}
	return 0;
#endif
}
static inline const cats_st_tensor* cats_st_find(const cats_st *s, const char *name)
{
	for (int i = 0; i < s->count; i++) if (!strcmp(s->tensor[i].name, name)) return &s->tensor[i];
	return NULL;
}
static inline int64_t cats_st_numel(const cats_st_tensor *t)
{
	int64_t n = 1;
	for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
	return n;
}
// Bytes a tensor of this dtype and element count occupies
static inline uint64_t cats_st_bytes(int dtype, int64_t n)
{
	return dtype == CATS_ST_F32 ? (uint64_t)n * 4 : dtype == CATS_ST_Q4_0 ? cats_q4_0_row_bytes((size_t)n) : (uint64_t)n * 2;
}
// Pointer to the tensor's bytes inside the mapping (valid until cats_st_close)
static inline const uint8_t* cats_st_raw(const cats_st *s, const cats_st_tensor *t) { return s->data + t->begin; }
// Widens (or copies) the whole tensor to float32. dst needs cats_st_numel() floats.
static inline int cats_st_read_f32(const cats_st *s, const cats_st_tensor *t, float *dst)
{
	int64_t n = cats_st_numel(t);
	const uint8_t *p = s->data + t->begin;
	if (t->dtype < 0 || cats_st_bytes(t->dtype, n) != t->end - t->begin) return -1;
	if (t->dtype == CATS_ST_Q4_0) {
		if (t->ndim != 2 || t->shape[1] % CATS_Q4_0_BLOCK) return -1;
		for (int64_t r = 0; r < t->shape[0]; r++)
			cats_q4_0_dequantize_row(p + (size_t)r * cats_q4_0_row_bytes((size_t)t->shape[1]), (size_t)t->shape[1], dst + r * t->shape[1]);
		return 0;
	}
	if (t->dtype == CATS_ST_F32) memcpy(dst, p, (size_t)n * 4);
	else {
		const int bf = t->dtype == CATS_ST_BF16;
		#pragma omp parallel for schedule(static) if(n > (1 << 16))
		for (int64_t i = 0; i < n; i++) {
			uint16_t h = (uint16_t)(p[2*i] | (p[2*i+1] << 8));
			dst[i] = bf ? cats_bf16_to_float(h) : cats_half_to_float(h);
		}
	}
	return 0;
}
// One row of a 2-D tensor (e.g. an embedding table) without widening the rest.
static inline int cats_st_read_row_f32(const cats_st *s, const cats_st_tensor *t, int64_t row, float *dst)
{
	if (t->ndim != 2 || row < 0 || row >= t->shape[0] || t->dtype < 0) return -1;
	int64_t w = t->shape[1];
	if (t->dtype == CATS_ST_Q4_0) {
		if (w % CATS_Q4_0_BLOCK) return -1;
		cats_q4_0_dequantize_row(s->data + t->begin + (size_t)row * cats_q4_0_row_bytes((size_t)w), (size_t)w, dst);
		return 0;
	}
	const uint8_t *p = s->data + t->begin + (size_t)row * w * (t->dtype == CATS_ST_F32 ? 4 : 2);
	if (t->dtype == CATS_ST_F32) memcpy(dst, p, (size_t)w * 4);
	else for (int64_t i = 0; i < w; i++) {
		uint16_t h = (uint16_t)(p[2*i] | (p[2*i+1] << 8));
		dst[i] = t->dtype == CATS_ST_BF16 ? cats_bf16_to_float(h) : cats_half_to_float(h);
	}
	return 0;
}

// Writes a safetensors file. Each entry points at its bytes (data), which are copied in order.
typedef struct {
	const char *name;
	const char *dtype;		// "F32", "F16", "BF16" or "Q4_0"
	int ndim;
	int64_t shape[4];
	const void *data;
	uint64_t bytes;
} cats_st_entry;
static inline int cats_st_write(const char *path, const cats_st_entry *e, int count, const char *format_note)
{
	size_t cap = 4096 + (format_note ? strlen(format_note) : 0);
	for (int i = 0; i < count; i++) cap += strlen(e[i].name) + strlen(e[i].dtype) + 256;	// 4 dims, 2 offsets and the punctuation
	char *js = (char*)(malloc)(cap + 8);	// +8: padding to a multiple of 8
	if (!js) return -1;
	size_t n = 0;
	uint64_t off = 0;
	n += (size_t)snprintf(js + n, cap - n, "{\"__metadata__\":{\"format\":\"%s\"}", format_note ? format_note : "pt");
	for (int i = 0; i < count; i++) {
		n += (size_t)snprintf(js + n, cap - n, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[", e[i].name, e[i].dtype);
		for (int d = 0; d < e[i].ndim; d++) n += (size_t)snprintf(js + n, cap - n, "%s%lld", d ? "," : "", (long long)e[i].shape[d]);
		n += (size_t)snprintf(js + n, cap - n, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + e[i].bytes));
		off += e[i].bytes;
	}
	n += (size_t)snprintf(js + n, cap - n, "}");
	while (n % 8) js[n++] = ' ';			// the data section starts 8-byte aligned
	FILE *f = fopen(path, "wb");
	int rc = -1;
	if (f) {
		uint64_t hl = n;
		rc = fwrite(&hl, 8, 1, f) == 1 && fwrite(js, 1, n, f) == n ? 0 : -1;
		for (int i = 0; i < count && !rc; i++) if (fwrite(e[i].data, 1, (size_t)e[i].bytes, f) != e[i].bytes) rc = -1;
		if (fclose(f)) rc = -1;
	}
	(free)(js);
	return rc;
}

// ---------------------------------------------------------------------------
// Transformer building blocks (float32, row-major [tokens][channels])
// ---------------------------------------------------------------------------
// RMSNorm without a learned scale (e.g. value normalisation)
static inline void cats_rmsnorm_noscale(float *dst, const float *src, unsigned d, float eps)
{
	double sum = 0;
	for (unsigned i = 0; i < d; i++) sum += (double)src[i] * src[i];
	float scale = 1.0f / sqrtf((float)(sum / d) + eps);
	for (unsigned i = 0; i < d; i++) dst[i] = src[i] * scale;
}
// RoPE tables in the "rotate_half" layout: cos/sin [n][dim], the first and second half repeat.
// theta^(-2j/dim) for j < dim/2; pos[i] is the position of token i.
static inline void cats_rope_table(float *cs, float *sn, const float *pos, size_t n, unsigned dim, float theta)
{
	unsigned h = dim / 2;
	for (size_t i = 0; i < n; i++) for (unsigned j = 0; j < h; j++) {
		double a = (double)pos[i] * pow((double)theta, -2.0 * j / dim);
		cs[i*dim + j] = cs[i*dim + j + h] = (float)cos(a);
		sn[i*dim + j] = sn[i*dim + j + h] = (float)sin(a);
	}
}
// Rotates one block of n channels in place: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s).
// Axial 2-D RoPE calls this once per spatial axis on consecutive blocks of the head.
static inline void cats_rope_block(float *x, const float *cs, const float *sn, unsigned n)
{
	unsigned h = n / 2;
	for (unsigned j = 0; j < h; j++) {
		float a = x[j], b = x[j + h];
		x[j]     = a * cs[j]     - b * sn[j];
		x[j + h] = b * cs[j + h] + a * sn[j + h];
	}
}
// Bidirectional multi-head attention with grouped K/V heads and no causal mask.
//   q [t][nq*hd], k and v [t][nkv*hd] -> out [t][nq*hd]; scores are scaled by `scale`.
//   window > 0 limits each token to keys with |i-j| < window (sliding window); 0 = full.
// Heads are first repacked contiguously (Q and K as [head][t][hd], V transposed as [head][hd][t]); the work is
// then cut into (head, block of CATS_ATT_ROWS queries) tasks spread over the OpenMP threads. Each task builds its
// small score block with the serial blocked GEMM, applies the softmax and multiplies by V, so the score matrix
// never exceeds CATS_ATT_ROWS x t floats and stays in cache. Cost is O(t^2 hd) per head.
#ifndef CATS_ATT_ROWS
#define CATS_ATT_ROWS	64
#endif
static inline int cats_attention_bidir(float *out, const float *q, const float *k, const float *v,
				       size_t t, unsigned nq, unsigned nkv, unsigned hd, float scale, size_t window)
{
	if (!nkv || nq % nkv || !t) return -1;
	const size_t qs = (size_t)nq * hd, ks = (size_t)nkv * hd, nb = (t + CATS_ATT_ROWS - 1) / CATS_ATT_ROWS;
	float *qp = (float*)(malloc)(nq * t * hd * sizeof(float)), *kp = (float*)(malloc)(nkv * t * hd * sizeof(float)),
	      *vt = (float*)(malloc)(nkv * hd * t * sizeof(float));
	if (!qp || !kp || !vt) { (free)(qp); (free)(kp); (free)(vt); return -1; }
	#pragma omp parallel for schedule(static)
	for (long i = 0; i < (long)t; i++) {
		for (unsigned h = 0; h < nq; h++) memcpy(qp + ((size_t)h * t + i) * hd, q + i * qs + (size_t)h * hd, hd * sizeof(float));
		for (unsigned g = 0; g < nkv; g++) memcpy(kp + ((size_t)g * t + i) * hd, k + i * ks + (size_t)g * hd, hd * sizeof(float));
	}
	#pragma omp parallel for schedule(static) collapse(2)
	for (long g = 0; g < (long)nkv; g++) for (long d = 0; d < (long)hd; d++) {
		float *dst = vt + ((size_t)g * hd + d) * t;
		for (size_t i = 0; i < t; i++) dst[i] = v[i * ks + (size_t)g * hd + d];
	}
	const unsigned group = nq / nkv;
	int fail = 0;
	#pragma omp parallel
	{
		float *sc = (float*)(malloc)(CATS_ATT_ROWS * t * sizeof(float)), *ob = (float*)(malloc)(CATS_ATT_ROWS * hd * sizeof(float));
		if (!sc || !ob) {
			#pragma omp atomic write
			fail = 1;
		}
		#pragma omp for schedule(dynamic, 1)
		for (long task = 0; task < (long)(nq * nb); task++) {
			if (fail) continue;
			const unsigned h = (unsigned)(task / nb), g = h / group;
			const size_t i0 = (size_t)(task % nb) * CATS_ATT_ROWS, rows = t - i0 < CATS_ATT_ROWS ? t - i0 : CATS_ATT_ROWS;
			cats_gemm_nt_cpu(sc, qp + ((size_t)h * t + i0) * hd, kp + (size_t)g * t * hd, rows, hd, t, NULL);	// q_i . k_j
			for (size_t r = 0; r < rows; r++) {
				float *row = sc + r * t;
				const size_t i = i0 + r;
				size_t lo = 0, hi = t;
				if (window) {
					lo = i + 1 > window ? i + 1 - window : 0;
					hi = i + window < t ? i + window : t;
				}
				cats_softmax_scaled(row, t, lo, hi, scale);
			}
			cats_gemm_nt_cpu(ob, sc, vt + (size_t)g * hd * t, rows, t, hd, NULL);				// P . V
			for (size_t r = 0; r < rows; r++) memcpy(out + (i0 + r) * qs + (size_t)h * hd, ob + r * hd, hd * sizeof(float));
		}
		(free)(sc); (free)(ob);
	}
	(free)(qp); (free)(kp); (free)(vt);
	return fail ? -1 : 0;
}

#endif /* CATSEYE_H */

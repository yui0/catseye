//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2026 Yuichiro Nakada
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

	int timesteps;		// RNN: number of input time steps
	int features;		// RNN: input features per time step
	int hiddens;		// RNN: recurrent hidden units

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
//	for (int b=0; b<l->p->batch; b++) for (int n=0; n<10; n++) CATS_LOG("%f[%f] ", l->dIn[b*l->inputs +l->inputs-1-n], l->z[b*l->inputs +l->inputs-1-n]);
//	CATS_LOG("%d\n", l->wsize);

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

/*
 * Vanilla recurrent layer with tanh state and full back-propagation through
 * time (BPTT). The input tensor is flattened as [batch][timesteps][features].
 * The layer output is the final hidden state [batch][hiddens].
 *
 * Weight layout: Wxh[hiddens][features], Whh[hiddens][hiddens], bias[hiddens].
 * workspace per sample: states[timesteps+1][hiddens] followed by
 *                       deltas[timesteps][hiddens].
 */
static void CatsEye_recurrent_forward(CatsEye_layer *l)
{
	const int T=l->timesteps, F=l->features, H=l->hiddens;
	const size_t state_size=((size_t)T+1)*(size_t)H;
	const size_t workspace_size=2*state_size;
	real *wxh=l->w;
	real *whh=wxh+(size_t)H*F;
	real *bias=whh+(size_t)H*H;

	for (int b=0; b<l->p->batch; ++b) {
		real *state=l->workspace+(size_t)b*workspace_size;
		real *input=l->x+(size_t)b*l->inputs;
		memset(state,0,(size_t)H*sizeof(real));
		for (int t=0; t<T; ++t) {
			real *prev=state+(size_t)t*H;
			real *cur=state+(size_t)(t+1)*H;
			const real *xt=input+(size_t)t*F;
			for (int j=0; j<H; ++j) {
				real z=bias[j];
				for (int c=0; c<F; ++c) z+=wxh[(size_t)j*F+c]*xt[c];
				for (int k=0; k<H; ++k) z+=whh[(size_t)j*H+k]*prev[k];
				cur[j]=tanh(z);
			}
		}
		memcpy(l->z+(size_t)b*H,state+(size_t)T*H,(size_t)H*sizeof(real));
	}
}

static void CatsEye_recurrent_backward(CatsEye_layer *l)
{
	const int T=l->timesteps, F=l->features, H=l->hiddens;
	const size_t state_size=((size_t)T+1)*(size_t)H;
	const size_t workspace_size=2*state_size;
	real *whh=l->w+(size_t)H*F;
	real *wxh=l->w;

	for (int b=0; b<l->p->batch; ++b) {
		real *state=l->workspace+(size_t)b*workspace_size;
		real *delta=state+state_size;
		for (int j=0; j<H; ++j) {
			real h=state[(size_t)T*H+j];
			delta[(size_t)(T-1)*H+j]=l->dOut[(size_t)b*H+j]*(1.0-h*h);
		}
		for (int t=T-2; t>=0; --t) {
			for (int k=0; k<H; ++k) {
				real sum=0;
				for (int j=0; j<H; ++j) sum+=whh[(size_t)j*H+k]*delta[(size_t)(t+1)*H+j];
				real h=state[(size_t)(t+1)*H+k];
				delta[(size_t)t*H+k]=sum*(1.0-h*h);
			}
		}
		/* The first recurrent layer has no preceding layer, so dIn is NULL. */
		if (l->dIn) {
			real *din=l->dIn+(size_t)b*l->inputs;
			for (int t=0; t<T; ++t) {
				for (int c=0; c<F; ++c) {
					real sum=0;
					for (int j=0; j<H; ++j) sum+=wxh[(size_t)j*F+c]*delta[(size_t)t*H+j];
					din[(size_t)t*F+c]=sum;
				}
			}
		}
	}
}

static void CatsEye_recurrent_update(CatsEye_layer *l)
{
	const int T=l->timesteps, F=l->features, H=l->hiddens;
	const size_t state_size=((size_t)T+1)*(size_t)H;
	const size_t workspace_size=2*state_size;
	const size_t whh_offset=(size_t)H*F;
	const size_t bias_offset=whh_offset+(size_t)H*H;
	const real inv_batch=1.0/(real)l->p->batch;

	for (int b=0; b<l->p->batch; ++b) {
		real *state=l->workspace+(size_t)b*workspace_size;
		real *delta=state+state_size;
		real *input=l->x+(size_t)b*l->inputs;
		for (int t=0; t<T; ++t) {
			const real *xt=input+(size_t)t*F;
			const real *hp=state+(size_t)t*H;
			const real *dt=delta+(size_t)t*H;
			for (int j=0; j<H; ++j) {
				real g=dt[j]*inv_batch;
				for (int c=0; c<F; ++c) l->dw[(size_t)j*F+c]+=g*xt[c];
				for (int k=0; k<H; ++k) l->dw[whh_offset+(size_t)j*H+k]+=g*hp[k];
				l->dw[bias_offset+j]+=g;
			}
		}
	}
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
	CatsEye_recurrent_forward,

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
	CatsEye_recurrent_backward,

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
	CatsEye_recurrent_update,

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
	CATS_PIXELSHUFFLER, CATS_RECURRENT,
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
	"subpixel", "rnn",
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
		if (!i) {
			l->dIn = NULL;
			if (l->type!=CATS_RECURRENT) l->backward = CatsEye_none; // first layer
		}
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

		case CATS_RECURRENT:
			if (l->timesteps < 1 || l->features < 1) {
				fprintf(stderr,"CatsEye: RNN requires positive timesteps and features\n"); exit(1);
			}
			if (!l->hiddens) l->hiddens=l->outputs;
			if (l->hiddens < 1 || (size_t)l->timesteps > SIZE_MAX/(size_t)l->features ||
			    (size_t)l->inputs != (size_t)l->timesteps*(size_t)l->features) {
				fprintf(stderr,"CatsEye: RNN input size must equal timesteps * features, and hiddens must be positive\n"); exit(1);
			}
			l->outputs=l->hiddens;
			l->ch=l->hiddens; l->ox=l->hiddens; l->oy=1;
			n[i]=l->features+l->hiddens;
			m[i]=l->hiddens;
			b[i]=1;
			l->wsize=(n[i]+b[i])*m[i];
			if (this->batch < 1 || (size_t)l->timesteps+1 > SIZE_MAX/(2*(size_t)l->hiddens)) {
				fprintf(stderr,"CatsEye: RNN workspace size overflow\n"); exit(1);
			}
			size_t rnn_workspace_per_sample=2*((size_t)l->timesteps+1)*(size_t)l->hiddens;
			if ((size_t)this->batch > SIZE_MAX/rnn_workspace_per_sample) {
				fprintf(stderr,"CatsEye: RNN batch workspace size overflow\n"); exit(1);
			}
			l->workspace=calloc(rnn_workspace_per_sample*(size_t)this->batch,sizeof(real));
			if (!l->workspace) { fprintf(stderr,"CatsEye: cannot allocate RNN workspace\n"); exit(1); }
			CATS_LOG("%3d %-12s steps:%d features:%d hidden:%d -> %d\n",i+1,CatsEye_string[l->type],l->timesteps,l->features,l->hiddens,l->outputs);
			break;

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
		if (l->type==CATS_RECURRENT) memset(l->bias, 0, sizeof(real)*l->hiddens);
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

	this->start = 0;
	this->stop = (this->layer[0].type==CATS_RECURRENT ? -1 : 0);
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
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define CATS_SIMD_NEON 1
// int8 dot product of two 16-lane vectors accumulated into 4 int32 lanes (SDOT where the CPU has it, as Apple M1+ does)
static inline int32x4_t cats_neon_dot(int32x4_t acc, int8x16_t a, int8x16_t b)
{
#if defined(__ARM_FEATURE_DOTPROD)
	return vdotq_s32(acc, a, b);
#else
	return vpadalq_s16(vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b))), vmull_high_s8(a, b));
#endif
}
#else
#define CATS_SIMD_NEON 0
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
#if CATS_SIMD_X86
CATS_AVX_FN static float cats_dot_f32_avx1(const float*x,const float*w,size_t n)
{
	__m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
	size_t j=0;
	for (; j+16<=n; j+=16) {
		a0=_mm256_add_ps(a0,_mm256_mul_ps(_mm256_loadu_ps(x+j),_mm256_loadu_ps(w+j)));
		a1=_mm256_add_ps(a1,_mm256_mul_ps(_mm256_loadu_ps(x+j+8),_mm256_loadu_ps(w+j+8)));
	}
	const __m256 a=_mm256_add_ps(a0,a1);
	__m128 h=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
	h=_mm_add_ps(h,_mm_movehl_ps(h,h));
	h=_mm_add_ss(h,_mm_shuffle_ps(h,h,1));
	float sum=_mm_cvtss_f32(h);
	for (; j<n; j++) sum+=x[j]*w[j];
	return sum;
}
#endif
static inline float cats_dot_f32(const float*x,const float*w,size_t n)
{
#if CATS_SIMD_X86
	if (n>=32&&cats_cpu_simd_level()>=2) {
		return cats_dot_fma(x,w,n);
	}
	if (n>=32&&cats_cpu_simd_level()>=1) {
		return cats_dot_f32_avx1(x,w,n);
	}
#elif CATS_SIMD_NEON
	if (n>=16) {
		float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0;
		size_t j=0;
		for (; j+16<=n; j+=16) {
			a0=vfmaq_f32(a0,vld1q_f32(x+j),vld1q_f32(w+j));
			a1=vfmaq_f32(a1,vld1q_f32(x+j+4),vld1q_f32(w+j+4));
			a2=vfmaq_f32(a2,vld1q_f32(x+j+8),vld1q_f32(w+j+8));
			a3=vfmaq_f32(a3,vld1q_f32(x+j+12),vld1q_f32(w+j+12));
		}
		float sum=vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3)));
		for (; j<n; j++) sum+=x[j]*w[j];
		return sum;
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
// ---- cats_pfor: run fn(ctx, lo, hi) over [0, n) in chunks. OpenMP natively; a pthread pool under Emscripten
// (-pthread; the pool is started by cats_set_threads). Nested calls and tiny jobs run serially in the caller.
typedef void (*cats_pf_fn)(void *ctx, long lo, long hi);
#if defined(__EMSCRIPTEN_PTHREADS__)
#include <pthread.h>
#define CATS_POOL_MAX 32
static struct {
	pthread_t th[CATS_POOL_MAX];
	pthread_mutex_t mu; pthread_cond_t wake, done;
	int nworkers, gen, pending, started;
	atomic_long next;
	long n, chunk; cats_pf_fn fn; void *ctx;
} cats_pool = { .mu = PTHREAD_MUTEX_INITIALIZER, .wake = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };
static _Thread_local int cats_in_pfor;
static void cats_pool_run(void)
{
	cats_in_pfor = 1;
	for (;;) {
		long lo = atomic_fetch_add(&cats_pool.next, cats_pool.chunk);
		if (lo >= cats_pool.n) break;
		long hi = lo + cats_pool.chunk;
		cats_pool.fn(cats_pool.ctx, lo, hi < cats_pool.n ? hi : cats_pool.n);
	}
	cats_in_pfor = 0;
}
static void *cats_pool_worker(void *arg)
{
	(void)arg;
	int seen = 0;
	for (;;) {
		pthread_mutex_lock(&cats_pool.mu);
		while (cats_pool.gen == seen) pthread_cond_wait(&cats_pool.wake, &cats_pool.mu);
		seen = cats_pool.gen;
		pthread_mutex_unlock(&cats_pool.mu);
		cats_pool_run();
		pthread_mutex_lock(&cats_pool.mu);
		if (--cats_pool.pending == 0) pthread_cond_signal(&cats_pool.done);
		pthread_mutex_unlock(&cats_pool.mu);
	}
	return NULL;
}
// Total threads (caller included). Workers are only ever added (they sleep when idle); `limit` caps how many take part.
static int cats_pool_limit = 1;
static inline int cats_pool_start(int total)
{
	if (total < 1) total = 1;
	if (total > CATS_POOL_MAX) total = CATS_POOL_MAX;
	while (cats_pool.started < total - 1) {
		if (pthread_create(&cats_pool.th[cats_pool.started], NULL, cats_pool_worker, NULL)) break;
		cats_pool.started++;
	}
	cats_pool_limit = total;
	return cats_pool.started + 1;
}
static inline void cats_pfor(long n, long chunk, cats_pf_fn fn, void *ctx)
{
	if (chunk < 1) chunk = 1;
	if (cats_pool.started == 0 || cats_in_pfor || n <= chunk) { fn(ctx, 0, n); return; }
	// every started worker joins in (the limit only matters when the pool shrinks, which it never does): keep it simple
	pthread_mutex_lock(&cats_pool.mu);
	cats_pool.n = n; cats_pool.chunk = chunk; cats_pool.fn = fn; cats_pool.ctx = ctx;
	atomic_store(&cats_pool.next, 0);
	cats_pool.pending = cats_pool.started;
	cats_pool.gen++;
	pthread_cond_broadcast(&cats_pool.wake);
	pthread_mutex_unlock(&cats_pool.mu);
	cats_pool_run();
	pthread_mutex_lock(&cats_pool.mu);
	while (cats_pool.pending) pthread_cond_wait(&cats_pool.done, &cats_pool.mu);
	pthread_mutex_unlock(&cats_pool.mu);
}
#else
static inline void cats_pfor(long n, long chunk, cats_pf_fn fn, void *ctx)
{
	if (chunk < 1) chunk = 1;
	const long nc = (n + chunk - 1) / chunk;
	#pragma omp parallel for schedule(dynamic,1) if(nc > 1)
	for (long c = 0; c < nc; c++) {
		long lo = c * chunk, hi = lo + chunk;
		fn(ctx, lo, hi < n ? hi : n);
	}
}
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
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
// WebAssembly SIMD: the same 4 x 2 tile with f32x4 accumulators.
static inline float cats_hsum4w(v128_t v)
{
	return wasm_f32x4_extract_lane(v,0)+wasm_f32x4_extract_lane(v,1)+wasm_f32x4_extract_lane(v,2)+wasm_f32x4_extract_lane(v,3);
}
static void cats_tile_wasm(float*c,size_t ldc,const float*x,size_t ldx,const float*w,
                           size_t ldw,size_t klen,size_t nt,size_t no,int acc,const float*bias)
{
	if (nt!=4||no!=2) {
		cats_tile_generic(c,ldc,x,ldx,w,ldw,klen,nt,no,acc,bias);
		return;
	}
	const float*x0=x,*x1=x+ldx,*x2=x+2*ldx,*x3=x+3*ldx,*w0=w,*w1=w+ldw;
	v128_t z=wasm_f32x4_splat(0.0f),a00=z,a01=z,a10=z,a11=z,a20=z,a21=z,a30=z,a31=z;
	size_t k=0;
	for (; k+4<=klen; k+=4) {
		v128_t b0=wasm_v128_load(w0+k),b1=wasm_v128_load(w1+k),v;
		v=wasm_v128_load(x0+k);a00=wasm_f32x4_add(a00,wasm_f32x4_mul(v,b0));a01=wasm_f32x4_add(a01,wasm_f32x4_mul(v,b1));
		v=wasm_v128_load(x1+k);a10=wasm_f32x4_add(a10,wasm_f32x4_mul(v,b0));a11=wasm_f32x4_add(a11,wasm_f32x4_mul(v,b1));
		v=wasm_v128_load(x2+k);a20=wasm_f32x4_add(a20,wasm_f32x4_mul(v,b0));a21=wasm_f32x4_add(a21,wasm_f32x4_mul(v,b1));
		v=wasm_v128_load(x3+k);a30=wasm_f32x4_add(a30,wasm_f32x4_mul(v,b0));a31=wasm_f32x4_add(a31,wasm_f32x4_mul(v,b1));
	}
	float r[4][2]={{cats_hsum4w(a00),cats_hsum4w(a01)},{cats_hsum4w(a10),cats_hsum4w(a11)},
	               {cats_hsum4w(a20),cats_hsum4w(a21)},{cats_hsum4w(a30),cats_hsum4w(a31)}};
	for (; k<klen; k++) {
		r[0][0]+=x0[k]*w0[k];r[0][1]+=x0[k]*w1[k];r[1][0]+=x1[k]*w0[k];r[1][1]+=x1[k]*w1[k];
		r[2][0]+=x2[k]*w0[k];r[2][1]+=x2[k]*w1[k];r[3][0]+=x3[k]*w0[k];r[3][1]+=x3[k]*w1[k];
	}
	for (int t=0; t<4; t++)for (int o=0; o<2; o++) {
			float v=r[t][o];
			if (acc) v+=c[(size_t)t*ldc+o]; else if (bias) v+=bias[o];
			c[(size_t)t*ldc+o]=v;
		}
}
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
typedef struct { float *c; const float *x, *w, *bias; size_t n, in, out; long nob; cats_tile_fn tile; } cats_gemm_ctx;
static void cats_gemm_body(void *vp, long lo, long hi)
{
	const cats_gemm_ctx *g = (const cats_gemm_ctx*)vp;
	const size_t n=g->n,in=g->in,out=g->out;
	for (long idx=lo; idx<hi; idx++) {
		long tb=idx/g->nob,ob=idx%g->nob;
		size_t t0=(size_t)tb*CATS_GEMM_TB,t1=t0+CATS_GEMM_TB;
		if (t1>n) t1=n;
		size_t o0=(size_t)ob*CATS_GEMM_OB,o1=o0+CATS_GEMM_OB;
		if (o1>out) o1=out;
		for (size_t k0=0; k0<in; k0+=CATS_GEMM_KC) {
			size_t klen=in-k0<CATS_GEMM_KC?in-k0:CATS_GEMM_KC;
			int acc=k0!=0;
			for (size_t t=t0; t<t1; t+=4) {
				size_t nt=t1-t<4?t1-t:4;
				for (size_t o=o0; o<o1; o+=2) {
					size_t no=o1-o<2?o1-o:2;
					g->tile(g->c+t*out+o,out,g->x+t*in+k0,in,g->w+o*in+k0,in,klen,nt,no,acc,
					        g->bias?g->bias+o:NULL);
				}
			}
		}
	}
}
static inline void cats_gemm_nt_cpu(float*c,const float*x,const float*w,size_t n,
                                      size_t in,size_t out,const float*bias)
{
	if (!n||!in||!out) {
		return;
	}
	cats_tile_fn tile=cats_tile_generic;
#ifdef __wasm_simd128__
	tile=cats_tile_wasm;
#endif
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
	cats_gemm_ctx g = { c, x, w, bias, n, in, out, 0, tile };
	const long ntb=(long)((n+CATS_GEMM_TB-1)/CATS_GEMM_TB);
	g.nob=(long)((out+CATS_GEMM_OB-1)/CATS_GEMM_OB);
	const int big=(double)n*(double)in*(double)out>1.0e6;
	if (big) cats_pfor(ntb*g.nob,1,cats_gemm_body,&g); else cats_gemm_body(&g,0,ntb*g.nob);
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
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif
// Threads to use when the user gave neither -j nor OMP_NUM_THREADS: one per physical core.
// Hyper-threads share the FP units, so they add no GEMM throughput and only add barrier cost.
// Returns 0 = leave the OpenMP default.
static inline int cats_physical_threads(void)
{
#if defined(_OPENMP) && defined(__APPLE__)
	// Apple silicon: only the performance cores (the efficiency cores would stall statically scheduled loops)
	int v=0;
	size_t sz=sizeof(v);
	if (sysctlbyname("hw.perflevel0.physicalcpu",&v,&sz,NULL,0)==0&&v>0) {
		return v;
	}
	sz=sizeof(v);
	if (sysctlbyname("hw.physicalcpu",&v,&sz,NULL,0)==0&&v>0) {
		return v;
	}
	return 0;
#elif defined(_OPENMP)
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
#if defined(__EMSCRIPTEN_PTHREADS__)
	cats_pool_start(requested > 0 ? requested : 4);
#elif defined(_OPENMP)
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
#if CATS_SIMD_NEON
static float cats_dot_q4_0_q8_0_neon(const uint8_t *a, const int8_t *b, int n)
{
	float sum = 0.0f;
	const uint8x16_t m = vdupq_n_u8(0x0F);
	const int8x16_t eight = vdupq_n_s8(8);
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q4_0_BYTES, b += CATS_Q8_0_BYTES) {
		const uint8x16_t q = vld1q_u8(a + 2);
		const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m)), eight);
		const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), eight);
		int32x4_t acc = cats_neon_dot(vdupq_n_s32(0), lo, vld1q_s8(b));
		acc = cats_neon_dot(acc, hi, vld1q_s8(b + 16));
		float ds;
		memcpy(&ds, b + 32, 4);
		sum += (float)vaddvq_s32(acc) * (cats_half_to_float(cats_load_u16(a)) * ds);
	}
	return sum;
}
#endif
static inline float cats_dot_q4_0_q8_0(const uint8_t *w, const int8_t *a, int n)
{
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) return cats_dot_q4_0_q8_0_avx(w, a, n);
#elif CATS_SIMD_NEON
	return cats_dot_q4_0_q8_0_neon(w, a, n);
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

// ---- Q8_0 weights (GGUF block: 2-byte f16 scale + 32 int8 = 34 bytes per 32 values) x Q8_0 activations ----
#define CATS_Q8_0W_BYTES	34
static inline size_t cats_q8_0w_row_bytes(size_t n) { return n / CATS_Q4_0_BLOCK * CATS_Q8_0W_BYTES; }
static inline void cats_q8_0w_dequantize_row(const uint8_t *in, size_t n, float *out)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, in += CATS_Q8_0W_BYTES, out += CATS_Q4_0_BLOCK) {
		float d = cats_half_to_float(cats_load_u16(in));
		for (int j = 0; j < 32; j++) out[j] = (int8_t)in[2 + j] * d;
	}
}
#if CATS_SIMD_X86
CATS_AVX_FN static float cats_dot_q8_0w_q8_0_avx(const uint8_t *x, const int8_t *y, int n)
{
	int nb = n >> 5;
	__m256 acc = _mm256_setzero_ps();
	do {
		float delta = cats_half_to_float(cats_load_u16(x));
		const __m128i i32_0 = cats_mul_sum_i8_pairs(_mm_loadu_si128((const __m128i*)(x + 2)), _mm_loadu_si128((const __m128i*)y));
		const __m128i i32_1 = cats_mul_sum_i8_pairs(_mm_loadu_si128((const __m128i*)(x + 18)), _mm_loadu_si128((const __m128i*)(y + 16)));
		const __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));
		float ds;
		memcpy(&ds, y + 32, 4);
		x += CATS_Q8_0W_BYTES; y += CATS_Q8_0_BYTES;
		acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(delta * ds), p));
	} while (--nb > 0);
	return cats_hsum_ps(acc);
}
#endif
static float cats_dot_q8_0w_q8_0_scalar(const uint8_t *a, const int8_t *b, int n)
{
	float val = 0.0f;
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q8_0W_BYTES, b += CATS_Q8_0_BYTES) {
		float delta = cats_half_to_float(cats_load_u16(a)), ds;
		int32_t ival = 0;
		for (int j = 0; j < 32; j++) ival += (int32_t)(int8_t)a[2 + j] * b[j];
		memcpy(&ds, b + 32, 4);
		val += ival * (delta * ds);
	}
	return val;
}
#if CATS_SIMD_NEON
static float cats_dot_q8_0w_q8_0_neon(const uint8_t *a, const int8_t *b, int n)
{
	float sum = 0.0f;
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q8_0W_BYTES, b += CATS_Q8_0_BYTES) {
		int32x4_t acc = cats_neon_dot(vdupq_n_s32(0), vld1q_s8((const int8_t*)(a + 2)), vld1q_s8(b));
		acc = cats_neon_dot(acc, vld1q_s8((const int8_t*)(a + 18)), vld1q_s8(b + 16));
		float ds;
		memcpy(&ds, b + 32, 4);
		sum += (float)vaddvq_s32(acc) * (cats_half_to_float(cats_load_u16(a)) * ds);
	}
	return sum;
}
#endif
static inline float cats_dot_q8_0w_q8_0(const uint8_t *w, const int8_t *a, int n)
{
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) return cats_dot_q8_0w_q8_0_avx(w, a, n);
#elif CATS_SIMD_NEON
	return cats_dot_q8_0w_q8_0_neon(w, a, n);
#endif
	return cats_dot_q8_0w_q8_0_scalar(w, a, n);
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

// Quantizes n floats (n % 32 == 0) to Q8_0 weight blocks (fp16 scale + 32 int8, 34 bytes per block).
static inline void cats_q8_0w_quantize_row(const float *x, size_t n, uint8_t *out)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, x += CATS_Q4_0_BLOCK, out += CATS_Q8_0W_BYTES) {
		float amax = 0;
		for (int j = 0; j < CATS_Q4_0_BLOCK; j++) if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
		uint16_t dh = cats_float_to_half(amax / 127.0f);
		float dq = cats_half_to_float(dh), id = dq ? 1.0f / dq : 0.0f;
		out[0] = (uint8_t)(dh & 0xff); out[1] = (uint8_t)(dh >> 8);
		for (int j = 0; j < CATS_Q4_0_BLOCK; j++) {
			int q = (int)lrintf(x[j] * id);
			out[2 + j] = (uint8_t)(int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
		}
	}
}
// C[n][out] = X[n][in] . W^T (+ bias) with W stored as Q8_0 rows (same strategy as cats_gemm_nt_q4_0).
static inline void cats_gemm_nt_q8_0(float *c, const float *x, const uint8_t *w, size_t n, size_t in, size_t out,
				     const float *bias)
{
	if (!n || !in || !out || in % CATS_Q4_0_BLOCK) return;
	const size_t arow = cats_q8_0_row_bytes(in), wrow = cats_q8_0w_row_bytes(in);
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
		for (long r = 0; r < (long)out; r++) cats_q8_0w_dequantize_row(w + (size_t)r * wrow, in, scratch + (size_t)r * in);
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
		for (size_t t = 0; t < n; t++) c[t * out + r] = cats_dot_q8_0w_q8_0(wr, xq + t * arow, (int)in) + b;
	}
	(free)(xq);
}


// ---- Q3_0: 3-bit weights in blocks of 32 along the input dimension, 14 bytes per block (3.5 bits per weight) ----
// [0..1] fp16 scale d, [2..9] the low 2 bits (byte i holds values i, i+8, i+16, i+24 at bit 0, 2, 4, 6),
// [10..13] the high bits (little endian uint32, bit i = value i). value = (lo | hi << 2) - 4) * d, i.e. -4d .. 3d.
#define CATS_Q3_0_BYTES	14
static inline size_t cats_q3_0_row_bytes(size_t n) { return n / CATS_Q4_0_BLOCK * CATS_Q3_0_BYTES; }
static inline void cats_q3_0_quantize_row(const float *x, size_t n, uint8_t *out, int refine)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, x += CATS_Q4_0_BLOCK, out += CATS_Q3_0_BYTES) {
		float amax = 0, mx = 0;
		for (int j = 0; j < 32; j++) if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); mx = x[j]; }
		int bq[32];
		float bd = 0, berr = FLT_MAX;
		for (int j = 0; j < 32; j++) bq[j] = 4;
		if (amax > 0) {
			const int tries = refine ? 17 : 1;
			for (int t = 0; t < tries; t++) {
				const float d0 = mx / -4.0f * (refine ? 0.8f + 0.025f * t : 1.0f), id = 1.0f / d0;
				int q[32];
				float sxq = 0, sqq = 0;
				for (int j = 0; j < 32; j++) {
					int v = (int)lrintf(x[j] * id);
					v = v < -4 ? -4 : v > 3 ? 3 : v;
					q[j] = v + 4;
					sxq += x[j] * (float)v; sqq += (float)(v * v);
				}
				const float d1 = sqq > 0 ? sxq / sqq : d0;		// least-squares scale for these levels
				const float d = cats_half_to_float(cats_float_to_half(d1));
				float err = 0;
				for (int j = 0; j < 32; j++) { const float e = x[j] - (float)(q[j] - 4) * d; err += e * e; }
				if (err < berr) { berr = err; bd = d; memcpy(bq, q, sizeof(q)); }
			}
		}
		const uint16_t dh = cats_float_to_half(bd);
		out[0] = (uint8_t)dh; out[1] = (uint8_t)(dh >> 8);
		memset(out + 2, 0, 12);
		uint32_t hm = 0;
		for (int j = 0; j < 32; j++) {
			out[2 + (j & 7)] |= (uint8_t)((bq[j] & 3) << (2 * (j >> 3)));
			hm |= (uint32_t)(bq[j] >> 2) << j;
		}
		memcpy(out + 10, &hm, 4);
	}
}
static inline void cats_q3_0_dequantize_row(const uint8_t *in, size_t n, float *out)
{
	for (size_t b = 0; b < n / CATS_Q4_0_BLOCK; b++, in += CATS_Q3_0_BYTES, out += CATS_Q4_0_BLOCK) {
		const float d = cats_half_to_float(cats_load_u16(in));
		uint32_t hm;
		memcpy(&hm, in + 10, 4);
		for (int j = 0; j < 32; j++) out[j] = (float)((int)(((in[2 + (j & 7)] >> (2 * (j >> 3))) & 3) | (((hm >> j) & 1) << 2)) - 4) * d;
	}
}
static float cats_dot_q3_0_q8_0_scalar(const uint8_t *a, const int8_t *b, int n)
{
	float val = 0.0f;
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q3_0_BYTES, b += CATS_Q8_0_BYTES) {
		uint32_t hm;
		memcpy(&hm, a + 10, 4);
		int32_t ival = 0;
		for (int j = 0; j < 32; j++) ival += ((int32_t)(((a[2 + (j & 7)] >> (2 * (j >> 3))) & 3) | (((hm >> j) & 1) << 2)) - 4) * b[j];
		float ds;
		memcpy(&ds, b + 32, 4);
		val += (float)ival * (cats_half_to_float(cats_load_u16(a)) * ds);
	}
	return val;
}
#if CATS_SIMD_X86
// 32 weights of a block as signed bytes (-4..3): the low 2 bits are masked / shifted out of the 8 packed bytes,
// the high bit is spread from the 4 bit bytes with a shuffle and a bit mask
CATS_AVX_FN static float cats_dot_q3_0_q8_0_avx(const uint8_t *a, const int8_t *b, int n)
{
	float sum = 0.0f;
	const __m128i m3 = _mm_set1_epi8(3), four = _mm_set1_epi8(4);
	const __m128i bit = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q3_0_BYTES, b += CATS_Q8_0_BYTES) {
		const __m128i qs = _mm_loadl_epi64((const __m128i*)(a + 2));
		const __m128i l0 = _mm_unpacklo_epi64(_mm_and_si128(qs, m3), _mm_and_si128(_mm_srli_epi16(qs, 2), m3));	// values 0..15
		const __m128i l1 = _mm_unpacklo_epi64(_mm_and_si128(_mm_srli_epi16(qs, 4), m3), _mm_and_si128(_mm_srli_epi16(qs, 6), m3));	// 16..31
		uint32_t hm;
		memcpy(&hm, a + 10, 4);
		const __m128i hv = _mm_cvtsi32_si128((int)hm);
		const __m128i h0 = _mm_shuffle_epi8(hv, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
		const __m128i h1 = _mm_shuffle_epi8(hv, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
		const __m128i x0 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(h0, bit), bit), four);
		const __m128i x1 = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(h1, bit), bit), four);
		const __m128i q0 = _mm_sub_epi8(_mm_or_si128(l0, x0), four), q1 = _mm_sub_epi8(_mm_or_si128(l1, x1), four);
		const __m128i s = _mm_add_epi32(cats_mul_sum_i8_pairs(q0, _mm_loadu_si128((const __m128i*)b)), cats_mul_sum_i8_pairs(q1, _mm_loadu_si128((const __m128i*)(b + 16))));
		__m128i t = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
		t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
		float ds;
		memcpy(&ds, b + 32, 4);
		sum += (float)_mm_cvtsi128_si32(t) * (cats_half_to_float(cats_load_u16(a)) * ds);
	}
	return sum;
}
#endif
#if CATS_SIMD_NEON
static float cats_dot_q3_0_q8_0_neon(const uint8_t *a, const int8_t *b, int n)
{
	float sum = 0.0f;
	const uint8x8_t m3 = vdup_n_u8(3);
	const uint8x16_t bit = vcombine_u8(vcreate_u8(0x8040201008040201ull), vcreate_u8(0x8040201008040201ull)), four = vdupq_n_u8(4);
	for (int i = 0; i < n / CATS_Q4_0_BLOCK; i++, a += CATS_Q3_0_BYTES, b += CATS_Q8_0_BYTES) {
		const uint8x8_t qs = vld1_u8(a + 2);
		const uint8x16_t l0 = vcombine_u8(vand_u8(qs, m3), vand_u8(vshr_n_u8(qs, 2), m3));
		const uint8x16_t l1 = vcombine_u8(vand_u8(vshr_n_u8(qs, 4), m3), vshr_n_u8(qs, 6));
		const uint8x16_t h0 = vcombine_u8(vdup_n_u8(a[10]), vdup_n_u8(a[11])), h1 = vcombine_u8(vdup_n_u8(a[12]), vdup_n_u8(a[13]));
		const uint8x16_t x0 = vandq_u8(vtstq_u8(h0, bit), four), x1 = vandq_u8(vtstq_u8(h1, bit), four);
		const int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l0, x0)), vreinterpretq_s8_u8(four));
		const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l1, x1)), vreinterpretq_s8_u8(four));
		int32x4_t acc = cats_neon_dot(vdupq_n_s32(0), q0, vld1q_s8(b));
		acc = cats_neon_dot(acc, q1, vld1q_s8(b + 16));
		float ds;
		memcpy(&ds, b + 32, 4);
		sum += (float)vaddvq_s32(acc) * (cats_half_to_float(cats_load_u16(a)) * ds);
	}
	return sum;
}
#endif
static inline float cats_dot_q3_0_q8_0(const uint8_t *w, const int8_t *a, int n)
{
#if CATS_SIMD_X86
	if (cats_cpu_simd_level() >= 1) return cats_dot_q3_0_q8_0_avx(w, a, n);
#elif CATS_SIMD_NEON
	return cats_dot_q3_0_q8_0_neon(w, a, n);
#endif
	return cats_dot_q3_0_q8_0_scalar(w, a, n);
}
// C[n][out] = X[n][in] . W^T (+ bias) with W stored as Q3_0 rows (same strategy as cats_gemm_nt_q4_0).
static inline void cats_gemm_nt_q3_0(float *c, const float *x, const uint8_t *w, size_t n, size_t in, size_t out,
				     const float *bias)
{
	if (!n || !in || !out || in % CATS_Q4_0_BLOCK) return;
	const size_t arow = cats_q8_0_row_bytes(in), wrow = cats_q3_0_row_bytes(in);
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
		for (long r = 0; r < (long)out; r++) cats_q3_0_dequantize_row(w + (size_t)r * wrow, in, scratch + (size_t)r * in);
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
		for (size_t t = 0; t < n; t++) c[t * out + r] = cats_dot_q3_0_q8_0(wr, xq + t * arow, (int)in) + b;
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
#define CATS_ST_Q8_0	4		// shape = logical [rows, cols]; rows are cats_q8_0w encoded (GGUF Q8_0 blocks)
#define CATS_ST_Q3_0	5		// shape = logical [rows, cols]; rows are cats_q3_0 encoded (3.5 bits per weight)
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
	int external;			// map belongs to the caller (cats_st_open_mem): never unmapped
} cats_st;

static inline void cats_st_close(cats_st *s)
{
#if !defined(_WIN32)
	if (s->map && !s->external) munmap(s->map, s->size);
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
					   !strncmp(d, "F16", 3) ? CATS_ST_F16 : !strncmp(d, "Q4_0", 4) ? CATS_ST_Q4_0 :
					   !strncmp(d, "Q8_0", 4) ? CATS_ST_Q8_0 : !strncmp(d, "Q3_0", 4) ? CATS_ST_Q3_0 : -1;
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
static inline int cats_st_init_map(cats_st *s)
{
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
}
// A safetensors file already in memory (WebAssembly, embedded data); the caller keeps the buffer alive.
static inline int cats_st_open_mem(cats_st *s, const void *buf, size_t size)
{
	memset(s, 0, sizeof(*s));
	if (size < 8) return -1;
	s->map = (uint8_t*)buf; s->size = size; s->external = 1;
	return cats_st_init_map(s);
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
	return cats_st_init_map(s);
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
	return dtype == CATS_ST_F32 ? (uint64_t)n * 4 : dtype == CATS_ST_Q4_0 ? cats_q4_0_row_bytes((size_t)n) :
	       dtype == CATS_ST_Q3_0 ? cats_q3_0_row_bytes((size_t)n) :
	       dtype == CATS_ST_Q8_0 ? cats_q8_0w_row_bytes((size_t)n) : (uint64_t)n * 2;
}
// Pointer to the tensor's bytes inside the mapping (valid until cats_st_close)
static inline const uint8_t* cats_st_raw(const cats_st *s, const cats_st_tensor *t) { return s->data + t->begin; }
typedef struct { const uint8_t *p; size_t rb, w; float *dst; int q8; int q3; } cats_dq_ctx;
static void cats_st_dequant_rows(void *vp, long lo, long hi)
{
	const cats_dq_ctx *d = (const cats_dq_ctx*)vp;
	for (long r = lo; r < hi; r++) {
		if (d->q8) cats_q8_0w_dequantize_row(d->p + (size_t)r * d->rb, d->w, d->dst + (size_t)r * d->w);
		else if (d->q3) cats_q3_0_dequantize_row(d->p + (size_t)r * d->rb, d->w, d->dst + (size_t)r * d->w);
		else cats_q4_0_dequantize_row(d->p + (size_t)r * d->rb, d->w, d->dst + (size_t)r * d->w);
	}
}
// Widens (or copies) the whole tensor to float32. dst needs cats_st_numel() floats.
static inline int cats_st_read_f32(const cats_st *s, const cats_st_tensor *t, float *dst)
{
	int64_t n = cats_st_numel(t);
	const uint8_t *p = s->data + t->begin;
	if (t->dtype < 0 || cats_st_bytes(t->dtype, n) != t->end - t->begin) return -1;
	if (t->dtype == CATS_ST_Q4_0 || t->dtype == CATS_ST_Q8_0 || t->dtype == CATS_ST_Q3_0) {
		if (t->ndim != 2 || t->shape[1] % CATS_Q4_0_BLOCK) return -1;
		const int q8 = t->dtype == CATS_ST_Q8_0, q3 = t->dtype == CATS_ST_Q3_0;
		const size_t rb = (size_t)cats_st_bytes(t->dtype, t->shape[1]);
		cats_dq_ctx dc = { p, rb, (size_t)t->shape[1], dst, q8, q3 };
		cats_pfor(t->shape[0], 16, cats_st_dequant_rows, &dc);
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
	if (t->dtype == CATS_ST_Q4_0 || t->dtype == CATS_ST_Q8_0 || t->dtype == CATS_ST_Q3_0) {
		if (w % CATS_Q4_0_BLOCK) return -1;
		if (t->dtype == CATS_ST_Q3_0) cats_q3_0_dequantize_row(s->data + t->begin + (size_t)row * cats_q3_0_row_bytes((size_t)w), (size_t)w, dst);
		else if (t->dtype == CATS_ST_Q8_0) cats_q8_0w_dequantize_row(s->data + t->begin + (size_t)row * cats_q8_0w_row_bytes((size_t)w), (size_t)w, dst);
		else cats_q4_0_dequantize_row(s->data + t->begin + (size_t)row * cats_q4_0_row_bytes((size_t)w), (size_t)w, dst);
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

// C[n][out] = X[n][in] . W^T (+ bias) for a 2-D tensor of any supported dtype (F32, F16, BF16, Q4_0, Q8_0),
// used straight from the mapping. Quantized types keep 4 / 8-bit weights in memory for 1..3 tokens (see
// cats_gemm_nt_q4_0); 16 / 32-bit types are widened row by row (n < 4) or once into a scratch matrix (not
// thread-safe). Returns 0, or -1 for an unsupported tensor.
static inline int cats_st_gemm_nt(float *c, const float *x, const cats_st *s, const cats_st_tensor *t, size_t n, const float *bias)
{
	if (!t || t->ndim != 2 || t->dtype < 0) return -1;
	const size_t out = (size_t)t->shape[0], in = (size_t)t->shape[1];
	const uint8_t *w = cats_st_raw(s, t);
	if (t->dtype == CATS_ST_Q4_0) { if (in % CATS_Q4_0_BLOCK) return -1; cats_gemm_nt_q4_0(c, x, w, n, in, out, bias); return 0; }
	if (t->dtype == CATS_ST_Q8_0) { if (in % CATS_Q4_0_BLOCK) return -1; cats_gemm_nt_q8_0(c, x, w, n, in, out, bias); return 0; }
	if (t->dtype == CATS_ST_Q3_0) { if (in % CATS_Q4_0_BLOCK) return -1; cats_gemm_nt_q3_0(c, x, w, n, in, out, bias); return 0; }
	if (n >= CATS_Q4_0_DEQUANT_MIN) {
		static float *scratch;
		static size_t cap;
		if (cap < in * out) {
			(free)(scratch);
			scratch = (float*)(malloc)(in * out * sizeof(float));
			cap = scratch ? in * out : 0;
			if (!scratch) return -1;
		}
		if (cats_st_read_f32(s, t, scratch)) return -1;
		cats_gemm_nt(c, x, scratch, n, in, out, bias);
		return 0;
	}
	#pragma omp parallel
	{
		float *row = (float*)(malloc)(in * sizeof(float));
		#pragma omp for schedule(static)
		for (long r = 0; r < (long)out; r++) {
			if (!row || cats_st_read_row_f32(s, t, r, row)) continue;
			for (size_t i = 0; i < n; i++) c[i * out + r] = cats_dot_f32(x + i * in, row, in) + (bias ? bias[r] : 0.0f);
		}
		(free)(row);
	}
	return 0;
}

// Writes a safetensors file. Each entry points at its bytes (data), which are copied in order.
typedef struct {
	const char *name;
	const char *dtype;		// "F32", "F16", "BF16", "Q4_0" or "Q8_0"
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
typedef struct { float *out; const float *qp, *kp, *vt; size_t t; unsigned nq, hd; size_t nb; unsigned group; float scale; size_t window, qs; int fail; } cats_att_ctx;
static void cats_att_tasks(void *vp, long lo, long hi)
{
	cats_att_ctx *a = (cats_att_ctx*)vp;
	const size_t t = a->t, hd = a->hd, qs = a->qs, nb = a->nb, window = a->window;
	float *sc = (float*)(malloc)(CATS_ATT_ROWS * t * sizeof(float)), *ob = (float*)(malloc)(CATS_ATT_ROWS * hd * sizeof(float));
	if (!sc || !ob) { a->fail = 1; (free)(sc); (free)(ob); return; }
	for (long task = lo; task < hi; task++) {
		const unsigned h = (unsigned)(task / nb), g = h / a->group;
		const size_t i0 = (size_t)(task % nb) * CATS_ATT_ROWS, rows = t - i0 < CATS_ATT_ROWS ? t - i0 : CATS_ATT_ROWS;
		cats_gemm_nt_cpu(sc, a->qp + ((size_t)h * t + i0) * hd, a->kp + (size_t)g * t * hd, rows, hd, t, NULL);	// q_i . k_j
		for (size_t r = 0; r < rows; r++) {
			float *row = sc + r * t;
			const size_t i = i0 + r;
			size_t l0 = 0, h0 = t;
			if (window) {
				l0 = i + 1 > window ? i + 1 - window : 0;
				h0 = i + window < t ? i + window : t;
			}
			cats_softmax_scaled(row, t, l0, h0, a->scale);
		}
		cats_gemm_nt_cpu(ob, sc, a->vt + (size_t)g * hd * t, rows, t, hd, NULL);				// P . V
		for (size_t r = 0; r < rows; r++) memcpy(a->out + (i0 + r) * qs + (size_t)h * hd, ob + r * hd, hd * sizeof(float));
	}
	(free)(sc); (free)(ob);
}
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
	cats_att_ctx ac = { out, qp, kp, vt, t, nq, hd, nb, nq / nkv, scale, window, qs, 0 };
	cats_pfor((long)(nq * nb), 1, cats_att_tasks, &ac);
	int fail = ac.fail;
	(free)(qp); (free)(kp); (free)(vt);
	return fail ? -1 : 0;
}

// ---------------------------------------------------------------------------
// Decoder-LM primitives shared by the single-token (KV cache) inference programs
// ---------------------------------------------------------------------------
// RMSNorm whose learned scale is stored as a delta: dst = src / rms(src) * (offset + weight)
// (PLaMo 2/3, Gemma use offset 1.0 or a smaller constant; offset 0 equals cats_rmsnorm with the given eps).
static inline void cats_rmsnorm_offset(float *dst, const float *src, const float *weight, float offset, unsigned d, float eps)
{
	double sum = 0;
	for (unsigned i = 0; i < d; i++) sum += (double)src[i] * src[i];
	float scale = 1.0f / sqrtf((float)(sum / d) + eps);
	for (unsigned i = 0; i < d; i++) dst[i] = src[i] * scale * (offset + weight[i]);
}
// Inverse frequencies of RoPE with YaRN scaling (inv[dim/2]); returns the attention concentration that
// multiplies both cos and sin (0.1 ln(factor) + 1). factor <= 1 gives plain RoPE and a return value of 1.
//   orig_len: pretraining context length, beta_fast / beta_slow: rotation counts bounding the blend ramp.
static inline float cats_rope_inv_freq_yarn(float *inv, unsigned dim, float theta, float factor,
					    float orig_len, float beta_fast, float beta_slow)
{
	const unsigned h = dim / 2;
	const double two_pi = 6.283185307179586;
	double low = 0, high = 1;
	if (factor > 1.0f) {
		low  = h * log((double)orig_len / ((double)beta_fast * two_pi)) / log((double)theta);
		high = h * log((double)orig_len / ((double)beta_slow * two_pi)) / log((double)theta);
	}
	for (unsigned j = 0; j < h; j++) {
		double freq = pow((double)theta, 2.0 * j / dim), ex = 1.0 / freq;
		if (factor <= 1.0f) { inv[j] = (float)ex; continue; }
		double ramp = (j - low) / (high - low), m = 1.0 - (ramp < 0 ? 0 : ramp > 1 ? 1 : ramp);
		inv[j] = (float)((1.0 / (factor * freq)) * (1.0 - m) + ex * m);
	}
	return factor > 1.0f ? 0.1f * logf(factor) + 1.0f : 1.0f;
}
// cos / sin [dim] (rotate_half layout, usable with cats_rope_block) of one position from inverse frequencies
static inline void cats_rope_pos_table(float *cs, float *sn, size_t pos, const float *inv, float concentration, unsigned dim)
{
	const unsigned h = dim / 2;
	for (unsigned j = 0; j < h; j++) {
		double a = (double)pos * inv[j];
		cs[j] = cs[j + h] = (float)(cos(a) * concentration);
		sn[j] = sn[j + h] = (float)(sin(a) * concentration);
	}
}
// Causal attention of ONE query token (position pos) over cached keys / values of positions 0..pos with
// grouped K/V heads. q [nq*hd]; kc, vc [positions][nkv*hd]; out [nq*hd]; scratch needs nq*(pos+1) floats.
// window > 0 keeps only the last `window` positions (sliding window, the query included); 0 = full.
static inline void cats_attention_cached(float *out, const float *q, const float *kc, const float *vc, size_t pos,
					 unsigned nq, unsigned nkv, unsigned hd, float scale, size_t window, float *scratch)
{
	const size_t lo = window && pos + 1 > window ? pos + 1 - window : 0, ks = (size_t)nkv * hd, n = pos + 1;
	const unsigned group = nq / nkv;
	#pragma omp parallel for schedule(static) if(nq > 1 && n > 256)
	for (long h = 0; h < (long)nq; h++) {
		const float *qh = q + (size_t)h * hd, *kg = kc + (size_t)(h / group) * hd, *vg = vc + (size_t)(h / group) * hd;
		float *sc = scratch + (size_t)h * n, *o = out + (size_t)h * hd;
		for (size_t t = lo; t < n; t++) sc[t] = cats_dot_f32(qh, kg + t * ks, hd);
		cats_softmax_scaled(sc, n, lo, n, scale);
		for (unsigned d = 0; d < hd; d++) o[d] = 0;
		for (size_t t = lo; t < n; t++) {
			const float p = sc[t];
			const float *v = vg + t * ks;
			for (unsigned d = 0; d < hd; d++) o[d] += p * v[d];
		}
	}
}

// ---------------------------------------------------------------------------
// Unigram-LM tokenizer (SentencePiece-style Viterbi search over a scored vocabulary, byte fallback).
// Vocabulary file: one JSON array per line, ["piece", score, "NORMAL|BYTE|CONTROL|UNKNOWN|..."]; a BYTE piece is
// "<0xXX>". The best segmentation maximises the summed scores (scores are rounded to 1e-4, ties go to the longest
// piece); characters without a piece become their UTF-8 bytes. CONTROL pieces that appear verbatim in the text
// (e.g. <|plamo:tag|>) are emitted as single tokens and split the text there (parse_special).
// ---------------------------------------------------------------------------
typedef struct {
	int n;			// vocabulary size
	char **piece;		// UTF-8 bytes of each piece (NUL terminated, may hold embedded NULs: see plen)
	int *plen;
	int64_t *score;		// score * 1e4, rounded
	uint8_t *type;		// 0 NORMAL, 1 CONTROL, 2 BYTE, 3 UNKNOWN / other
	int byte_id[256];	// id of "<0xXX>" or -1
	int *table;		// open-addressing hash: piece bytes -> id + 1
	size_t tcap;
	int maxcp;		// longest piece in code points
	int *ctrl;		// ids of CONTROL pieces (special-token scan)
	int nctrl;
	int break_spaces;	// >0: a run of at least this many spaces is segmented on its own (0 = off)
	int break_repeat;	// >0: a run of at least this many equal characters (not "\n") is segmented on its own
} cats_unigram;

static inline uint32_t cats_unigram_hash(const char *s, int n)
{
	uint32_t h = 2166136261u;
	for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u;
	return h;
}
static inline int cats_unigram_lookup(const cats_unigram *u, const char *s, int n)
{
	size_t i = cats_unigram_hash(s, n) & (u->tcap - 1);
	while (u->table[i]) {
		int id = u->table[i] - 1;
		if (u->plen[id] == n && !memcmp(u->piece[id], s, (size_t)n)) return id;
		i = (i + 1) & (u->tcap - 1);
	}
	return -1;
}
static inline int cats_utf8_len(unsigned char c)
{
	return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
}
static inline int cats_utf8_put(char *o, unsigned c)
{
	if (c < 0x80) { o[0] = (char)c; return 1; }
	if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 63)); return 2; }
	if (c < 0x10000) { o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 63)); o[2] = (char)(0x80 | (c & 63)); return 3; }
	o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 63)); o[2] = (char)(0x80 | ((c >> 6) & 63)); o[3] = (char)(0x80 | (c & 63));
	return 4;
}
static inline void cats_unigram_free(cats_unigram *u)
{
	for (int i = 0; i < u->n; i++) (free)(u->piece[i]);
	(free)(u->piece); (free)(u->plen); (free)(u->score); (free)(u->type); (free)(u->table); (free)(u->ctrl);
	memset(u, 0, sizeof(*u));
}
// Parses one line ["piece", score, "TYPE"]; returns 0 on success.
static inline int cats_unigram_parse_line(const char *l, char *piece, int *plen, double *score, char *type, size_t tcap)
{
	const char *p = strchr(l, '"');
	if (!p) return -1;
	p++;
	int n = 0;
	while (*p && *p != '"') {
		if (*p != '\\') { piece[n++] = *p++; continue; }
		p++;
		switch (*p) {
		case 'n': piece[n++] = '\n'; p++; break;
		case 't': piece[n++] = '\t'; p++; break;
		case 'r': piece[n++] = '\r'; p++; break;
		case 'b': piece[n++] = '\b'; p++; break;
		case 'f': piece[n++] = '\f'; p++; break;
		case 'u': {
			unsigned c = (unsigned)strtoul((char[5]){p[1], p[2], p[3], p[4], 0}, NULL, 16);
			p += 5;
			if (c >= 0xD800 && c < 0xDC00 && p[0] == '\\' && p[1] == 'u') {	// surrogate pair
				unsigned lo = (unsigned)strtoul((char[5]){p[2], p[3], p[4], p[5], 0}, NULL, 16);
				c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
				p += 6;
			}
			n += cats_utf8_put(piece + n, c);
			break;
		}
		default: piece[n++] = *p++;	// \" \\ \/
		}
	}
	if (*p != '"') return -1;
	piece[n] = 0;
	*plen = n;
	p++;
	while (*p == ',' || *p == ' ') p++;
	*score = strtod(p, (char**)&p);
	type[0] = 0;
	const char *q = strchr(p, '"');
	if (q) {
		q++;
		size_t k = 0;
		while (q[k] && q[k] != '"' && k + 1 < tcap) { type[k] = q[k]; k++; }
		type[k] = 0;
	}
	return 0;
}
static inline int cats_unigram_load(cats_unigram *u, const char *path)
{
	memset(u, 0, sizeof(*u));
	FILE *f = fopen(path, "rb");
	if (!f) return -1;
	size_t cap = 1024, lcap = 1 << 16;
	char *line = (char*)(malloc)(lcap), *piece = (char*)(malloc)(lcap);
	u->piece = (char**)(malloc)(cap * sizeof(char*));
	u->plen = (int*)(malloc)(cap * sizeof(int));
	u->score = (int64_t*)(malloc)(cap * sizeof(int64_t));
	u->type = (uint8_t*)(malloc)(cap);
	if (!line || !piece || !u->piece || !u->plen || !u->score || !u->type) { fclose(f); return -1; }
	for (int i = 0; i < 256; i++) u->byte_id[i] = -1;
	while (fgets(line, (int)lcap, f)) {
		int pl; double sc; char ty[32];
		if (cats_unigram_parse_line(line, piece, &pl, &sc, ty, sizeof(ty))) continue;
		if ((size_t)u->n == cap) {
			cap *= 2;
			u->piece = (char**)(realloc)(u->piece, cap * sizeof(char*));
			u->plen = (int*)(realloc)(u->plen, cap * sizeof(int));
			u->score = (int64_t*)(realloc)(u->score, cap * sizeof(int64_t));
			u->type = (uint8_t*)(realloc)(u->type, cap);
			if (!u->piece || !u->plen || !u->score || !u->type) { fclose(f); return -1; }
		}
		int id = u->n++;
		u->piece[id] = (char*)(malloc)((size_t)pl + 1);
		memcpy(u->piece[id], piece, (size_t)pl + 1);
		u->plen[id] = pl;
		u->score[id] = (int64_t)llround(sc * 1e4);
		u->type[id] = !strcmp(ty, "NORMAL") ? 0 : !strcmp(ty, "CONTROL") ? 1 : !strcmp(ty, "BYTE") ? 2 : 3;
		if (u->type[id] == 2 && pl == 6 && !strncmp(piece, "<0x", 3)) u->byte_id[strtoul(piece + 3, NULL, 16) & 255] = id;
	}
	fclose(f);
	(free)(line); (free)(piece);
	u->tcap = 16;
	while (u->tcap < (size_t)u->n * 2) u->tcap <<= 1;
	u->table = (int*)(calloc)(u->tcap, sizeof(int));
	u->ctrl = (int*)(malloc)((size_t)u->n * sizeof(int));
	if (!u->table || !u->ctrl) return -1;
	for (int id = 0; id < u->n; id++) {
		if (u->type[id] == 2) continue;	// bytes are only a fallback
		size_t i = cats_unigram_hash(u->piece[id], u->plen[id]) & (u->tcap - 1);
		while (u->table[i] && !(u->plen[u->table[i] - 1] == u->plen[id] && !memcmp(u->piece[u->table[i] - 1], u->piece[id], (size_t)u->plen[id])))
			i = (i + 1) & (u->tcap - 1);
		u->table[i] = id + 1;			// a later duplicate replaces the earlier one
		int cp = 0;
		for (int k = 0; k < u->plen[id]; k += cats_utf8_len((unsigned char)u->piece[id][k])) cp++;
		if (cp > u->maxcp) u->maxcp = cp;
		if (u->type[id] == 1 && u->plen[id] > 2) u->ctrl[u->nctrl++] = id;
	}
	return u->n ? 0 : -1;
}
// Viterbi segmentation of text[0..len) as ONE segment. Returns the number of ids (may exceed cap) or -1.
static inline int cats_unigram_encode_seg(const cats_unigram *u, const char *text, size_t len, int *ids, int cap)
{
	if (!len) return 0;
	size_t *off = (size_t*)(malloc)((len + 2) * sizeof(size_t));
	int64_t *best = (int64_t*)(malloc)((len + 2) * sizeof(int64_t));
	int *clen = (int*)(malloc)((len + 2) * sizeof(int)), *cid = (int*)(malloc)((len + 2) * sizeof(int));
	if (!off || !best || !clen || !cid) { (free)(off); (free)(best); (free)(clen); (free)(cid); return -1; }
	size_t m = 0;
	for (size_t i = 0; i < len; m++) {
		off[m] = i;
		size_t l = (size_t)cats_utf8_len((unsigned char)text[i]);
		i += i + l <= len ? l : 1;
	}
	off[m] = len;
	const int64_t UNKNOWN_COST = 10000000;
	best[m] = 0;
	for (long i = (long)m - 1; i >= 0; i--) {
		best[i] = INT64_MAX;
		long lmax = (long)m - i < u->maxcp ? (long)m - i : u->maxcp;
		for (long l = lmax; l >= 1; l--) {
			int id = cats_unigram_lookup(u, text + off[i], (int)(off[i + l] - off[i]));
			if (id < 0) continue;
			int64_t c = best[i + l] - u->score[id];
			if (c < best[i]) { best[i] = c; clen[i] = (int)l; cid[i] = id; }
		}
		if (best[i + 1] + UNKNOWN_COST < best[i]) { best[i] = best[i + 1] + UNKNOWN_COST; clen[i] = 1; cid[i] = -1; }
	}
	int n = 0;
	for (size_t i = 0; i < m; i += (size_t)clen[i]) {
		if (cid[i] >= 0) { if (n < cap) ids[n] = cid[i]; n++; continue; }
		for (size_t b = off[i]; b < off[i + 1]; b++) {	// byte fallback
			int id = u->byte_id[(uint8_t)text[b]];
			if (n < cap) ids[n] = id < 0 ? 0 : id;
			n++;
		}
	}
	(free)(off); (free)(best); (free)(clen); (free)(cid);
	return n;
}
// Segmentation of text[0..len): runs selected by break_spaces / break_repeat are cut out and encoded alone, so the
// token boundaries do not shift with distant context (same rule as the PLaMo tokenizer).
static inline int cats_unigram_encode_plain(const cats_unigram *u, const char *text, size_t len, int *ids, int cap)
{
	int n = 0;
	size_t start = 0, i = 0;
	while (i <= len) {
		size_t run = 0, l = 0;
		if (i < len && (u->break_spaces || u->break_repeat)) {
			l = (size_t)cats_utf8_len((unsigned char)text[i]);
			if (i + l > len) l = 1;
			size_t j = i + l;
			run = 1;
			while (j + l <= len && !memcmp(text + j, text + i, l)) { j += l; run++; }
			int hit = (u->break_repeat && run >= (size_t)u->break_repeat && text[i] != '\n') ||
				  (u->break_spaces && text[i] == ' ' && run >= (size_t)u->break_spaces);
			if (!hit) { i += l; continue; }
			run = j - i;
		} else if (i < len) { i = len; continue; }
		int w = cats_unigram_encode_seg(u, text + start, i - start, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
		if (w < 0) return -1;
		n += w;
		if (i >= len) break;
		w = cats_unigram_encode_seg(u, text + i, run, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
		if (w < 0) return -1;
		n += w;
		i += run;
		start = i;
	}
	return n;
}
// Full encoding: CONTROL pieces found verbatim become single tokens (parse_special) and cut the text there.
// Returns the number of ids (may exceed cap, then only cap were stored) or -1 on error.
static inline int cats_unigram_encode(const cats_unigram *u, const char *text, int parse_special, int *ids, int cap)
{
	const size_t len = strlen(text);
	int n = 0;
	size_t start = 0;
	for (size_t i = 0; i <= len; ) {
		int hit = -1;
		if (parse_special && i < len && text[i] == '<') {
			for (int k = 0; k < u->nctrl; k++) {
				int id = u->ctrl[k];
				if (i + (size_t)u->plen[id] <= len && !memcmp(text + i, u->piece[id], (size_t)u->plen[id]) && (hit < 0 || u->plen[id] > u->plen[hit])) hit = id;
			}
		}
		if (hit < 0 && i < len) { i++; continue; }
		int w = cats_unigram_encode_plain(u, text + start, i - start, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
		if (w < 0) return -1;
		n += w;
		if (hit < 0) break;
		if (n < cap) ids[n] = hit;
		n++;
		i += (size_t)u->plen[hit];
		start = i;
	}
	return n;
}
// Appends the bytes of token `id` to out (cap bytes, NUL terminated); control tokens are skipped.
// Returns the bytes written.
static inline size_t cats_unigram_piece(const cats_unigram *u, int id, char *out, size_t cap)
{
	if (id < 0 || id >= u->n || u->type[id] == 1 || u->type[id] == 3) return 0;
	size_t n = 0;
	if (u->type[id] == 2) { if (cap > 1) out[n++] = (char)strtoul(u->piece[id] + 3, NULL, 16); }
	else { n = (size_t)u->plen[id] < cap - 1 ? (size_t)u->plen[id] : cap - 1; memcpy(out, u->piece[id], n); }
	out[n] = 0;
	return n;
}

// ---------------------------------------------------------------------------
// Low-precision float formats of recent checkpoints (NVIDIA ModelOpt): FP8 E4M3, E8M0 scales, FP4 E2M1 and the
// NVFP4 / MXFP8 block layouts, decoded row by row to float32.
// ---------------------------------------------------------------------------
static inline float cats_f8e4m3_to_float(uint8_t b)	// float8_e4m3fn
{
	const int e = (b >> 3) & 15, m = b & 7;
	if (e == 15 && m == 7) return NAN;
	const float v = e ? ldexpf(1.0f + (float)m / 8.0f, e - 7) : ldexpf((float)m / 8.0f, -6);
	return (b & 0x80) ? -v : v;
}
static inline float cats_e8m0_to_float(uint8_t b) { return ldexpf(1.0f, (int)b - 127); }
static const float cats_e2m1_value[16] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6 };
// NVFP4: n values (n % 16 == 0) as n / 2 bytes (the even element in the low nibble) and n / 16 E4M3 block scales;
// `factor` is the per-tensor scale (amax / outer scale denominator for ModelOpt).
static inline void cats_nvfp4_dequantize_row(const uint8_t *codes, const uint8_t *scales, size_t n, float factor, float *out)
{
	for (size_t b = 0; b < n / 16; b++) {
		const float s = cats_f8e4m3_to_float(scales[b]) * factor;
		for (int j = 0; j < 8; j++) {
			const uint8_t c = codes[b * 8 + j];
			out[b * 16 + 2 * j] = cats_e2m1_value[c & 15] * s;
			out[b * 16 + 2 * j + 1] = cats_e2m1_value[c >> 4] * s;
		}
	}
}
// MXFP8: n values (n % 32 == 0) as E4M3 bytes with one power-of-two scale (E8M0) per 32 values.
static inline void cats_mxfp8_dequantize_row(const uint8_t *v, const uint8_t *scales, size_t n, float *out)
{
	for (size_t b = 0; b < n / 32; b++) {
		const float s = cats_e8m0_to_float(scales[b]);
		for (int j = 0; j < 32; j++) out[b * 32 + j] = cats_f8e4m3_to_float(v[b * 32 + j]) * s;
	}
}

// ---- small helpers for gated / capped architectures ----
static inline float cats_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float cats_softcap(float x, float cap) { return tanhf(x / cap) * cap; }
// Scales x down to an L2 norm of at most `limit`; shorter vectors are unchanged.
static inline void cats_l2_clip(float *x, size_t n, float limit)
{
	double s = 0;
	for (size_t i = 0; i < n; i++) s += (double)x[i] * x[i];
	const float f = limit / ((float)sqrt(s) + 1e-6f);
	if (f < 1.0f) for (size_t i = 0; i < n; i++) x[i] *= f;
}

// ---------------------------------------------------------------------------
// Byte-level BPE tokenizer (GPT-2 / tiktoken style, e.g. Hugging Face tokenizer.json with a ByteLevel pre-tokenizer
// and the "o200k"-like split regex). The model is a compact blob (cats_bpe_from_json) so it can be stored inside a
// single model file:  "CBPE" u32 version, u32 n, u32 nmerge, u32 flags(1 = ignore_merges), then n x { u32 len,
// u8 special, bytes }, then nmerge x { u32 a, u32 b, u32 merged } in priority order. Token ids index the vocabulary.
// ---------------------------------------------------------------------------
// Code point classes (generated by tools/gen_unicode_classes.py): 1 Lu|Lt, 2 Ll, 3 Lm|Lo, 4 N, 5 white space, 6 M, else 0.
// generated by tools/gen_unicode_classes.py (Unicode 16.0.0): 2363 ranges
static const uint32_t cats_ucls_tab[][3] = {
	{0x9,0xD,5}, {0x20,0x20,5}, {0x30,0x39,4}, {0x41,0x5A,1}, {0x61,0x7A,2}, {0x85,0x85,5}, {0xA0,0xA0,5},
	{0xAA,0xAA,3}, {0xB2,0xB3,4}, {0xB5,0xB5,2}, {0xB9,0xB9,4}, {0xBA,0xBA,3}, {0xBC,0xBE,4}, {0xC0,0xD6,1},
	{0xD8,0xDE,1}, {0xDF,0xF6,2}, {0xF8,0xFF,2}, {0x100,0x100,1}, {0x101,0x101,2}, {0x102,0x102,1},
	{0x103,0x103,2}, {0x104,0x104,1}, {0x105,0x105,2}, {0x106,0x106,1}, {0x107,0x107,2}, {0x108,0x108,1},
	{0x109,0x109,2}, {0x10A,0x10A,1}, {0x10B,0x10B,2}, {0x10C,0x10C,1}, {0x10D,0x10D,2}, {0x10E,0x10E,1},
	{0x10F,0x10F,2}, {0x110,0x110,1}, {0x111,0x111,2}, {0x112,0x112,1}, {0x113,0x113,2}, {0x114,0x114,1},
	{0x115,0x115,2}, {0x116,0x116,1}, {0x117,0x117,2}, {0x118,0x118,1}, {0x119,0x119,2}, {0x11A,0x11A,1},
	{0x11B,0x11B,2}, {0x11C,0x11C,1}, {0x11D,0x11D,2}, {0x11E,0x11E,1}, {0x11F,0x11F,2}, {0x120,0x120,1},
	{0x121,0x121,2}, {0x122,0x122,1}, {0x123,0x123,2}, {0x124,0x124,1}, {0x125,0x125,2}, {0x126,0x126,1},
	{0x127,0x127,2}, {0x128,0x128,1}, {0x129,0x129,2}, {0x12A,0x12A,1}, {0x12B,0x12B,2}, {0x12C,0x12C,1},
	{0x12D,0x12D,2}, {0x12E,0x12E,1}, {0x12F,0x12F,2}, {0x130,0x130,1}, {0x131,0x131,2}, {0x132,0x132,1},
	{0x133,0x133,2}, {0x134,0x134,1}, {0x135,0x135,2}, {0x136,0x136,1}, {0x137,0x138,2}, {0x139,0x139,1},
	{0x13A,0x13A,2}, {0x13B,0x13B,1}, {0x13C,0x13C,2}, {0x13D,0x13D,1}, {0x13E,0x13E,2}, {0x13F,0x13F,1},
	{0x140,0x140,2}, {0x141,0x141,1}, {0x142,0x142,2}, {0x143,0x143,1}, {0x144,0x144,2}, {0x145,0x145,1},
	{0x146,0x146,2}, {0x147,0x147,1}, {0x148,0x149,2}, {0x14A,0x14A,1}, {0x14B,0x14B,2}, {0x14C,0x14C,1},
	{0x14D,0x14D,2}, {0x14E,0x14E,1}, {0x14F,0x14F,2}, {0x150,0x150,1}, {0x151,0x151,2}, {0x152,0x152,1},
	{0x153,0x153,2}, {0x154,0x154,1}, {0x155,0x155,2}, {0x156,0x156,1}, {0x157,0x157,2}, {0x158,0x158,1},
	{0x159,0x159,2}, {0x15A,0x15A,1}, {0x15B,0x15B,2}, {0x15C,0x15C,1}, {0x15D,0x15D,2}, {0x15E,0x15E,1},
	{0x15F,0x15F,2}, {0x160,0x160,1}, {0x161,0x161,2}, {0x162,0x162,1}, {0x163,0x163,2}, {0x164,0x164,1},
	{0x165,0x165,2}, {0x166,0x166,1}, {0x167,0x167,2}, {0x168,0x168,1}, {0x169,0x169,2}, {0x16A,0x16A,1},
	{0x16B,0x16B,2}, {0x16C,0x16C,1}, {0x16D,0x16D,2}, {0x16E,0x16E,1}, {0x16F,0x16F,2}, {0x170,0x170,1},
	{0x171,0x171,2}, {0x172,0x172,1}, {0x173,0x173,2}, {0x174,0x174,1}, {0x175,0x175,2}, {0x176,0x176,1},
	{0x177,0x177,2}, {0x178,0x179,1}, {0x17A,0x17A,2}, {0x17B,0x17B,1}, {0x17C,0x17C,2}, {0x17D,0x17D,1},
	{0x17E,0x180,2}, {0x181,0x182,1}, {0x183,0x183,2}, {0x184,0x184,1}, {0x185,0x185,2}, {0x186,0x187,1},
	{0x188,0x188,2}, {0x189,0x18B,1}, {0x18C,0x18D,2}, {0x18E,0x191,1}, {0x192,0x192,2}, {0x193,0x194,1},
	{0x195,0x195,2}, {0x196,0x198,1}, {0x199,0x19B,2}, {0x19C,0x19D,1}, {0x19E,0x19E,2}, {0x19F,0x1A0,1},
	{0x1A1,0x1A1,2}, {0x1A2,0x1A2,1}, {0x1A3,0x1A3,2}, {0x1A4,0x1A4,1}, {0x1A5,0x1A5,2}, {0x1A6,0x1A7,1},
	{0x1A8,0x1A8,2}, {0x1A9,0x1A9,1}, {0x1AA,0x1AB,2}, {0x1AC,0x1AC,1}, {0x1AD,0x1AD,2}, {0x1AE,0x1AF,1},
	{0x1B0,0x1B0,2}, {0x1B1,0x1B3,1}, {0x1B4,0x1B4,2}, {0x1B5,0x1B5,1}, {0x1B6,0x1B6,2}, {0x1B7,0x1B8,1},
	{0x1B9,0x1BA,2}, {0x1BB,0x1BB,3}, {0x1BC,0x1BC,1}, {0x1BD,0x1BF,2}, {0x1C0,0x1C3,3}, {0x1C4,0x1C5,1},
	{0x1C6,0x1C6,2}, {0x1C7,0x1C8,1}, {0x1C9,0x1C9,2}, {0x1CA,0x1CB,1}, {0x1CC,0x1CC,2}, {0x1CD,0x1CD,1},
	{0x1CE,0x1CE,2}, {0x1CF,0x1CF,1}, {0x1D0,0x1D0,2}, {0x1D1,0x1D1,1}, {0x1D2,0x1D2,2}, {0x1D3,0x1D3,1},
	{0x1D4,0x1D4,2}, {0x1D5,0x1D5,1}, {0x1D6,0x1D6,2}, {0x1D7,0x1D7,1}, {0x1D8,0x1D8,2}, {0x1D9,0x1D9,1},
	{0x1DA,0x1DA,2}, {0x1DB,0x1DB,1}, {0x1DC,0x1DD,2}, {0x1DE,0x1DE,1}, {0x1DF,0x1DF,2}, {0x1E0,0x1E0,1},
	{0x1E1,0x1E1,2}, {0x1E2,0x1E2,1}, {0x1E3,0x1E3,2}, {0x1E4,0x1E4,1}, {0x1E5,0x1E5,2}, {0x1E6,0x1E6,1},
	{0x1E7,0x1E7,2}, {0x1E8,0x1E8,1}, {0x1E9,0x1E9,2}, {0x1EA,0x1EA,1}, {0x1EB,0x1EB,2}, {0x1EC,0x1EC,1},
	{0x1ED,0x1ED,2}, {0x1EE,0x1EE,1}, {0x1EF,0x1F0,2}, {0x1F1,0x1F2,1}, {0x1F3,0x1F3,2}, {0x1F4,0x1F4,1},
	{0x1F5,0x1F5,2}, {0x1F6,0x1F8,1}, {0x1F9,0x1F9,2}, {0x1FA,0x1FA,1}, {0x1FB,0x1FB,2}, {0x1FC,0x1FC,1},
	{0x1FD,0x1FD,2}, {0x1FE,0x1FE,1}, {0x1FF,0x1FF,2}, {0x200,0x200,1}, {0x201,0x201,2}, {0x202,0x202,1},
	{0x203,0x203,2}, {0x204,0x204,1}, {0x205,0x205,2}, {0x206,0x206,1}, {0x207,0x207,2}, {0x208,0x208,1},
	{0x209,0x209,2}, {0x20A,0x20A,1}, {0x20B,0x20B,2}, {0x20C,0x20C,1}, {0x20D,0x20D,2}, {0x20E,0x20E,1},
	{0x20F,0x20F,2}, {0x210,0x210,1}, {0x211,0x211,2}, {0x212,0x212,1}, {0x213,0x213,2}, {0x214,0x214,1},
	{0x215,0x215,2}, {0x216,0x216,1}, {0x217,0x217,2}, {0x218,0x218,1}, {0x219,0x219,2}, {0x21A,0x21A,1},
	{0x21B,0x21B,2}, {0x21C,0x21C,1}, {0x21D,0x21D,2}, {0x21E,0x21E,1}, {0x21F,0x21F,2}, {0x220,0x220,1},
	{0x221,0x221,2}, {0x222,0x222,1}, {0x223,0x223,2}, {0x224,0x224,1}, {0x225,0x225,2}, {0x226,0x226,1},
	{0x227,0x227,2}, {0x228,0x228,1}, {0x229,0x229,2}, {0x22A,0x22A,1}, {0x22B,0x22B,2}, {0x22C,0x22C,1},
	{0x22D,0x22D,2}, {0x22E,0x22E,1}, {0x22F,0x22F,2}, {0x230,0x230,1}, {0x231,0x231,2}, {0x232,0x232,1},
	{0x233,0x239,2}, {0x23A,0x23B,1}, {0x23C,0x23C,2}, {0x23D,0x23E,1}, {0x23F,0x240,2}, {0x241,0x241,1},
	{0x242,0x242,2}, {0x243,0x246,1}, {0x247,0x247,2}, {0x248,0x248,1}, {0x249,0x249,2}, {0x24A,0x24A,1},
	{0x24B,0x24B,2}, {0x24C,0x24C,1}, {0x24D,0x24D,2}, {0x24E,0x24E,1}, {0x24F,0x293,2}, {0x294,0x294,3},
	{0x295,0x2AF,2}, {0x2B0,0x2C1,3}, {0x2C6,0x2D1,3}, {0x2E0,0x2E4,3}, {0x2EC,0x2EC,3}, {0x2EE,0x2EE,3},
	{0x300,0x36F,6}, {0x370,0x370,1}, {0x371,0x371,2}, {0x372,0x372,1}, {0x373,0x373,2}, {0x374,0x374,3},
	{0x376,0x376,1}, {0x377,0x377,2}, {0x37A,0x37A,3}, {0x37B,0x37D,2}, {0x37F,0x37F,1}, {0x386,0x386,1},
	{0x388,0x38A,1}, {0x38C,0x38C,1}, {0x38E,0x38F,1}, {0x390,0x390,2}, {0x391,0x3A1,1}, {0x3A3,0x3AB,1},
	{0x3AC,0x3CE,2}, {0x3CF,0x3CF,1}, {0x3D0,0x3D1,2}, {0x3D2,0x3D4,1}, {0x3D5,0x3D7,2}, {0x3D8,0x3D8,1},
	{0x3D9,0x3D9,2}, {0x3DA,0x3DA,1}, {0x3DB,0x3DB,2}, {0x3DC,0x3DC,1}, {0x3DD,0x3DD,2}, {0x3DE,0x3DE,1},
	{0x3DF,0x3DF,2}, {0x3E0,0x3E0,1}, {0x3E1,0x3E1,2}, {0x3E2,0x3E2,1}, {0x3E3,0x3E3,2}, {0x3E4,0x3E4,1},
	{0x3E5,0x3E5,2}, {0x3E6,0x3E6,1}, {0x3E7,0x3E7,2}, {0x3E8,0x3E8,1}, {0x3E9,0x3E9,2}, {0x3EA,0x3EA,1},
	{0x3EB,0x3EB,2}, {0x3EC,0x3EC,1}, {0x3ED,0x3ED,2}, {0x3EE,0x3EE,1}, {0x3EF,0x3F3,2}, {0x3F4,0x3F4,1},
	{0x3F5,0x3F5,2}, {0x3F7,0x3F7,1}, {0x3F8,0x3F8,2}, {0x3F9,0x3FA,1}, {0x3FB,0x3FC,2}, {0x3FD,0x42F,1},
	{0x430,0x45F,2}, {0x460,0x460,1}, {0x461,0x461,2}, {0x462,0x462,1}, {0x463,0x463,2}, {0x464,0x464,1},
	{0x465,0x465,2}, {0x466,0x466,1}, {0x467,0x467,2}, {0x468,0x468,1}, {0x469,0x469,2}, {0x46A,0x46A,1},
	{0x46B,0x46B,2}, {0x46C,0x46C,1}, {0x46D,0x46D,2}, {0x46E,0x46E,1}, {0x46F,0x46F,2}, {0x470,0x470,1},
	{0x471,0x471,2}, {0x472,0x472,1}, {0x473,0x473,2}, {0x474,0x474,1}, {0x475,0x475,2}, {0x476,0x476,1},
	{0x477,0x477,2}, {0x478,0x478,1}, {0x479,0x479,2}, {0x47A,0x47A,1}, {0x47B,0x47B,2}, {0x47C,0x47C,1},
	{0x47D,0x47D,2}, {0x47E,0x47E,1}, {0x47F,0x47F,2}, {0x480,0x480,1}, {0x481,0x481,2}, {0x483,0x489,6},
	{0x48A,0x48A,1}, {0x48B,0x48B,2}, {0x48C,0x48C,1}, {0x48D,0x48D,2}, {0x48E,0x48E,1}, {0x48F,0x48F,2},
	{0x490,0x490,1}, {0x491,0x491,2}, {0x492,0x492,1}, {0x493,0x493,2}, {0x494,0x494,1}, {0x495,0x495,2},
	{0x496,0x496,1}, {0x497,0x497,2}, {0x498,0x498,1}, {0x499,0x499,2}, {0x49A,0x49A,1}, {0x49B,0x49B,2},
	{0x49C,0x49C,1}, {0x49D,0x49D,2}, {0x49E,0x49E,1}, {0x49F,0x49F,2}, {0x4A0,0x4A0,1}, {0x4A1,0x4A1,2},
	{0x4A2,0x4A2,1}, {0x4A3,0x4A3,2}, {0x4A4,0x4A4,1}, {0x4A5,0x4A5,2}, {0x4A6,0x4A6,1}, {0x4A7,0x4A7,2},
	{0x4A8,0x4A8,1}, {0x4A9,0x4A9,2}, {0x4AA,0x4AA,1}, {0x4AB,0x4AB,2}, {0x4AC,0x4AC,1}, {0x4AD,0x4AD,2},
	{0x4AE,0x4AE,1}, {0x4AF,0x4AF,2}, {0x4B0,0x4B0,1}, {0x4B1,0x4B1,2}, {0x4B2,0x4B2,1}, {0x4B3,0x4B3,2},
	{0x4B4,0x4B4,1}, {0x4B5,0x4B5,2}, {0x4B6,0x4B6,1}, {0x4B7,0x4B7,2}, {0x4B8,0x4B8,1}, {0x4B9,0x4B9,2},
	{0x4BA,0x4BA,1}, {0x4BB,0x4BB,2}, {0x4BC,0x4BC,1}, {0x4BD,0x4BD,2}, {0x4BE,0x4BE,1}, {0x4BF,0x4BF,2},
	{0x4C0,0x4C1,1}, {0x4C2,0x4C2,2}, {0x4C3,0x4C3,1}, {0x4C4,0x4C4,2}, {0x4C5,0x4C5,1}, {0x4C6,0x4C6,2},
	{0x4C7,0x4C7,1}, {0x4C8,0x4C8,2}, {0x4C9,0x4C9,1}, {0x4CA,0x4CA,2}, {0x4CB,0x4CB,1}, {0x4CC,0x4CC,2},
	{0x4CD,0x4CD,1}, {0x4CE,0x4CF,2}, {0x4D0,0x4D0,1}, {0x4D1,0x4D1,2}, {0x4D2,0x4D2,1}, {0x4D3,0x4D3,2},
	{0x4D4,0x4D4,1}, {0x4D5,0x4D5,2}, {0x4D6,0x4D6,1}, {0x4D7,0x4D7,2}, {0x4D8,0x4D8,1}, {0x4D9,0x4D9,2},
	{0x4DA,0x4DA,1}, {0x4DB,0x4DB,2}, {0x4DC,0x4DC,1}, {0x4DD,0x4DD,2}, {0x4DE,0x4DE,1}, {0x4DF,0x4DF,2},
	{0x4E0,0x4E0,1}, {0x4E1,0x4E1,2}, {0x4E2,0x4E2,1}, {0x4E3,0x4E3,2}, {0x4E4,0x4E4,1}, {0x4E5,0x4E5,2},
	{0x4E6,0x4E6,1}, {0x4E7,0x4E7,2}, {0x4E8,0x4E8,1}, {0x4E9,0x4E9,2}, {0x4EA,0x4EA,1}, {0x4EB,0x4EB,2},
	{0x4EC,0x4EC,1}, {0x4ED,0x4ED,2}, {0x4EE,0x4EE,1}, {0x4EF,0x4EF,2}, {0x4F0,0x4F0,1}, {0x4F1,0x4F1,2},
	{0x4F2,0x4F2,1}, {0x4F3,0x4F3,2}, {0x4F4,0x4F4,1}, {0x4F5,0x4F5,2}, {0x4F6,0x4F6,1}, {0x4F7,0x4F7,2},
	{0x4F8,0x4F8,1}, {0x4F9,0x4F9,2}, {0x4FA,0x4FA,1}, {0x4FB,0x4FB,2}, {0x4FC,0x4FC,1}, {0x4FD,0x4FD,2},
	{0x4FE,0x4FE,1}, {0x4FF,0x4FF,2}, {0x500,0x500,1}, {0x501,0x501,2}, {0x502,0x502,1}, {0x503,0x503,2},
	{0x504,0x504,1}, {0x505,0x505,2}, {0x506,0x506,1}, {0x507,0x507,2}, {0x508,0x508,1}, {0x509,0x509,2},
	{0x50A,0x50A,1}, {0x50B,0x50B,2}, {0x50C,0x50C,1}, {0x50D,0x50D,2}, {0x50E,0x50E,1}, {0x50F,0x50F,2},
	{0x510,0x510,1}, {0x511,0x511,2}, {0x512,0x512,1}, {0x513,0x513,2}, {0x514,0x514,1}, {0x515,0x515,2},
	{0x516,0x516,1}, {0x517,0x517,2}, {0x518,0x518,1}, {0x519,0x519,2}, {0x51A,0x51A,1}, {0x51B,0x51B,2},
	{0x51C,0x51C,1}, {0x51D,0x51D,2}, {0x51E,0x51E,1}, {0x51F,0x51F,2}, {0x520,0x520,1}, {0x521,0x521,2},
	{0x522,0x522,1}, {0x523,0x523,2}, {0x524,0x524,1}, {0x525,0x525,2}, {0x526,0x526,1}, {0x527,0x527,2},
	{0x528,0x528,1}, {0x529,0x529,2}, {0x52A,0x52A,1}, {0x52B,0x52B,2}, {0x52C,0x52C,1}, {0x52D,0x52D,2},
	{0x52E,0x52E,1}, {0x52F,0x52F,2}, {0x531,0x556,1}, {0x559,0x559,3}, {0x560,0x588,2}, {0x591,0x5BD,6},
	{0x5BF,0x5BF,6}, {0x5C1,0x5C2,6}, {0x5C4,0x5C5,6}, {0x5C7,0x5C7,6}, {0x5D0,0x5EA,3}, {0x5EF,0x5F2,3},
	{0x610,0x61A,6}, {0x620,0x64A,3}, {0x64B,0x65F,6}, {0x660,0x669,4}, {0x66E,0x66F,3}, {0x670,0x670,6},
	{0x671,0x6D3,3}, {0x6D5,0x6D5,3}, {0x6D6,0x6DC,6}, {0x6DF,0x6E4,6}, {0x6E5,0x6E6,3}, {0x6E7,0x6E8,6},
	{0x6EA,0x6ED,6}, {0x6EE,0x6EF,3}, {0x6F0,0x6F9,4}, {0x6FA,0x6FC,3}, {0x6FF,0x6FF,3}, {0x710,0x710,3},
	{0x711,0x711,6}, {0x712,0x72F,3}, {0x730,0x74A,6}, {0x74D,0x7A5,3}, {0x7A6,0x7B0,6}, {0x7B1,0x7B1,3},
	{0x7C0,0x7C9,4}, {0x7CA,0x7EA,3}, {0x7EB,0x7F3,6}, {0x7F4,0x7F5,3}, {0x7FA,0x7FA,3}, {0x7FD,0x7FD,6},
	{0x800,0x815,3}, {0x816,0x819,6}, {0x81A,0x81A,3}, {0x81B,0x823,6}, {0x824,0x824,3}, {0x825,0x827,6},
	{0x828,0x828,3}, {0x829,0x82D,6}, {0x840,0x858,3}, {0x859,0x85B,6}, {0x860,0x86A,3}, {0x870,0x887,3},
	{0x889,0x88E,3}, {0x897,0x89F,6}, {0x8A0,0x8C9,3}, {0x8CA,0x8E1,6}, {0x8E3,0x903,6}, {0x904,0x939,3},
	{0x93A,0x93C,6}, {0x93D,0x93D,3}, {0x93E,0x94F,6}, {0x950,0x950,3}, {0x951,0x957,6}, {0x958,0x961,3},
	{0x962,0x963,6}, {0x966,0x96F,4}, {0x971,0x980,3}, {0x981,0x983,6}, {0x985,0x98C,3}, {0x98F,0x990,3},
	{0x993,0x9A8,3}, {0x9AA,0x9B0,3}, {0x9B2,0x9B2,3}, {0x9B6,0x9B9,3}, {0x9BC,0x9BC,6}, {0x9BD,0x9BD,3},
	{0x9BE,0x9C4,6}, {0x9C7,0x9C8,6}, {0x9CB,0x9CD,6}, {0x9CE,0x9CE,3}, {0x9D7,0x9D7,6}, {0x9DC,0x9DD,3},
	{0x9DF,0x9E1,3}, {0x9E2,0x9E3,6}, {0x9E6,0x9EF,4}, {0x9F0,0x9F1,3}, {0x9F4,0x9F9,4}, {0x9FC,0x9FC,3},
	{0x9FE,0x9FE,6}, {0xA01,0xA03,6}, {0xA05,0xA0A,3}, {0xA0F,0xA10,3}, {0xA13,0xA28,3}, {0xA2A,0xA30,3},
	{0xA32,0xA33,3}, {0xA35,0xA36,3}, {0xA38,0xA39,3}, {0xA3C,0xA3C,6}, {0xA3E,0xA42,6}, {0xA47,0xA48,6},
	{0xA4B,0xA4D,6}, {0xA51,0xA51,6}, {0xA59,0xA5C,3}, {0xA5E,0xA5E,3}, {0xA66,0xA6F,4}, {0xA70,0xA71,6},
	{0xA72,0xA74,3}, {0xA75,0xA75,6}, {0xA81,0xA83,6}, {0xA85,0xA8D,3}, {0xA8F,0xA91,3}, {0xA93,0xAA8,3},
	{0xAAA,0xAB0,3}, {0xAB2,0xAB3,3}, {0xAB5,0xAB9,3}, {0xABC,0xABC,6}, {0xABD,0xABD,3}, {0xABE,0xAC5,6},
	{0xAC7,0xAC9,6}, {0xACB,0xACD,6}, {0xAD0,0xAD0,3}, {0xAE0,0xAE1,3}, {0xAE2,0xAE3,6}, {0xAE6,0xAEF,4},
	{0xAF9,0xAF9,3}, {0xAFA,0xAFF,6}, {0xB01,0xB03,6}, {0xB05,0xB0C,3}, {0xB0F,0xB10,3}, {0xB13,0xB28,3},
	{0xB2A,0xB30,3}, {0xB32,0xB33,3}, {0xB35,0xB39,3}, {0xB3C,0xB3C,6}, {0xB3D,0xB3D,3}, {0xB3E,0xB44,6},
	{0xB47,0xB48,6}, {0xB4B,0xB4D,6}, {0xB55,0xB57,6}, {0xB5C,0xB5D,3}, {0xB5F,0xB61,3}, {0xB62,0xB63,6},
	{0xB66,0xB6F,4}, {0xB71,0xB71,3}, {0xB72,0xB77,4}, {0xB82,0xB82,6}, {0xB83,0xB83,3}, {0xB85,0xB8A,3},
	{0xB8E,0xB90,3}, {0xB92,0xB95,3}, {0xB99,0xB9A,3}, {0xB9C,0xB9C,3}, {0xB9E,0xB9F,3}, {0xBA3,0xBA4,3},
	{0xBA8,0xBAA,3}, {0xBAE,0xBB9,3}, {0xBBE,0xBC2,6}, {0xBC6,0xBC8,6}, {0xBCA,0xBCD,6}, {0xBD0,0xBD0,3},
	{0xBD7,0xBD7,6}, {0xBE6,0xBF2,4}, {0xC00,0xC04,6}, {0xC05,0xC0C,3}, {0xC0E,0xC10,3}, {0xC12,0xC28,3},
	{0xC2A,0xC39,3}, {0xC3C,0xC3C,6}, {0xC3D,0xC3D,3}, {0xC3E,0xC44,6}, {0xC46,0xC48,6}, {0xC4A,0xC4D,6},
	{0xC55,0xC56,6}, {0xC58,0xC5A,3}, {0xC5D,0xC5D,3}, {0xC60,0xC61,3}, {0xC62,0xC63,6}, {0xC66,0xC6F,4},
	{0xC78,0xC7E,4}, {0xC80,0xC80,3}, {0xC81,0xC83,6}, {0xC85,0xC8C,3}, {0xC8E,0xC90,3}, {0xC92,0xCA8,3},
	{0xCAA,0xCB3,3}, {0xCB5,0xCB9,3}, {0xCBC,0xCBC,6}, {0xCBD,0xCBD,3}, {0xCBE,0xCC4,6}, {0xCC6,0xCC8,6},
	{0xCCA,0xCCD,6}, {0xCD5,0xCD6,6}, {0xCDD,0xCDE,3}, {0xCE0,0xCE1,3}, {0xCE2,0xCE3,6}, {0xCE6,0xCEF,4},
	{0xCF1,0xCF2,3}, {0xCF3,0xCF3,6}, {0xD00,0xD03,6}, {0xD04,0xD0C,3}, {0xD0E,0xD10,3}, {0xD12,0xD3A,3},
	{0xD3B,0xD3C,6}, {0xD3D,0xD3D,3}, {0xD3E,0xD44,6}, {0xD46,0xD48,6}, {0xD4A,0xD4D,6}, {0xD4E,0xD4E,3},
	{0xD54,0xD56,3}, {0xD57,0xD57,6}, {0xD58,0xD5E,4}, {0xD5F,0xD61,3}, {0xD62,0xD63,6}, {0xD66,0xD78,4},
	{0xD7A,0xD7F,3}, {0xD81,0xD83,6}, {0xD85,0xD96,3}, {0xD9A,0xDB1,3}, {0xDB3,0xDBB,3}, {0xDBD,0xDBD,3},
	{0xDC0,0xDC6,3}, {0xDCA,0xDCA,6}, {0xDCF,0xDD4,6}, {0xDD6,0xDD6,6}, {0xDD8,0xDDF,6}, {0xDE6,0xDEF,4},
	{0xDF2,0xDF3,6}, {0xE01,0xE30,3}, {0xE31,0xE31,6}, {0xE32,0xE33,3}, {0xE34,0xE3A,6}, {0xE40,0xE46,3},
	{0xE47,0xE4E,6}, {0xE50,0xE59,4}, {0xE81,0xE82,3}, {0xE84,0xE84,3}, {0xE86,0xE8A,3}, {0xE8C,0xEA3,3},
	{0xEA5,0xEA5,3}, {0xEA7,0xEB0,3}, {0xEB1,0xEB1,6}, {0xEB2,0xEB3,3}, {0xEB4,0xEBC,6}, {0xEBD,0xEBD,3},
	{0xEC0,0xEC4,3}, {0xEC6,0xEC6,3}, {0xEC8,0xECE,6}, {0xED0,0xED9,4}, {0xEDC,0xEDF,3}, {0xF00,0xF00,3},
	{0xF18,0xF19,6}, {0xF20,0xF33,4}, {0xF35,0xF35,6}, {0xF37,0xF37,6}, {0xF39,0xF39,6}, {0xF3E,0xF3F,6},
	{0xF40,0xF47,3}, {0xF49,0xF6C,3}, {0xF71,0xF84,6}, {0xF86,0xF87,6}, {0xF88,0xF8C,3}, {0xF8D,0xF97,6},
	{0xF99,0xFBC,6}, {0xFC6,0xFC6,6}, {0x1000,0x102A,3}, {0x102B,0x103E,6}, {0x103F,0x103F,3}, {0x1040,0x1049,4},
	{0x1050,0x1055,3}, {0x1056,0x1059,6}, {0x105A,0x105D,3}, {0x105E,0x1060,6}, {0x1061,0x1061,3},
	{0x1062,0x1064,6}, {0x1065,0x1066,3}, {0x1067,0x106D,6}, {0x106E,0x1070,3}, {0x1071,0x1074,6},
	{0x1075,0x1081,3}, {0x1082,0x108D,6}, {0x108E,0x108E,3}, {0x108F,0x108F,6}, {0x1090,0x1099,4},
	{0x109A,0x109D,6}, {0x10A0,0x10C5,1}, {0x10C7,0x10C7,1}, {0x10CD,0x10CD,1}, {0x10D0,0x10FA,2},
	{0x10FC,0x10FC,3}, {0x10FD,0x10FF,2}, {0x1100,0x1248,3}, {0x124A,0x124D,3}, {0x1250,0x1256,3},
	{0x1258,0x1258,3}, {0x125A,0x125D,3}, {0x1260,0x1288,3}, {0x128A,0x128D,3}, {0x1290,0x12B0,3},
	{0x12B2,0x12B5,3}, {0x12B8,0x12BE,3}, {0x12C0,0x12C0,3}, {0x12C2,0x12C5,3}, {0x12C8,0x12D6,3},
	{0x12D8,0x1310,3}, {0x1312,0x1315,3}, {0x1318,0x135A,3}, {0x135D,0x135F,6}, {0x1369,0x137C,4},
	{0x1380,0x138F,3}, {0x13A0,0x13F5,1}, {0x13F8,0x13FD,2}, {0x1401,0x166C,3}, {0x166F,0x167F,3},
	{0x1680,0x1680,5}, {0x1681,0x169A,3}, {0x16A0,0x16EA,3}, {0x16EE,0x16F0,4}, {0x16F1,0x16F8,3},
	{0x1700,0x1711,3}, {0x1712,0x1715,6}, {0x171F,0x1731,3}, {0x1732,0x1734,6}, {0x1740,0x1751,3},
	{0x1752,0x1753,6}, {0x1760,0x176C,3}, {0x176E,0x1770,3}, {0x1772,0x1773,6}, {0x1780,0x17B3,3},
	{0x17B4,0x17D3,6}, {0x17D7,0x17D7,3}, {0x17DC,0x17DC,3}, {0x17DD,0x17DD,6}, {0x17E0,0x17E9,4},
	{0x17F0,0x17F9,4}, {0x180B,0x180D,6}, {0x180F,0x180F,6}, {0x1810,0x1819,4}, {0x1820,0x1878,3},
	{0x1880,0x1884,3}, {0x1885,0x1886,6}, {0x1887,0x18A8,3}, {0x18A9,0x18A9,6}, {0x18AA,0x18AA,3},
	{0x18B0,0x18F5,3}, {0x1900,0x191E,3}, {0x1920,0x192B,6}, {0x1930,0x193B,6}, {0x1946,0x194F,4},
	{0x1950,0x196D,3}, {0x1970,0x1974,3}, {0x1980,0x19AB,3}, {0x19B0,0x19C9,3}, {0x19D0,0x19DA,4},
	{0x1A00,0x1A16,3}, {0x1A17,0x1A1B,6}, {0x1A20,0x1A54,3}, {0x1A55,0x1A5E,6}, {0x1A60,0x1A7C,6},
	{0x1A7F,0x1A7F,6}, {0x1A80,0x1A89,4}, {0x1A90,0x1A99,4}, {0x1AA7,0x1AA7,3}, {0x1AB0,0x1ACE,6},
	{0x1B00,0x1B04,6}, {0x1B05,0x1B33,3}, {0x1B34,0x1B44,6}, {0x1B45,0x1B4C,3}, {0x1B50,0x1B59,4},
	{0x1B6B,0x1B73,6}, {0x1B80,0x1B82,6}, {0x1B83,0x1BA0,3}, {0x1BA1,0x1BAD,6}, {0x1BAE,0x1BAF,3},
	{0x1BB0,0x1BB9,4}, {0x1BBA,0x1BE5,3}, {0x1BE6,0x1BF3,6}, {0x1C00,0x1C23,3}, {0x1C24,0x1C37,6},
	{0x1C40,0x1C49,4}, {0x1C4D,0x1C4F,3}, {0x1C50,0x1C59,4}, {0x1C5A,0x1C7D,3}, {0x1C80,0x1C88,2},
	{0x1C89,0x1C89,1}, {0x1C8A,0x1C8A,2}, {0x1C90,0x1CBA,1}, {0x1CBD,0x1CBF,1}, {0x1CD0,0x1CD2,6},
	{0x1CD4,0x1CE8,6}, {0x1CE9,0x1CEC,3}, {0x1CED,0x1CED,6}, {0x1CEE,0x1CF3,3}, {0x1CF4,0x1CF4,6},
	{0x1CF5,0x1CF6,3}, {0x1CF7,0x1CF9,6}, {0x1CFA,0x1CFA,3}, {0x1D00,0x1D2B,2}, {0x1D2C,0x1D6A,3},
	{0x1D6B,0x1D77,2}, {0x1D78,0x1D78,3}, {0x1D79,0x1D9A,2}, {0x1D9B,0x1DBF,3}, {0x1DC0,0x1DFF,6},
	{0x1E00,0x1E00,1}, {0x1E01,0x1E01,2}, {0x1E02,0x1E02,1}, {0x1E03,0x1E03,2}, {0x1E04,0x1E04,1},
	{0x1E05,0x1E05,2}, {0x1E06,0x1E06,1}, {0x1E07,0x1E07,2}, {0x1E08,0x1E08,1}, {0x1E09,0x1E09,2},
	{0x1E0A,0x1E0A,1}, {0x1E0B,0x1E0B,2}, {0x1E0C,0x1E0C,1}, {0x1E0D,0x1E0D,2}, {0x1E0E,0x1E0E,1},
	{0x1E0F,0x1E0F,2}, {0x1E10,0x1E10,1}, {0x1E11,0x1E11,2}, {0x1E12,0x1E12,1}, {0x1E13,0x1E13,2},
	{0x1E14,0x1E14,1}, {0x1E15,0x1E15,2}, {0x1E16,0x1E16,1}, {0x1E17,0x1E17,2}, {0x1E18,0x1E18,1},
	{0x1E19,0x1E19,2}, {0x1E1A,0x1E1A,1}, {0x1E1B,0x1E1B,2}, {0x1E1C,0x1E1C,1}, {0x1E1D,0x1E1D,2},
	{0x1E1E,0x1E1E,1}, {0x1E1F,0x1E1F,2}, {0x1E20,0x1E20,1}, {0x1E21,0x1E21,2}, {0x1E22,0x1E22,1},
	{0x1E23,0x1E23,2}, {0x1E24,0x1E24,1}, {0x1E25,0x1E25,2}, {0x1E26,0x1E26,1}, {0x1E27,0x1E27,2},
	{0x1E28,0x1E28,1}, {0x1E29,0x1E29,2}, {0x1E2A,0x1E2A,1}, {0x1E2B,0x1E2B,2}, {0x1E2C,0x1E2C,1},
	{0x1E2D,0x1E2D,2}, {0x1E2E,0x1E2E,1}, {0x1E2F,0x1E2F,2}, {0x1E30,0x1E30,1}, {0x1E31,0x1E31,2},
	{0x1E32,0x1E32,1}, {0x1E33,0x1E33,2}, {0x1E34,0x1E34,1}, {0x1E35,0x1E35,2}, {0x1E36,0x1E36,1},
	{0x1E37,0x1E37,2}, {0x1E38,0x1E38,1}, {0x1E39,0x1E39,2}, {0x1E3A,0x1E3A,1}, {0x1E3B,0x1E3B,2},
	{0x1E3C,0x1E3C,1}, {0x1E3D,0x1E3D,2}, {0x1E3E,0x1E3E,1}, {0x1E3F,0x1E3F,2}, {0x1E40,0x1E40,1},
	{0x1E41,0x1E41,2}, {0x1E42,0x1E42,1}, {0x1E43,0x1E43,2}, {0x1E44,0x1E44,1}, {0x1E45,0x1E45,2},
	{0x1E46,0x1E46,1}, {0x1E47,0x1E47,2}, {0x1E48,0x1E48,1}, {0x1E49,0x1E49,2}, {0x1E4A,0x1E4A,1},
	{0x1E4B,0x1E4B,2}, {0x1E4C,0x1E4C,1}, {0x1E4D,0x1E4D,2}, {0x1E4E,0x1E4E,1}, {0x1E4F,0x1E4F,2},
	{0x1E50,0x1E50,1}, {0x1E51,0x1E51,2}, {0x1E52,0x1E52,1}, {0x1E53,0x1E53,2}, {0x1E54,0x1E54,1},
	{0x1E55,0x1E55,2}, {0x1E56,0x1E56,1}, {0x1E57,0x1E57,2}, {0x1E58,0x1E58,1}, {0x1E59,0x1E59,2},
	{0x1E5A,0x1E5A,1}, {0x1E5B,0x1E5B,2}, {0x1E5C,0x1E5C,1}, {0x1E5D,0x1E5D,2}, {0x1E5E,0x1E5E,1},
	{0x1E5F,0x1E5F,2}, {0x1E60,0x1E60,1}, {0x1E61,0x1E61,2}, {0x1E62,0x1E62,1}, {0x1E63,0x1E63,2},
	{0x1E64,0x1E64,1}, {0x1E65,0x1E65,2}, {0x1E66,0x1E66,1}, {0x1E67,0x1E67,2}, {0x1E68,0x1E68,1},
	{0x1E69,0x1E69,2}, {0x1E6A,0x1E6A,1}, {0x1E6B,0x1E6B,2}, {0x1E6C,0x1E6C,1}, {0x1E6D,0x1E6D,2},
	{0x1E6E,0x1E6E,1}, {0x1E6F,0x1E6F,2}, {0x1E70,0x1E70,1}, {0x1E71,0x1E71,2}, {0x1E72,0x1E72,1},
	{0x1E73,0x1E73,2}, {0x1E74,0x1E74,1}, {0x1E75,0x1E75,2}, {0x1E76,0x1E76,1}, {0x1E77,0x1E77,2},
	{0x1E78,0x1E78,1}, {0x1E79,0x1E79,2}, {0x1E7A,0x1E7A,1}, {0x1E7B,0x1E7B,2}, {0x1E7C,0x1E7C,1},
	{0x1E7D,0x1E7D,2}, {0x1E7E,0x1E7E,1}, {0x1E7F,0x1E7F,2}, {0x1E80,0x1E80,1}, {0x1E81,0x1E81,2},
	{0x1E82,0x1E82,1}, {0x1E83,0x1E83,2}, {0x1E84,0x1E84,1}, {0x1E85,0x1E85,2}, {0x1E86,0x1E86,1},
	{0x1E87,0x1E87,2}, {0x1E88,0x1E88,1}, {0x1E89,0x1E89,2}, {0x1E8A,0x1E8A,1}, {0x1E8B,0x1E8B,2},
	{0x1E8C,0x1E8C,1}, {0x1E8D,0x1E8D,2}, {0x1E8E,0x1E8E,1}, {0x1E8F,0x1E8F,2}, {0x1E90,0x1E90,1},
	{0x1E91,0x1E91,2}, {0x1E92,0x1E92,1}, {0x1E93,0x1E93,2}, {0x1E94,0x1E94,1}, {0x1E95,0x1E9D,2},
	{0x1E9E,0x1E9E,1}, {0x1E9F,0x1E9F,2}, {0x1EA0,0x1EA0,1}, {0x1EA1,0x1EA1,2}, {0x1EA2,0x1EA2,1},
	{0x1EA3,0x1EA3,2}, {0x1EA4,0x1EA4,1}, {0x1EA5,0x1EA5,2}, {0x1EA6,0x1EA6,1}, {0x1EA7,0x1EA7,2},
	{0x1EA8,0x1EA8,1}, {0x1EA9,0x1EA9,2}, {0x1EAA,0x1EAA,1}, {0x1EAB,0x1EAB,2}, {0x1EAC,0x1EAC,1},
	{0x1EAD,0x1EAD,2}, {0x1EAE,0x1EAE,1}, {0x1EAF,0x1EAF,2}, {0x1EB0,0x1EB0,1}, {0x1EB1,0x1EB1,2},
	{0x1EB2,0x1EB2,1}, {0x1EB3,0x1EB3,2}, {0x1EB4,0x1EB4,1}, {0x1EB5,0x1EB5,2}, {0x1EB6,0x1EB6,1},
	{0x1EB7,0x1EB7,2}, {0x1EB8,0x1EB8,1}, {0x1EB9,0x1EB9,2}, {0x1EBA,0x1EBA,1}, {0x1EBB,0x1EBB,2},
	{0x1EBC,0x1EBC,1}, {0x1EBD,0x1EBD,2}, {0x1EBE,0x1EBE,1}, {0x1EBF,0x1EBF,2}, {0x1EC0,0x1EC0,1},
	{0x1EC1,0x1EC1,2}, {0x1EC2,0x1EC2,1}, {0x1EC3,0x1EC3,2}, {0x1EC4,0x1EC4,1}, {0x1EC5,0x1EC5,2},
	{0x1EC6,0x1EC6,1}, {0x1EC7,0x1EC7,2}, {0x1EC8,0x1EC8,1}, {0x1EC9,0x1EC9,2}, {0x1ECA,0x1ECA,1},
	{0x1ECB,0x1ECB,2}, {0x1ECC,0x1ECC,1}, {0x1ECD,0x1ECD,2}, {0x1ECE,0x1ECE,1}, {0x1ECF,0x1ECF,2},
	{0x1ED0,0x1ED0,1}, {0x1ED1,0x1ED1,2}, {0x1ED2,0x1ED2,1}, {0x1ED3,0x1ED3,2}, {0x1ED4,0x1ED4,1},
	{0x1ED5,0x1ED5,2}, {0x1ED6,0x1ED6,1}, {0x1ED7,0x1ED7,2}, {0x1ED8,0x1ED8,1}, {0x1ED9,0x1ED9,2},
	{0x1EDA,0x1EDA,1}, {0x1EDB,0x1EDB,2}, {0x1EDC,0x1EDC,1}, {0x1EDD,0x1EDD,2}, {0x1EDE,0x1EDE,1},
	{0x1EDF,0x1EDF,2}, {0x1EE0,0x1EE0,1}, {0x1EE1,0x1EE1,2}, {0x1EE2,0x1EE2,1}, {0x1EE3,0x1EE3,2},
	{0x1EE4,0x1EE4,1}, {0x1EE5,0x1EE5,2}, {0x1EE6,0x1EE6,1}, {0x1EE7,0x1EE7,2}, {0x1EE8,0x1EE8,1},
	{0x1EE9,0x1EE9,2}, {0x1EEA,0x1EEA,1}, {0x1EEB,0x1EEB,2}, {0x1EEC,0x1EEC,1}, {0x1EED,0x1EED,2},
	{0x1EEE,0x1EEE,1}, {0x1EEF,0x1EEF,2}, {0x1EF0,0x1EF0,1}, {0x1EF1,0x1EF1,2}, {0x1EF2,0x1EF2,1},
	{0x1EF3,0x1EF3,2}, {0x1EF4,0x1EF4,1}, {0x1EF5,0x1EF5,2}, {0x1EF6,0x1EF6,1}, {0x1EF7,0x1EF7,2},
	{0x1EF8,0x1EF8,1}, {0x1EF9,0x1EF9,2}, {0x1EFA,0x1EFA,1}, {0x1EFB,0x1EFB,2}, {0x1EFC,0x1EFC,1},
	{0x1EFD,0x1EFD,2}, {0x1EFE,0x1EFE,1}, {0x1EFF,0x1F07,2}, {0x1F08,0x1F0F,1}, {0x1F10,0x1F15,2},
	{0x1F18,0x1F1D,1}, {0x1F20,0x1F27,2}, {0x1F28,0x1F2F,1}, {0x1F30,0x1F37,2}, {0x1F38,0x1F3F,1},
	{0x1F40,0x1F45,2}, {0x1F48,0x1F4D,1}, {0x1F50,0x1F57,2}, {0x1F59,0x1F59,1}, {0x1F5B,0x1F5B,1},
	{0x1F5D,0x1F5D,1}, {0x1F5F,0x1F5F,1}, {0x1F60,0x1F67,2}, {0x1F68,0x1F6F,1}, {0x1F70,0x1F7D,2},
	{0x1F80,0x1F87,2}, {0x1F88,0x1F8F,1}, {0x1F90,0x1F97,2}, {0x1F98,0x1F9F,1}, {0x1FA0,0x1FA7,2},
	{0x1FA8,0x1FAF,1}, {0x1FB0,0x1FB4,2}, {0x1FB6,0x1FB7,2}, {0x1FB8,0x1FBC,1}, {0x1FBE,0x1FBE,2},
	{0x1FC2,0x1FC4,2}, {0x1FC6,0x1FC7,2}, {0x1FC8,0x1FCC,1}, {0x1FD0,0x1FD3,2}, {0x1FD6,0x1FD7,2},
	{0x1FD8,0x1FDB,1}, {0x1FE0,0x1FE7,2}, {0x1FE8,0x1FEC,1}, {0x1FF2,0x1FF4,2}, {0x1FF6,0x1FF7,2},
	{0x1FF8,0x1FFC,1}, {0x2000,0x200A,5}, {0x2028,0x2029,5}, {0x202F,0x202F,5}, {0x205F,0x205F,5},
	{0x2070,0x2070,4}, {0x2071,0x2071,3}, {0x2074,0x2079,4}, {0x207F,0x207F,3}, {0x2080,0x2089,4},
	{0x2090,0x209C,3}, {0x20D0,0x20F0,6}, {0x2102,0x2102,1}, {0x2107,0x2107,1}, {0x210A,0x210A,2},
	{0x210B,0x210D,1}, {0x210E,0x210F,2}, {0x2110,0x2112,1}, {0x2113,0x2113,2}, {0x2115,0x2115,1},
	{0x2119,0x211D,1}, {0x2124,0x2124,1}, {0x2126,0x2126,1}, {0x2128,0x2128,1}, {0x212A,0x212D,1},
	{0x212F,0x212F,2}, {0x2130,0x2133,1}, {0x2134,0x2134,2}, {0x2135,0x2138,3}, {0x2139,0x2139,2},
	{0x213C,0x213D,2}, {0x213E,0x213F,1}, {0x2145,0x2145,1}, {0x2146,0x2149,2}, {0x214E,0x214E,2},
	{0x2150,0x2182,4}, {0x2183,0x2183,1}, {0x2184,0x2184,2}, {0x2185,0x2189,4}, {0x2460,0x249B,4},
	{0x24EA,0x24FF,4}, {0x2776,0x2793,4}, {0x2C00,0x2C2F,1}, {0x2C30,0x2C5F,2}, {0x2C60,0x2C60,1},
	{0x2C61,0x2C61,2}, {0x2C62,0x2C64,1}, {0x2C65,0x2C66,2}, {0x2C67,0x2C67,1}, {0x2C68,0x2C68,2},
	{0x2C69,0x2C69,1}, {0x2C6A,0x2C6A,2}, {0x2C6B,0x2C6B,1}, {0x2C6C,0x2C6C,2}, {0x2C6D,0x2C70,1},
	{0x2C71,0x2C71,2}, {0x2C72,0x2C72,1}, {0x2C73,0x2C74,2}, {0x2C75,0x2C75,1}, {0x2C76,0x2C7B,2},
	{0x2C7C,0x2C7D,3}, {0x2C7E,0x2C80,1}, {0x2C81,0x2C81,2}, {0x2C82,0x2C82,1}, {0x2C83,0x2C83,2},
	{0x2C84,0x2C84,1}, {0x2C85,0x2C85,2}, {0x2C86,0x2C86,1}, {0x2C87,0x2C87,2}, {0x2C88,0x2C88,1},
	{0x2C89,0x2C89,2}, {0x2C8A,0x2C8A,1}, {0x2C8B,0x2C8B,2}, {0x2C8C,0x2C8C,1}, {0x2C8D,0x2C8D,2},
	{0x2C8E,0x2C8E,1}, {0x2C8F,0x2C8F,2}, {0x2C90,0x2C90,1}, {0x2C91,0x2C91,2}, {0x2C92,0x2C92,1},
	{0x2C93,0x2C93,2}, {0x2C94,0x2C94,1}, {0x2C95,0x2C95,2}, {0x2C96,0x2C96,1}, {0x2C97,0x2C97,2},
	{0x2C98,0x2C98,1}, {0x2C99,0x2C99,2}, {0x2C9A,0x2C9A,1}, {0x2C9B,0x2C9B,2}, {0x2C9C,0x2C9C,1},
	{0x2C9D,0x2C9D,2}, {0x2C9E,0x2C9E,1}, {0x2C9F,0x2C9F,2}, {0x2CA0,0x2CA0,1}, {0x2CA1,0x2CA1,2},
	{0x2CA2,0x2CA2,1}, {0x2CA3,0x2CA3,2}, {0x2CA4,0x2CA4,1}, {0x2CA5,0x2CA5,2}, {0x2CA6,0x2CA6,1},
	{0x2CA7,0x2CA7,2}, {0x2CA8,0x2CA8,1}, {0x2CA9,0x2CA9,2}, {0x2CAA,0x2CAA,1}, {0x2CAB,0x2CAB,2},
	{0x2CAC,0x2CAC,1}, {0x2CAD,0x2CAD,2}, {0x2CAE,0x2CAE,1}, {0x2CAF,0x2CAF,2}, {0x2CB0,0x2CB0,1},
	{0x2CB1,0x2CB1,2}, {0x2CB2,0x2CB2,1}, {0x2CB3,0x2CB3,2}, {0x2CB4,0x2CB4,1}, {0x2CB5,0x2CB5,2},
	{0x2CB6,0x2CB6,1}, {0x2CB7,0x2CB7,2}, {0x2CB8,0x2CB8,1}, {0x2CB9,0x2CB9,2}, {0x2CBA,0x2CBA,1},
	{0x2CBB,0x2CBB,2}, {0x2CBC,0x2CBC,1}, {0x2CBD,0x2CBD,2}, {0x2CBE,0x2CBE,1}, {0x2CBF,0x2CBF,2},
	{0x2CC0,0x2CC0,1}, {0x2CC1,0x2CC1,2}, {0x2CC2,0x2CC2,1}, {0x2CC3,0x2CC3,2}, {0x2CC4,0x2CC4,1},
	{0x2CC5,0x2CC5,2}, {0x2CC6,0x2CC6,1}, {0x2CC7,0x2CC7,2}, {0x2CC8,0x2CC8,1}, {0x2CC9,0x2CC9,2},
	{0x2CCA,0x2CCA,1}, {0x2CCB,0x2CCB,2}, {0x2CCC,0x2CCC,1}, {0x2CCD,0x2CCD,2}, {0x2CCE,0x2CCE,1},
	{0x2CCF,0x2CCF,2}, {0x2CD0,0x2CD0,1}, {0x2CD1,0x2CD1,2}, {0x2CD2,0x2CD2,1}, {0x2CD3,0x2CD3,2},
	{0x2CD4,0x2CD4,1}, {0x2CD5,0x2CD5,2}, {0x2CD6,0x2CD6,1}, {0x2CD7,0x2CD7,2}, {0x2CD8,0x2CD8,1},
	{0x2CD9,0x2CD9,2}, {0x2CDA,0x2CDA,1}, {0x2CDB,0x2CDB,2}, {0x2CDC,0x2CDC,1}, {0x2CDD,0x2CDD,2},
	{0x2CDE,0x2CDE,1}, {0x2CDF,0x2CDF,2}, {0x2CE0,0x2CE0,1}, {0x2CE1,0x2CE1,2}, {0x2CE2,0x2CE2,1},
	{0x2CE3,0x2CE4,2}, {0x2CEB,0x2CEB,1}, {0x2CEC,0x2CEC,2}, {0x2CED,0x2CED,1}, {0x2CEE,0x2CEE,2},
	{0x2CEF,0x2CF1,6}, {0x2CF2,0x2CF2,1}, {0x2CF3,0x2CF3,2}, {0x2CFD,0x2CFD,4}, {0x2D00,0x2D25,2},
	{0x2D27,0x2D27,2}, {0x2D2D,0x2D2D,2}, {0x2D30,0x2D67,3}, {0x2D6F,0x2D6F,3}, {0x2D7F,0x2D7F,6},
	{0x2D80,0x2D96,3}, {0x2DA0,0x2DA6,3}, {0x2DA8,0x2DAE,3}, {0x2DB0,0x2DB6,3}, {0x2DB8,0x2DBE,3},
	{0x2DC0,0x2DC6,3}, {0x2DC8,0x2DCE,3}, {0x2DD0,0x2DD6,3}, {0x2DD8,0x2DDE,3}, {0x2DE0,0x2DFF,6},
	{0x2E2F,0x2E2F,3}, {0x3000,0x3000,5}, {0x3005,0x3006,3}, {0x3007,0x3007,4}, {0x3021,0x3029,4},
	{0x302A,0x302F,6}, {0x3031,0x3035,3}, {0x3038,0x303A,4}, {0x303B,0x303C,3}, {0x3041,0x3096,3},
	{0x3099,0x309A,6}, {0x309D,0x309F,3}, {0x30A1,0x30FA,3}, {0x30FC,0x30FF,3}, {0x3105,0x312F,3},
	{0x3131,0x318E,3}, {0x3192,0x3195,4}, {0x31A0,0x31BF,3}, {0x31F0,0x31FF,3}, {0x3220,0x3229,4},
	{0x3248,0x324F,4}, {0x3251,0x325F,4}, {0x3280,0x3289,4}, {0x32B1,0x32BF,4}, {0x3400,0x4DBF,3},
	{0x4E00,0xA48C,3}, {0xA4D0,0xA4FD,3}, {0xA500,0xA60C,3}, {0xA610,0xA61F,3}, {0xA620,0xA629,4},
	{0xA62A,0xA62B,3}, {0xA640,0xA640,1}, {0xA641,0xA641,2}, {0xA642,0xA642,1}, {0xA643,0xA643,2},
	{0xA644,0xA644,1}, {0xA645,0xA645,2}, {0xA646,0xA646,1}, {0xA647,0xA647,2}, {0xA648,0xA648,1},
	{0xA649,0xA649,2}, {0xA64A,0xA64A,1}, {0xA64B,0xA64B,2}, {0xA64C,0xA64C,1}, {0xA64D,0xA64D,2},
	{0xA64E,0xA64E,1}, {0xA64F,0xA64F,2}, {0xA650,0xA650,1}, {0xA651,0xA651,2}, {0xA652,0xA652,1},
	{0xA653,0xA653,2}, {0xA654,0xA654,1}, {0xA655,0xA655,2}, {0xA656,0xA656,1}, {0xA657,0xA657,2},
	{0xA658,0xA658,1}, {0xA659,0xA659,2}, {0xA65A,0xA65A,1}, {0xA65B,0xA65B,2}, {0xA65C,0xA65C,1},
	{0xA65D,0xA65D,2}, {0xA65E,0xA65E,1}, {0xA65F,0xA65F,2}, {0xA660,0xA660,1}, {0xA661,0xA661,2},
	{0xA662,0xA662,1}, {0xA663,0xA663,2}, {0xA664,0xA664,1}, {0xA665,0xA665,2}, {0xA666,0xA666,1},
	{0xA667,0xA667,2}, {0xA668,0xA668,1}, {0xA669,0xA669,2}, {0xA66A,0xA66A,1}, {0xA66B,0xA66B,2},
	{0xA66C,0xA66C,1}, {0xA66D,0xA66D,2}, {0xA66E,0xA66E,3}, {0xA66F,0xA672,6}, {0xA674,0xA67D,6},
	{0xA67F,0xA67F,3}, {0xA680,0xA680,1}, {0xA681,0xA681,2}, {0xA682,0xA682,1}, {0xA683,0xA683,2},
	{0xA684,0xA684,1}, {0xA685,0xA685,2}, {0xA686,0xA686,1}, {0xA687,0xA687,2}, {0xA688,0xA688,1},
	{0xA689,0xA689,2}, {0xA68A,0xA68A,1}, {0xA68B,0xA68B,2}, {0xA68C,0xA68C,1}, {0xA68D,0xA68D,2},
	{0xA68E,0xA68E,1}, {0xA68F,0xA68F,2}, {0xA690,0xA690,1}, {0xA691,0xA691,2}, {0xA692,0xA692,1},
	{0xA693,0xA693,2}, {0xA694,0xA694,1}, {0xA695,0xA695,2}, {0xA696,0xA696,1}, {0xA697,0xA697,2},
	{0xA698,0xA698,1}, {0xA699,0xA699,2}, {0xA69A,0xA69A,1}, {0xA69B,0xA69B,2}, {0xA69C,0xA69D,3},
	{0xA69E,0xA69F,6}, {0xA6A0,0xA6E5,3}, {0xA6E6,0xA6EF,4}, {0xA6F0,0xA6F1,6}, {0xA717,0xA71F,3},
	{0xA722,0xA722,1}, {0xA723,0xA723,2}, {0xA724,0xA724,1}, {0xA725,0xA725,2}, {0xA726,0xA726,1},
	{0xA727,0xA727,2}, {0xA728,0xA728,1}, {0xA729,0xA729,2}, {0xA72A,0xA72A,1}, {0xA72B,0xA72B,2},
	{0xA72C,0xA72C,1}, {0xA72D,0xA72D,2}, {0xA72E,0xA72E,1}, {0xA72F,0xA731,2}, {0xA732,0xA732,1},
	{0xA733,0xA733,2}, {0xA734,0xA734,1}, {0xA735,0xA735,2}, {0xA736,0xA736,1}, {0xA737,0xA737,2},
	{0xA738,0xA738,1}, {0xA739,0xA739,2}, {0xA73A,0xA73A,1}, {0xA73B,0xA73B,2}, {0xA73C,0xA73C,1},
	{0xA73D,0xA73D,2}, {0xA73E,0xA73E,1}, {0xA73F,0xA73F,2}, {0xA740,0xA740,1}, {0xA741,0xA741,2},
	{0xA742,0xA742,1}, {0xA743,0xA743,2}, {0xA744,0xA744,1}, {0xA745,0xA745,2}, {0xA746,0xA746,1},
	{0xA747,0xA747,2}, {0xA748,0xA748,1}, {0xA749,0xA749,2}, {0xA74A,0xA74A,1}, {0xA74B,0xA74B,2},
	{0xA74C,0xA74C,1}, {0xA74D,0xA74D,2}, {0xA74E,0xA74E,1}, {0xA74F,0xA74F,2}, {0xA750,0xA750,1},
	{0xA751,0xA751,2}, {0xA752,0xA752,1}, {0xA753,0xA753,2}, {0xA754,0xA754,1}, {0xA755,0xA755,2},
	{0xA756,0xA756,1}, {0xA757,0xA757,2}, {0xA758,0xA758,1}, {0xA759,0xA759,2}, {0xA75A,0xA75A,1},
	{0xA75B,0xA75B,2}, {0xA75C,0xA75C,1}, {0xA75D,0xA75D,2}, {0xA75E,0xA75E,1}, {0xA75F,0xA75F,2},
	{0xA760,0xA760,1}, {0xA761,0xA761,2}, {0xA762,0xA762,1}, {0xA763,0xA763,2}, {0xA764,0xA764,1},
	{0xA765,0xA765,2}, {0xA766,0xA766,1}, {0xA767,0xA767,2}, {0xA768,0xA768,1}, {0xA769,0xA769,2},
	{0xA76A,0xA76A,1}, {0xA76B,0xA76B,2}, {0xA76C,0xA76C,1}, {0xA76D,0xA76D,2}, {0xA76E,0xA76E,1},
	{0xA76F,0xA76F,2}, {0xA770,0xA770,3}, {0xA771,0xA778,2}, {0xA779,0xA779,1}, {0xA77A,0xA77A,2},
	{0xA77B,0xA77B,1}, {0xA77C,0xA77C,2}, {0xA77D,0xA77E,1}, {0xA77F,0xA77F,2}, {0xA780,0xA780,1},
	{0xA781,0xA781,2}, {0xA782,0xA782,1}, {0xA783,0xA783,2}, {0xA784,0xA784,1}, {0xA785,0xA785,2},
	{0xA786,0xA786,1}, {0xA787,0xA787,2}, {0xA788,0xA788,3}, {0xA78B,0xA78B,1}, {0xA78C,0xA78C,2},
	{0xA78D,0xA78D,1}, {0xA78E,0xA78E,2}, {0xA78F,0xA78F,3}, {0xA790,0xA790,1}, {0xA791,0xA791,2},
	{0xA792,0xA792,1}, {0xA793,0xA795,2}, {0xA796,0xA796,1}, {0xA797,0xA797,2}, {0xA798,0xA798,1},
	{0xA799,0xA799,2}, {0xA79A,0xA79A,1}, {0xA79B,0xA79B,2}, {0xA79C,0xA79C,1}, {0xA79D,0xA79D,2},
	{0xA79E,0xA79E,1}, {0xA79F,0xA79F,2}, {0xA7A0,0xA7A0,1}, {0xA7A1,0xA7A1,2}, {0xA7A2,0xA7A2,1},
	{0xA7A3,0xA7A3,2}, {0xA7A4,0xA7A4,1}, {0xA7A5,0xA7A5,2}, {0xA7A6,0xA7A6,1}, {0xA7A7,0xA7A7,2},
	{0xA7A8,0xA7A8,1}, {0xA7A9,0xA7A9,2}, {0xA7AA,0xA7AE,1}, {0xA7AF,0xA7AF,2}, {0xA7B0,0xA7B4,1},
	{0xA7B5,0xA7B5,2}, {0xA7B6,0xA7B6,1}, {0xA7B7,0xA7B7,2}, {0xA7B8,0xA7B8,1}, {0xA7B9,0xA7B9,2},
	{0xA7BA,0xA7BA,1}, {0xA7BB,0xA7BB,2}, {0xA7BC,0xA7BC,1}, {0xA7BD,0xA7BD,2}, {0xA7BE,0xA7BE,1},
	{0xA7BF,0xA7BF,2}, {0xA7C0,0xA7C0,1}, {0xA7C1,0xA7C1,2}, {0xA7C2,0xA7C2,1}, {0xA7C3,0xA7C3,2},
	{0xA7C4,0xA7C7,1}, {0xA7C8,0xA7C8,2}, {0xA7C9,0xA7C9,1}, {0xA7CA,0xA7CA,2}, {0xA7CB,0xA7CC,1},
	{0xA7CD,0xA7CD,2}, {0xA7D0,0xA7D0,1}, {0xA7D1,0xA7D1,2}, {0xA7D3,0xA7D3,2}, {0xA7D5,0xA7D5,2},
	{0xA7D6,0xA7D6,1}, {0xA7D7,0xA7D7,2}, {0xA7D8,0xA7D8,1}, {0xA7D9,0xA7D9,2}, {0xA7DA,0xA7DA,1},
	{0xA7DB,0xA7DB,2}, {0xA7DC,0xA7DC,1}, {0xA7F2,0xA7F4,3}, {0xA7F5,0xA7F5,1}, {0xA7F6,0xA7F6,2},
	{0xA7F7,0xA7F9,3}, {0xA7FA,0xA7FA,2}, {0xA7FB,0xA801,3}, {0xA802,0xA802,6}, {0xA803,0xA805,3},
	{0xA806,0xA806,6}, {0xA807,0xA80A,3}, {0xA80B,0xA80B,6}, {0xA80C,0xA822,3}, {0xA823,0xA827,6},
	{0xA82C,0xA82C,6}, {0xA830,0xA835,4}, {0xA840,0xA873,3}, {0xA880,0xA881,6}, {0xA882,0xA8B3,3},
	{0xA8B4,0xA8C5,6}, {0xA8D0,0xA8D9,4}, {0xA8E0,0xA8F1,6}, {0xA8F2,0xA8F7,3}, {0xA8FB,0xA8FB,3},
	{0xA8FD,0xA8FE,3}, {0xA8FF,0xA8FF,6}, {0xA900,0xA909,4}, {0xA90A,0xA925,3}, {0xA926,0xA92D,6},
	{0xA930,0xA946,3}, {0xA947,0xA953,6}, {0xA960,0xA97C,3}, {0xA980,0xA983,6}, {0xA984,0xA9B2,3},
	{0xA9B3,0xA9C0,6}, {0xA9CF,0xA9CF,3}, {0xA9D0,0xA9D9,4}, {0xA9E0,0xA9E4,3}, {0xA9E5,0xA9E5,6},
	{0xA9E6,0xA9EF,3}, {0xA9F0,0xA9F9,4}, {0xA9FA,0xA9FE,3}, {0xAA00,0xAA28,3}, {0xAA29,0xAA36,6},
	{0xAA40,0xAA42,3}, {0xAA43,0xAA43,6}, {0xAA44,0xAA4B,3}, {0xAA4C,0xAA4D,6}, {0xAA50,0xAA59,4},
	{0xAA60,0xAA76,3}, {0xAA7A,0xAA7A,3}, {0xAA7B,0xAA7D,6}, {0xAA7E,0xAAAF,3}, {0xAAB0,0xAAB0,6},
	{0xAAB1,0xAAB1,3}, {0xAAB2,0xAAB4,6}, {0xAAB5,0xAAB6,3}, {0xAAB7,0xAAB8,6}, {0xAAB9,0xAABD,3},
	{0xAABE,0xAABF,6}, {0xAAC0,0xAAC0,3}, {0xAAC1,0xAAC1,6}, {0xAAC2,0xAAC2,3}, {0xAADB,0xAADD,3},
	{0xAAE0,0xAAEA,3}, {0xAAEB,0xAAEF,6}, {0xAAF2,0xAAF4,3}, {0xAAF5,0xAAF6,6}, {0xAB01,0xAB06,3},
	{0xAB09,0xAB0E,3}, {0xAB11,0xAB16,3}, {0xAB20,0xAB26,3}, {0xAB28,0xAB2E,3}, {0xAB30,0xAB5A,2},
	{0xAB5C,0xAB5F,3}, {0xAB60,0xAB68,2}, {0xAB69,0xAB69,3}, {0xAB70,0xABBF,2}, {0xABC0,0xABE2,3},
	{0xABE3,0xABEA,6}, {0xABEC,0xABED,6}, {0xABF0,0xABF9,4}, {0xAC00,0xD7A3,3}, {0xD7B0,0xD7C6,3},
	{0xD7CB,0xD7FB,3}, {0xF900,0xFA6D,3}, {0xFA70,0xFAD9,3}, {0xFB00,0xFB06,2}, {0xFB13,0xFB17,2},
	{0xFB1D,0xFB1D,3}, {0xFB1E,0xFB1E,6}, {0xFB1F,0xFB28,3}, {0xFB2A,0xFB36,3}, {0xFB38,0xFB3C,3},
	{0xFB3E,0xFB3E,3}, {0xFB40,0xFB41,3}, {0xFB43,0xFB44,3}, {0xFB46,0xFBB1,3}, {0xFBD3,0xFD3D,3},
	{0xFD50,0xFD8F,3}, {0xFD92,0xFDC7,3}, {0xFDF0,0xFDFB,3}, {0xFE00,0xFE0F,6}, {0xFE20,0xFE2F,6},
	{0xFE70,0xFE74,3}, {0xFE76,0xFEFC,3}, {0xFF10,0xFF19,4}, {0xFF21,0xFF3A,1}, {0xFF41,0xFF5A,2},
	{0xFF66,0xFFBE,3}, {0xFFC2,0xFFC7,3}, {0xFFCA,0xFFCF,3}, {0xFFD2,0xFFD7,3}, {0xFFDA,0xFFDC,3},
	{0x10000,0x1000B,3}, {0x1000D,0x10026,3}, {0x10028,0x1003A,3}, {0x1003C,0x1003D,3}, {0x1003F,0x1004D,3},
	{0x10050,0x1005D,3}, {0x10080,0x100FA,3}, {0x10107,0x10133,4}, {0x10140,0x10178,4}, {0x1018A,0x1018B,4},
	{0x101FD,0x101FD,6}, {0x10280,0x1029C,3}, {0x102A0,0x102D0,3}, {0x102E0,0x102E0,6}, {0x102E1,0x102FB,4},
	{0x10300,0x1031F,3}, {0x10320,0x10323,4}, {0x1032D,0x10340,3}, {0x10341,0x10341,4}, {0x10342,0x10349,3},
	{0x1034A,0x1034A,4}, {0x10350,0x10375,3}, {0x10376,0x1037A,6}, {0x10380,0x1039D,3}, {0x103A0,0x103C3,3},
	{0x103C8,0x103CF,3}, {0x103D1,0x103D5,4}, {0x10400,0x10427,1}, {0x10428,0x1044F,2}, {0x10450,0x1049D,3},
	{0x104A0,0x104A9,4}, {0x104B0,0x104D3,1}, {0x104D8,0x104FB,2}, {0x10500,0x10527,3}, {0x10530,0x10563,3},
	{0x10570,0x1057A,1}, {0x1057C,0x1058A,1}, {0x1058C,0x10592,1}, {0x10594,0x10595,1}, {0x10597,0x105A1,2},
	{0x105A3,0x105B1,2}, {0x105B3,0x105B9,2}, {0x105BB,0x105BC,2}, {0x105C0,0x105F3,3}, {0x10600,0x10736,3},
	{0x10740,0x10755,3}, {0x10760,0x10767,3}, {0x10780,0x10785,3}, {0x10787,0x107B0,3}, {0x107B2,0x107BA,3},
	{0x10800,0x10805,3}, {0x10808,0x10808,3}, {0x1080A,0x10835,3}, {0x10837,0x10838,3}, {0x1083C,0x1083C,3},
	{0x1083F,0x10855,3}, {0x10858,0x1085F,4}, {0x10860,0x10876,3}, {0x10879,0x1087F,4}, {0x10880,0x1089E,3},
	{0x108A7,0x108AF,4}, {0x108E0,0x108F2,3}, {0x108F4,0x108F5,3}, {0x108FB,0x108FF,4}, {0x10900,0x10915,3},
	{0x10916,0x1091B,4}, {0x10920,0x10939,3}, {0x10980,0x109B7,3}, {0x109BC,0x109BD,4}, {0x109BE,0x109BF,3},
	{0x109C0,0x109CF,4}, {0x109D2,0x109FF,4}, {0x10A00,0x10A00,3}, {0x10A01,0x10A03,6}, {0x10A05,0x10A06,6},
	{0x10A0C,0x10A0F,6}, {0x10A10,0x10A13,3}, {0x10A15,0x10A17,3}, {0x10A19,0x10A35,3}, {0x10A38,0x10A3A,6},
	{0x10A3F,0x10A3F,6}, {0x10A40,0x10A48,4}, {0x10A60,0x10A7C,3}, {0x10A7D,0x10A7E,4}, {0x10A80,0x10A9C,3},
	{0x10A9D,0x10A9F,4}, {0x10AC0,0x10AC7,3}, {0x10AC9,0x10AE4,3}, {0x10AE5,0x10AE6,6}, {0x10AEB,0x10AEF,4},
	{0x10B00,0x10B35,3}, {0x10B40,0x10B55,3}, {0x10B58,0x10B5F,4}, {0x10B60,0x10B72,3}, {0x10B78,0x10B7F,4},
	{0x10B80,0x10B91,3}, {0x10BA9,0x10BAF,4}, {0x10C00,0x10C48,3}, {0x10C80,0x10CB2,1}, {0x10CC0,0x10CF2,2},
	{0x10CFA,0x10CFF,4}, {0x10D00,0x10D23,3}, {0x10D24,0x10D27,6}, {0x10D30,0x10D39,4}, {0x10D40,0x10D49,4},
	{0x10D4A,0x10D4F,3}, {0x10D50,0x10D65,1}, {0x10D69,0x10D6D,6}, {0x10D6F,0x10D6F,3}, {0x10D70,0x10D85,2},
	{0x10E60,0x10E7E,4}, {0x10E80,0x10EA9,3}, {0x10EAB,0x10EAC,6}, {0x10EB0,0x10EB1,3}, {0x10EC2,0x10EC4,3},
	{0x10EFC,0x10EFF,6}, {0x10F00,0x10F1C,3}, {0x10F1D,0x10F26,4}, {0x10F27,0x10F27,3}, {0x10F30,0x10F45,3},
	{0x10F46,0x10F50,6}, {0x10F51,0x10F54,4}, {0x10F70,0x10F81,3}, {0x10F82,0x10F85,6}, {0x10FB0,0x10FC4,3},
	{0x10FC5,0x10FCB,4}, {0x10FE0,0x10FF6,3}, {0x11000,0x11002,6}, {0x11003,0x11037,3}, {0x11038,0x11046,6},
	{0x11052,0x1106F,4}, {0x11070,0x11070,6}, {0x11071,0x11072,3}, {0x11073,0x11074,6}, {0x11075,0x11075,3},
	{0x1107F,0x11082,6}, {0x11083,0x110AF,3}, {0x110B0,0x110BA,6}, {0x110C2,0x110C2,6}, {0x110D0,0x110E8,3},
	{0x110F0,0x110F9,4}, {0x11100,0x11102,6}, {0x11103,0x11126,3}, {0x11127,0x11134,6}, {0x11136,0x1113F,4},
	{0x11144,0x11144,3}, {0x11145,0x11146,6}, {0x11147,0x11147,3}, {0x11150,0x11172,3}, {0x11173,0x11173,6},
	{0x11176,0x11176,3}, {0x11180,0x11182,6}, {0x11183,0x111B2,3}, {0x111B3,0x111C0,6}, {0x111C1,0x111C4,3},
	{0x111C9,0x111CC,6}, {0x111CE,0x111CF,6}, {0x111D0,0x111D9,4}, {0x111DA,0x111DA,3}, {0x111DC,0x111DC,3},
	{0x111E1,0x111F4,4}, {0x11200,0x11211,3}, {0x11213,0x1122B,3}, {0x1122C,0x11237,6}, {0x1123E,0x1123E,6},
	{0x1123F,0x11240,3}, {0x11241,0x11241,6}, {0x11280,0x11286,3}, {0x11288,0x11288,3}, {0x1128A,0x1128D,3},
	{0x1128F,0x1129D,3}, {0x1129F,0x112A8,3}, {0x112B0,0x112DE,3}, {0x112DF,0x112EA,6}, {0x112F0,0x112F9,4},
	{0x11300,0x11303,6}, {0x11305,0x1130C,3}, {0x1130F,0x11310,3}, {0x11313,0x11328,3}, {0x1132A,0x11330,3},
	{0x11332,0x11333,3}, {0x11335,0x11339,3}, {0x1133B,0x1133C,6}, {0x1133D,0x1133D,3}, {0x1133E,0x11344,6},
	{0x11347,0x11348,6}, {0x1134B,0x1134D,6}, {0x11350,0x11350,3}, {0x11357,0x11357,6}, {0x1135D,0x11361,3},
	{0x11362,0x11363,6}, {0x11366,0x1136C,6}, {0x11370,0x11374,6}, {0x11380,0x11389,3}, {0x1138B,0x1138B,3},
	{0x1138E,0x1138E,3}, {0x11390,0x113B5,3}, {0x113B7,0x113B7,3}, {0x113B8,0x113C0,6}, {0x113C2,0x113C2,6},
	{0x113C5,0x113C5,6}, {0x113C7,0x113CA,6}, {0x113CC,0x113D0,6}, {0x113D1,0x113D1,3}, {0x113D2,0x113D2,6},
	{0x113D3,0x113D3,3}, {0x113E1,0x113E2,6}, {0x11400,0x11434,3}, {0x11435,0x11446,6}, {0x11447,0x1144A,3},
	{0x11450,0x11459,4}, {0x1145E,0x1145E,6}, {0x1145F,0x11461,3}, {0x11480,0x114AF,3}, {0x114B0,0x114C3,6},
	{0x114C4,0x114C5,3}, {0x114C7,0x114C7,3}, {0x114D0,0x114D9,4}, {0x11580,0x115AE,3}, {0x115AF,0x115B5,6},
	{0x115B8,0x115C0,6}, {0x115D8,0x115DB,3}, {0x115DC,0x115DD,6}, {0x11600,0x1162F,3}, {0x11630,0x11640,6},
	{0x11644,0x11644,3}, {0x11650,0x11659,4}, {0x11680,0x116AA,3}, {0x116AB,0x116B7,6}, {0x116B8,0x116B8,3},
	{0x116C0,0x116C9,4}, {0x116D0,0x116E3,4}, {0x11700,0x1171A,3}, {0x1171D,0x1172B,6}, {0x11730,0x1173B,4},
	{0x11740,0x11746,3}, {0x11800,0x1182B,3}, {0x1182C,0x1183A,6}, {0x118A0,0x118BF,1}, {0x118C0,0x118DF,2},
	{0x118E0,0x118F2,4}, {0x118FF,0x11906,3}, {0x11909,0x11909,3}, {0x1190C,0x11913,3}, {0x11915,0x11916,3},
	{0x11918,0x1192F,3}, {0x11930,0x11935,6}, {0x11937,0x11938,6}, {0x1193B,0x1193E,6}, {0x1193F,0x1193F,3},
	{0x11940,0x11940,6}, {0x11941,0x11941,3}, {0x11942,0x11943,6}, {0x11950,0x11959,4}, {0x119A0,0x119A7,3},
	{0x119AA,0x119D0,3}, {0x119D1,0x119D7,6}, {0x119DA,0x119E0,6}, {0x119E1,0x119E1,3}, {0x119E3,0x119E3,3},
	{0x119E4,0x119E4,6}, {0x11A00,0x11A00,3}, {0x11A01,0x11A0A,6}, {0x11A0B,0x11A32,3}, {0x11A33,0x11A39,6},
	{0x11A3A,0x11A3A,3}, {0x11A3B,0x11A3E,6}, {0x11A47,0x11A47,6}, {0x11A50,0x11A50,3}, {0x11A51,0x11A5B,6},
	{0x11A5C,0x11A89,3}, {0x11A8A,0x11A99,6}, {0x11A9D,0x11A9D,3}, {0x11AB0,0x11AF8,3}, {0x11BC0,0x11BE0,3},
	{0x11BF0,0x11BF9,4}, {0x11C00,0x11C08,3}, {0x11C0A,0x11C2E,3}, {0x11C2F,0x11C36,6}, {0x11C38,0x11C3F,6},
	{0x11C40,0x11C40,3}, {0x11C50,0x11C6C,4}, {0x11C72,0x11C8F,3}, {0x11C92,0x11CA7,6}, {0x11CA9,0x11CB6,6},
	{0x11D00,0x11D06,3}, {0x11D08,0x11D09,3}, {0x11D0B,0x11D30,3}, {0x11D31,0x11D36,6}, {0x11D3A,0x11D3A,6},
	{0x11D3C,0x11D3D,6}, {0x11D3F,0x11D45,6}, {0x11D46,0x11D46,3}, {0x11D47,0x11D47,6}, {0x11D50,0x11D59,4},
	{0x11D60,0x11D65,3}, {0x11D67,0x11D68,3}, {0x11D6A,0x11D89,3}, {0x11D8A,0x11D8E,6}, {0x11D90,0x11D91,6},
	{0x11D93,0x11D97,6}, {0x11D98,0x11D98,3}, {0x11DA0,0x11DA9,4}, {0x11EE0,0x11EF2,3}, {0x11EF3,0x11EF6,6},
	{0x11F00,0x11F01,6}, {0x11F02,0x11F02,3}, {0x11F03,0x11F03,6}, {0x11F04,0x11F10,3}, {0x11F12,0x11F33,3},
	{0x11F34,0x11F3A,6}, {0x11F3E,0x11F42,6}, {0x11F50,0x11F59,4}, {0x11F5A,0x11F5A,6}, {0x11FB0,0x11FB0,3},
	{0x11FC0,0x11FD4,4}, {0x12000,0x12399,3}, {0x12400,0x1246E,4}, {0x12480,0x12543,3}, {0x12F90,0x12FF0,3},
	{0x13000,0x1342F,3}, {0x13440,0x13440,6}, {0x13441,0x13446,3}, {0x13447,0x13455,6}, {0x13460,0x143FA,3},
	{0x14400,0x14646,3}, {0x16100,0x1611D,3}, {0x1611E,0x1612F,6}, {0x16130,0x16139,4}, {0x16800,0x16A38,3},
	{0x16A40,0x16A5E,3}, {0x16A60,0x16A69,4}, {0x16A70,0x16ABE,3}, {0x16AC0,0x16AC9,4}, {0x16AD0,0x16AED,3},
	{0x16AF0,0x16AF4,6}, {0x16B00,0x16B2F,3}, {0x16B30,0x16B36,6}, {0x16B40,0x16B43,3}, {0x16B50,0x16B59,4},
	{0x16B5B,0x16B61,4}, {0x16B63,0x16B77,3}, {0x16B7D,0x16B8F,3}, {0x16D40,0x16D6C,3}, {0x16D70,0x16D79,4},
	{0x16E40,0x16E5F,1}, {0x16E60,0x16E7F,2}, {0x16E80,0x16E96,4}, {0x16F00,0x16F4A,3}, {0x16F4F,0x16F4F,6},
	{0x16F50,0x16F50,3}, {0x16F51,0x16F87,6}, {0x16F8F,0x16F92,6}, {0x16F93,0x16F9F,3}, {0x16FE0,0x16FE1,3},
	{0x16FE3,0x16FE3,3}, {0x16FE4,0x16FE4,6}, {0x16FF0,0x16FF1,6}, {0x17000,0x187F7,3}, {0x18800,0x18CD5,3},
	{0x18CFF,0x18D08,3}, {0x1AFF0,0x1AFF3,3}, {0x1AFF5,0x1AFFB,3}, {0x1AFFD,0x1AFFE,3}, {0x1B000,0x1B122,3},
	{0x1B132,0x1B132,3}, {0x1B150,0x1B152,3}, {0x1B155,0x1B155,3}, {0x1B164,0x1B167,3}, {0x1B170,0x1B2FB,3},
	{0x1BC00,0x1BC6A,3}, {0x1BC70,0x1BC7C,3}, {0x1BC80,0x1BC88,3}, {0x1BC90,0x1BC99,3}, {0x1BC9D,0x1BC9E,6},
	{0x1CCF0,0x1CCF9,4}, {0x1CF00,0x1CF2D,6}, {0x1CF30,0x1CF46,6}, {0x1D165,0x1D169,6}, {0x1D16D,0x1D172,6},
	{0x1D17B,0x1D182,6}, {0x1D185,0x1D18B,6}, {0x1D1AA,0x1D1AD,6}, {0x1D242,0x1D244,6}, {0x1D2C0,0x1D2D3,4},
	{0x1D2E0,0x1D2F3,4}, {0x1D360,0x1D378,4}, {0x1D400,0x1D419,1}, {0x1D41A,0x1D433,2}, {0x1D434,0x1D44D,1},
	{0x1D44E,0x1D454,2}, {0x1D456,0x1D467,2}, {0x1D468,0x1D481,1}, {0x1D482,0x1D49B,2}, {0x1D49C,0x1D49C,1},
	{0x1D49E,0x1D49F,1}, {0x1D4A2,0x1D4A2,1}, {0x1D4A5,0x1D4A6,1}, {0x1D4A9,0x1D4AC,1}, {0x1D4AE,0x1D4B5,1},
	{0x1D4B6,0x1D4B9,2}, {0x1D4BB,0x1D4BB,2}, {0x1D4BD,0x1D4C3,2}, {0x1D4C5,0x1D4CF,2}, {0x1D4D0,0x1D4E9,1},
	{0x1D4EA,0x1D503,2}, {0x1D504,0x1D505,1}, {0x1D507,0x1D50A,1}, {0x1D50D,0x1D514,1}, {0x1D516,0x1D51C,1},
	{0x1D51E,0x1D537,2}, {0x1D538,0x1D539,1}, {0x1D53B,0x1D53E,1}, {0x1D540,0x1D544,1}, {0x1D546,0x1D546,1},
	{0x1D54A,0x1D550,1}, {0x1D552,0x1D56B,2}, {0x1D56C,0x1D585,1}, {0x1D586,0x1D59F,2}, {0x1D5A0,0x1D5B9,1},
	{0x1D5BA,0x1D5D3,2}, {0x1D5D4,0x1D5ED,1}, {0x1D5EE,0x1D607,2}, {0x1D608,0x1D621,1}, {0x1D622,0x1D63B,2},
	{0x1D63C,0x1D655,1}, {0x1D656,0x1D66F,2}, {0x1D670,0x1D689,1}, {0x1D68A,0x1D6A5,2}, {0x1D6A8,0x1D6C0,1},
	{0x1D6C2,0x1D6DA,2}, {0x1D6DC,0x1D6E1,2}, {0x1D6E2,0x1D6FA,1}, {0x1D6FC,0x1D714,2}, {0x1D716,0x1D71B,2},
	{0x1D71C,0x1D734,1}, {0x1D736,0x1D74E,2}, {0x1D750,0x1D755,2}, {0x1D756,0x1D76E,1}, {0x1D770,0x1D788,2},
	{0x1D78A,0x1D78F,2}, {0x1D790,0x1D7A8,1}, {0x1D7AA,0x1D7C2,2}, {0x1D7C4,0x1D7C9,2}, {0x1D7CA,0x1D7CA,1},
	{0x1D7CB,0x1D7CB,2}, {0x1D7CE,0x1D7FF,4}, {0x1DA00,0x1DA36,6}, {0x1DA3B,0x1DA6C,6}, {0x1DA75,0x1DA75,6},
	{0x1DA84,0x1DA84,6}, {0x1DA9B,0x1DA9F,6}, {0x1DAA1,0x1DAAF,6}, {0x1DF00,0x1DF09,2}, {0x1DF0A,0x1DF0A,3},
	{0x1DF0B,0x1DF1E,2}, {0x1DF25,0x1DF2A,2}, {0x1E000,0x1E006,6}, {0x1E008,0x1E018,6}, {0x1E01B,0x1E021,6},
	{0x1E023,0x1E024,6}, {0x1E026,0x1E02A,6}, {0x1E030,0x1E06D,3}, {0x1E08F,0x1E08F,6}, {0x1E100,0x1E12C,3},
	{0x1E130,0x1E136,6}, {0x1E137,0x1E13D,3}, {0x1E140,0x1E149,4}, {0x1E14E,0x1E14E,3}, {0x1E290,0x1E2AD,3},
	{0x1E2AE,0x1E2AE,6}, {0x1E2C0,0x1E2EB,3}, {0x1E2EC,0x1E2EF,6}, {0x1E2F0,0x1E2F9,4}, {0x1E4D0,0x1E4EB,3},
	{0x1E4EC,0x1E4EF,6}, {0x1E4F0,0x1E4F9,4}, {0x1E5D0,0x1E5ED,3}, {0x1E5EE,0x1E5EF,6}, {0x1E5F0,0x1E5F0,3},
	{0x1E5F1,0x1E5FA,4}, {0x1E7E0,0x1E7E6,3}, {0x1E7E8,0x1E7EB,3}, {0x1E7ED,0x1E7EE,3}, {0x1E7F0,0x1E7FE,3},
	{0x1E800,0x1E8C4,3}, {0x1E8C7,0x1E8CF,4}, {0x1E8D0,0x1E8D6,6}, {0x1E900,0x1E921,1}, {0x1E922,0x1E943,2},
	{0x1E944,0x1E94A,6}, {0x1E94B,0x1E94B,3}, {0x1E950,0x1E959,4}, {0x1EC71,0x1ECAB,4}, {0x1ECAD,0x1ECAF,4},
	{0x1ECB1,0x1ECB4,4}, {0x1ED01,0x1ED2D,4}, {0x1ED2F,0x1ED3D,4}, {0x1EE00,0x1EE03,3}, {0x1EE05,0x1EE1F,3},
	{0x1EE21,0x1EE22,3}, {0x1EE24,0x1EE24,3}, {0x1EE27,0x1EE27,3}, {0x1EE29,0x1EE32,3}, {0x1EE34,0x1EE37,3},
	{0x1EE39,0x1EE39,3}, {0x1EE3B,0x1EE3B,3}, {0x1EE42,0x1EE42,3}, {0x1EE47,0x1EE47,3}, {0x1EE49,0x1EE49,3},
	{0x1EE4B,0x1EE4B,3}, {0x1EE4D,0x1EE4F,3}, {0x1EE51,0x1EE52,3}, {0x1EE54,0x1EE54,3}, {0x1EE57,0x1EE57,3},
	{0x1EE59,0x1EE59,3}, {0x1EE5B,0x1EE5B,3}, {0x1EE5D,0x1EE5D,3}, {0x1EE5F,0x1EE5F,3}, {0x1EE61,0x1EE62,3},
	{0x1EE64,0x1EE64,3}, {0x1EE67,0x1EE6A,3}, {0x1EE6C,0x1EE72,3}, {0x1EE74,0x1EE77,3}, {0x1EE79,0x1EE7C,3},
	{0x1EE7E,0x1EE7E,3}, {0x1EE80,0x1EE89,3}, {0x1EE8B,0x1EE9B,3}, {0x1EEA1,0x1EEA3,3}, {0x1EEA5,0x1EEA9,3},
	{0x1EEAB,0x1EEBB,3}, {0x1F100,0x1F10C,4}, {0x1FBF0,0x1FBF9,4}, {0x20000,0x2A6DF,3}, {0x2A700,0x2B739,3},
	{0x2B740,0x2B81D,3}, {0x2B820,0x2CEA1,3}, {0x2CEB0,0x2EBE0,3}, {0x2EBF0,0x2EE5D,3}, {0x2F800,0x2FA1D,3},
	{0x30000,0x3134A,3}, {0x31350,0x323AF,3}, {0xE0100,0xE01EF,6},
};
static inline int cats_ucls(uint32_t c)
{
	size_t lo = 0, hi = sizeof(cats_ucls_tab) / sizeof(*cats_ucls_tab);
	while (lo < hi) {
		size_t mid = (lo + hi) / 2;
		if (c < cats_ucls_tab[mid][0]) hi = mid;
		else if (c > cats_ucls_tab[mid][1]) lo = mid + 1;
		else return (int)cats_ucls_tab[mid][2];
	}
	return 0;
}
// Decodes one code point at s[*i] (an invalid byte is returned as itself) and advances *i.
static inline uint32_t cats_u8_next(const uint8_t *s, size_t len, size_t *i)
{
	uint32_t c = s[*i];
	int l = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
	if (l == 1 || *i + (size_t)l > len) { (*i)++; return c; }
	uint32_t v = c & (0xFF >> (l + 1));
	for (int k = 1; k < l; k++) {
		if ((s[*i + k] & 0xC0) != 0x80) { (*i)++; return c; }
		v = (v << 6) | (s[*i + k] & 0x3F);
	}
	*i += (size_t)l;
	return v;
}

typedef struct {
	int n, nmerge, ignore_merges;
	const uint8_t **tok;		// token bytes (point into the blob)
	uint32_t *tlen;
	uint8_t *special;
	const uint32_t *merge;		// nmerge x 3
	int *vslot;  size_t vcap;	// bytes -> id (regular tokens only)
	int64_t *mkey; int *mval; size_t mcap;	// (a << 32 | b) -> rank
	int byte_id[256];
	int *sp; int nsp;		// special token ids, longest first
} cats_bpe;

static inline uint32_t cats_bpe_hash(const uint8_t *s, size_t n)
{
	uint32_t h = 2166136261u;
	for (size_t i = 0; i < n; i++) h = (h ^ s[i]) * 16777619u;
	return h;
}
static inline int cats_bpe_lookup(const cats_bpe *t, const uint8_t *s, size_t n)
{
	size_t i = cats_bpe_hash(s, n) & (t->vcap - 1);
	while (t->vslot[i] >= 0) {
		const int id = t->vslot[i];
		if (t->tlen[id] == n && !memcmp(t->tok[id], s, n)) return id;
		i = (i + 1) & (t->vcap - 1);
	}
	return -1;
}
static inline int cats_bpe_merge_rank(const cats_bpe *t, int a, int b)
{
	const int64_t k = ((int64_t)a << 32) | (uint32_t)b;
	size_t i = (size_t)((uint64_t)k * 0x9E3779B97F4A7C15ull >> 20) & (t->mcap - 1);
	while (t->mval[i] >= 0) {
		if (t->mkey[i] == k) return t->mval[i];
		i = (i + 1) & (t->mcap - 1);
	}
	return -1;
}
static inline void cats_bpe_free(cats_bpe *t)
{
	(free)(t->tok); (free)(t->tlen); (free)(t->special); (free)(t->vslot);
	(free)(t->mkey); (free)(t->mval); (free)(t->sp);
	memset(t, 0, sizeof(*t));
}
static inline uint32_t cats_bpe_rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
// Loads a blob (kept by reference: it must outlive the tokenizer). Returns 0, or -1 for a malformed blob.
static inline int cats_bpe_load(cats_bpe *t, const uint8_t *b, size_t size)
{
	memset(t, 0, sizeof(*t));
	if (size < 20 || memcmp(b, "CBPE", 4) || cats_bpe_rd32(b + 4) != 1) return -1;
	t->n = (int)cats_bpe_rd32(b + 8);
	t->nmerge = (int)cats_bpe_rd32(b + 12);
	t->ignore_merges = (int)(cats_bpe_rd32(b + 16) & 1);
	t->tok = (const uint8_t**)(calloc)((size_t)t->n, sizeof(*t->tok));
	t->tlen = (uint32_t*)(calloc)((size_t)t->n, sizeof(*t->tlen));
	t->special = (uint8_t*)(calloc)((size_t)t->n, 1);
	t->sp = (int*)(calloc)((size_t)t->n + 1, sizeof(int));
	if (!t->tok || !t->tlen || !t->special || !t->sp) { cats_bpe_free(t); return -1; }
	size_t off = 20;
	for (int i = 0; i < t->n; i++) {
		if (off + 5 > size) { cats_bpe_free(t); return -1; }
		const uint32_t l = cats_bpe_rd32(b + off);
		t->special[i] = b[off + 4];
		off += 5;
		if (off + l > size) { cats_bpe_free(t); return -1; }
		t->tok[i] = b + off;
		t->tlen[i] = l;
		off += l;
	}
	off = (off + 3) & ~(size_t)3;	// the merge table starts 4-byte aligned inside the blob
	if (off + (size_t)t->nmerge * 12 > size || (uintptr_t)(b + off) % 4) {
		// misaligned merges: fall back to a copy so that reads stay well defined
		uint32_t *m = (uint32_t*)(malloc)((size_t)t->nmerge * 12);
		if (!m || off + (size_t)t->nmerge * 12 > size) { (free)(m); cats_bpe_free(t); return -1; }
		memcpy(m, b + off, (size_t)t->nmerge * 12);
		t->merge = m;
	} else t->merge = (const uint32_t*)(b + off);
	for (t->vcap = 1024; t->vcap < (size_t)t->n * 2; t->vcap <<= 1) ;
	for (t->mcap = 1024; t->mcap < (size_t)t->nmerge * 2; t->mcap <<= 1) ;
	t->vslot = (int*)(malloc)(t->vcap * sizeof(int));
	t->mkey = (int64_t*)(malloc)(t->mcap * sizeof(int64_t));
	t->mval = (int*)(malloc)(t->mcap * sizeof(int));
	if (!t->vslot || !t->mkey || !t->mval) { cats_bpe_free(t); return -1; }
	for (size_t i = 0; i < t->vcap; i++) t->vslot[i] = -1;
	for (size_t i = 0; i < t->mcap; i++) t->mval[i] = -1;
	for (int i = 0; i < 256; i++) t->byte_id[i] = -1;
	for (int id = 0; id < t->n; id++) {
		if (t->special[id]) { if (t->tlen[id]) t->sp[t->nsp++] = id; continue; }
		size_t s = cats_bpe_hash(t->tok[id], t->tlen[id]) & (t->vcap - 1);
		while (t->vslot[s] >= 0) s = (s + 1) & (t->vcap - 1);
		t->vslot[s] = id;
		if (t->tlen[id] == 1) t->byte_id[t->tok[id][0]] = id;
	}
	for (int r = 0; r < t->nmerge; r++) {
		const int64_t k = ((int64_t)t->merge[3 * r] << 32) | t->merge[3 * r + 1];
		size_t s = (size_t)((uint64_t)k * 0x9E3779B97F4A7C15ull >> 20) & (t->mcap - 1);
		while (t->mval[s] >= 0 && t->mkey[s] != k) s = (s + 1) & (t->mcap - 1);
		if (t->mval[s] < 0) { t->mkey[s] = k; t->mval[s] = r; }	// first (highest priority) wins
	}
	// longest special tokens first, so that a prefix of another one does not shadow it
	for (int i = 1; i < t->nsp; i++) {
		const int v = t->sp[i];
		int j = i - 1;
		while (j >= 0 && t->tlen[t->sp[j]] < t->tlen[v]) { t->sp[j + 1] = t->sp[j]; j--; }
		t->sp[j + 1] = v;
	}
	return 0;
}

// Length of the pre-token starting at cp[i] (cp[0..n) code points). Implements
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |\p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* |\s*[\r\n]+ |\s+(?!\S) |\s+
// with the backtracking of a regex engine (leftmost alternative that matches wins).
static inline int cats_bpe_isA(uint32_t c) { const int k = cats_ucls(c); return k == 1 || k == 3 || k == 6; }
static inline int cats_bpe_isB(uint32_t c) { const int k = cats_ucls(c); return k == 2 || k == 3 || k == 6; }
static inline size_t cats_bpe_contraction(const uint32_t *cp, size_t n, size_t i)
{
	if (i >= n || cp[i] != '\'' || i + 1 >= n) return i;
	const uint32_t a = cp[i + 1] | 32, b = i + 2 < n ? (cp[i + 2] | 32) : 0;
	if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
	if ((a == 'r' || a == 'v') && b == 'e') return i + 3;
	if (a == 'l' && b == 'l') return i + 3;
	return i;
}
static inline size_t cats_bpe_word(const uint32_t *cp, size_t n, size_t i)
{
	const uint32_t c = cp[i];
	const int k = cats_ucls(c);
	size_t e;
	for (int alt = 1; alt <= 2; alt++) {
		for (int pre = 1; pre >= 0; pre--) {
			size_t s = i;
			if (pre) { if (c == '\r' || c == '\n' || k == 1 || k == 2 || k == 3 || k == 4) continue; s = i + 1; }
			size_t r = 0;
			while (s + r < n && cats_bpe_isA(cp[s + r])) r++;
			if (alt == 1) {
				size_t kk = r + 1;
				while (kk-- > 0) if (s + kk < n && cats_bpe_isB(cp[s + kk])) break;
				if (kk == (size_t)-1) continue;	// no split point: this alternative fails
				e = s + kk;
				while (e < n && cats_bpe_isB(cp[e])) e++;
			} else {
				if (r == 0) continue;
				e = s + r;
				while (e < n && cats_bpe_isB(cp[e])) e++;
			}
			return cats_bpe_contraction(cp, n, e);
		}
	}
	if (k == 4) { e = i; while (e < n && e < i + 3 && cats_ucls(cp[e]) == 4) e++; return e; }
	// ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
	{
		size_t s = i;
		if (c == ' ') s = i + 1;
		e = s;
		while (e < n && (cats_ucls(cp[e]) == 0 || cats_ucls(cp[e]) == 6)) e++;
		if (e > s) { while (e < n && (cp[e] == '\r' || cp[e] == '\n' || cp[e] == '/')) e++; return e; }
	}
	if (k == 5) {
		size_t w = i;
		while (w < n && cats_ucls(cp[w]) == 5) w++;
		size_t nl = w;	// \s*[\r\n]+ : up to and including the last line break of the run
		while (nl > i && cp[nl - 1] != '\r' && cp[nl - 1] != '\n') nl--;
		if (nl > i) return nl;
		if (w == n) return w;			// \s+(?!\S)
		if (w - i >= 2) return w - 1;
		return w;				// \s+
	}
	return i + 1;
}

// Byte-level BPE of one pre-token (raw bytes). Appends ids to ids[*n] (capacity cap; *n counts all of them).
static inline void cats_bpe_word_encode(const cats_bpe *t, const uint8_t *w, size_t len, int *ids, int cap, int *n)
{
	if (t->ignore_merges) {
		const int id = cats_bpe_lookup(t, w, len);
		if (id >= 0) { if (*n < cap) ids[*n] = id; (*n)++; return; }
	}
	enum { CHUNK = 4096 };
	for (size_t off = 0; off < len; off += CHUNK) {
		const size_t m = len - off < CHUNK ? len - off : CHUNK;
		int *sym = (int*)(malloc)(m * sizeof(int));
		if (!sym) return;
		size_t cnt = m;
		for (size_t i = 0; i < m; i++) sym[i] = t->byte_id[w[off + i]];
		while (cnt > 1) {
			int best = -1; size_t bi = 0;
			for (size_t i = 0; i + 1 < cnt; i++) {
				const int r = cats_bpe_merge_rank(t, sym[i], sym[i + 1]);
				if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
			}
			if (best < 0) break;
			sym[bi] = (int)t->merge[3 * best + 2];
			memmove(sym + bi + 1, sym + bi + 2, (cnt - bi - 2) * sizeof(int));
			cnt--;
		}
		for (size_t i = 0; i < cnt; i++) { if (*n < cap) ids[*n] = sym[i]; (*n)++; }
		(free)(sym);
	}
}
static inline int cats_bpe_encode_plain(const cats_bpe *t, const uint8_t *s, size_t len, int *ids, int cap)
{
	int n = 0;
	uint32_t *cp = (uint32_t*)(malloc)((len + 1) * sizeof(uint32_t));
	size_t *off = (size_t*)(malloc)((len + 2) * sizeof(size_t));
	if (!cp || !off) { (free)(cp); (free)(off); return 0; }
	size_t m = 0;
	for (size_t i = 0; i < len; ) { off[m] = i; cp[m++] = cats_u8_next(s, len, &i); }
	off[m] = len;
	for (size_t i = 0; i < m; ) {
		size_t e = cats_bpe_word(cp, m, i);
		if (e <= i) e = i + 1;
		cats_bpe_word_encode(t, s + off[i], off[e] - off[i], ids, cap, &n);
		i = e;
	}
	(free)(cp); (free)(off);
	return n;
}
// Encodes text to ids. parse_special: added tokens such as <|end_message|> written in the text become their id.
// Returns the number of ids (may exceed cap, then only cap were stored).
static inline int cats_bpe_encode(const cats_bpe *t, const char *text, int parse_special, int *ids, int cap)
{
	const uint8_t *s = (const uint8_t*)text;
	const size_t len = strlen(text);
	int n = 0;
	size_t start = 0;
	for (size_t i = 0; i <= len; ) {
		int hit = -1;
		if (parse_special && i < len && s[i] == '<') {
			for (int k = 0; k < t->nsp; k++) {
				const int id = t->sp[k];
				if (t->tlen[id] <= len - i && !memcmp(s + i, t->tok[id], t->tlen[id])) { hit = id; break; }
			}
		}
		if (hit >= 0 || i == len) {
			if (i > start) {
				int w = cats_bpe_encode_plain(t, s + start, i - start, ids + (n < cap ? n : cap), n < cap ? cap - n : 0);
				n += w;
			}
			if (hit < 0) break;
			if (n < cap) ids[n] = hit;
			n++;
			i += t->tlen[hit];
			start = i;
		} else i++;
	}
	return n;
}
// Bytes of token `id` (special tokens only if show_special); returns the length written (NUL terminated, cap bytes).
static inline size_t cats_bpe_piece(const cats_bpe *t, int id, int show_special, char *out, size_t cap)
{
	if (id < 0 || id >= t->n || (t->special[id] && !show_special) || cap < 2) return 0;
	const size_t n = t->tlen[id] < cap - 1 ? t->tlen[id] : cap - 1;
	memcpy(out, t->tok[id], n);
	out[n] = 0;
	return n;
}

// ---- building the blob from a Hugging Face tokenizer.json (used by the quantizers) ----
typedef struct { const char *p, *e; } cats_jr;
static inline void cats_jr_ws(cats_jr *r) { while (r->p < r->e && (*r->p == ' ' || *r->p == '\n' || *r->p == '\t' || *r->p == '\r')) r->p++; }
static inline int cats_jr_eat(cats_jr *r, char c) { cats_jr_ws(r); if (r->p < r->e && *r->p == c) { r->p++; return 1; } return 0; }
// Parses a JSON string into a malloc'ed UTF-8 buffer (NUL terminated); *len gets its length.
static inline char *cats_jr_str(cats_jr *r, size_t *len)
{
	cats_jr_ws(r);
	if (r->p >= r->e || *r->p != '"') return NULL;
	r->p++;
	size_t cap = (size_t)(r->e - r->p > 4096 ? 4096 : r->e - r->p) + 64, n = 0;
	char *o = (char*)(malloc)(cap);
	if (!o) return NULL;
	while (r->p < r->e && *r->p != '"') {
		if (n + 8 > cap) { cap *= 2; char *x = (char*)(realloc)(o, cap); if (!x) { (free)(o); return NULL; } o = x; }
		unsigned c = (unsigned char)*r->p++;
		if (c == '\\' && r->p < r->e) {
			c = (unsigned char)*r->p++;
			switch (c) {
			case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break;
			case 'b': c = '\b'; break; case 'f': c = '\f'; break;
			case 'u': {
				unsigned v = 0;
				for (int k = 0; k < 4 && r->p < r->e; k++) { char h = *r->p++; v = v * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10); }
				if (v >= 0xD800 && v < 0xDC00 && r->p + 6 <= r->e && r->p[0] == '\\' && r->p[1] == 'u') {
					unsigned w = 0;
					for (int k = 0; k < 4; k++) { char h = r->p[2 + k]; w = w * 16 + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10); }
					r->p += 6;
					v = 0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00);
				}
				c = v; break;
			}
			default: break;
			}
			n += (size_t)cats_utf8_put(o + n, c);
		} else o[n++] = (char)c;
	}
	if (r->p < r->e) r->p++;
	o[n] = 0;
	if (len) *len = n;
	return o;
}
static inline int cats_jr_skip(cats_jr *r)
{
	cats_jr_ws(r);
	if (r->p >= r->e) return -1;
	if (*r->p == '"') { char *s = cats_jr_str(r, NULL); (free)(s); return s ? 0 : -1; }
	if (*r->p == '{' || *r->p == '[') {
		const char open = *r->p, close = open == '{' ? '}' : ']';
		r->p++;
		if (cats_jr_eat(r, close)) return 0;
		do {
			if (open == '{') { char *k = cats_jr_str(r, NULL); if (!k) return -1; (free)(k); if (!cats_jr_eat(r, ':')) return -1; }
			if (cats_jr_skip(r)) return -1;
		} while (cats_jr_eat(r, ','));
		return cats_jr_eat(r, close) ? 0 : -1;
	}
	while (r->p < r->e && *r->p != ',' && *r->p != '}' && *r->p != ']') r->p++;
	return 0;
}
// GPT-2 byte <-> printable code point table: returns the byte for code point c, or -1.
static inline int cats_bytelevel_byte(uint32_t c)
{
	static int16_t inv[324];
	static int ready;
	if (!ready) {
		for (int i = 0; i < 324; i++) inv[i] = -1;
		int n = 0;
		for (int b = 0; b < 256; b++) {
			if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) inv[b] = (int16_t)b;
			else inv[256 + n++] = (int16_t)b;
		}
		ready = 1;
	}
	return c < 324 ? inv[c] : -1;
}
typedef struct { uint8_t *b; size_t n, cap; } cats_buf;
static inline int cats_buf_put(cats_buf *o, const void *p, size_t n)
{
	if (o->n + n > o->cap) {
		size_t c = o->cap ? o->cap : 1 << 20;
		while (c < o->n + n) c *= 2;
		uint8_t *x = (uint8_t*)(realloc)(o->b, c);
		if (!x) return -1;
		o->b = x; o->cap = c;
	}
	memcpy(o->b + o->n, p, n);
	o->n += n;
	return 0;
}
static inline void cats_buf_u32(cats_buf *o, uint32_t v) { uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) }; cats_buf_put(o, b, 4); }
// Converts tokenizer.json (BPE model with "vocab", "merges" and "added_tokens") into the blob described above.
// The vocabulary size is max(id) + 1; *out is malloc'ed. Returns 0 or -1.
static inline int cats_bpe_from_json(const char *json, size_t size, uint8_t **out, size_t *out_size)
{
	cats_jr r = { json, json + size };
	int maxid = -1, ignore_merges = 0, ok = -1;
	char **toks = NULL; size_t *tl = NULL; uint8_t *special = NULL;
	size_t cap = 0;
	cats_buf blob = { 0 };
	const char *vocab_pos = NULL, *merges_pos = NULL, *added_pos = NULL;
	// pass 1: locate the pieces
	if (!cats_jr_eat(&r, '{')) return -1;
	do {
		char *k = cats_jr_str(&r, NULL);
		if (!k || !cats_jr_eat(&r, ':')) { (free)(k); return -1; }
		if (!strcmp(k, "added_tokens")) { cats_jr_ws(&r); added_pos = r.p; }
		if (!strcmp(k, "model")) {
			cats_jr_ws(&r);
			cats_jr m = r;
			if (!cats_jr_eat(&m, '{')) { (free)(k); return -1; }
			do {
				char *mk = cats_jr_str(&m, NULL);
				if (!mk || !cats_jr_eat(&m, ':')) { (free)(mk); (free)(k); return -1; }
				cats_jr_ws(&m);
				if (!strcmp(mk, "vocab")) vocab_pos = m.p;
				else if (!strcmp(mk, "merges")) merges_pos = m.p;
				else if (!strcmp(mk, "ignore_merges")) ignore_merges = !strncmp(m.p, "true", 4);
				(free)(mk);
				if (cats_jr_skip(&m)) { (free)(k); return -1; }
			} while (cats_jr_eat(&m, ','));
		}
		(free)(k);
		if (cats_jr_skip(&r)) return -1;
	} while (cats_jr_eat(&r, ','));
	if (!vocab_pos || !merges_pos) return -1;
#define CATS_BPE_SET(id, str, l, sp) do { \
		if ((size_t)(id) >= cap) { size_t nc = cap ? cap : 4096; while (nc <= (size_t)(id)) nc *= 2; \
			toks = (char**)(realloc)(toks, nc * sizeof(char*)); tl = (size_t*)(realloc)(tl, nc * sizeof(size_t)); \
			special = (uint8_t*)(realloc)(special, nc); \
			for (size_t q = cap; q < nc; q++) { toks[q] = NULL; tl[q] = 0; special[q] = 0; } cap = nc; } \
		(free)(toks[id]); toks[id] = (str); tl[id] = (l); special[id] = (sp); if ((id) > maxid) maxid = (id); } while (0)
	// vocab: map the byte-level characters back to raw bytes
	r.p = vocab_pos;
	if (!cats_jr_eat(&r, '{')) goto done;
	if (!cats_jr_eat(&r, '}')) do {
		size_t l;
		char *s = cats_jr_str(&r, &l);
		if (!s || !cats_jr_eat(&r, ':')) { (free)(s); goto done; }
		cats_jr_ws(&r);
		int id = (int)strtol(r.p, (char**)&r.p, 10);
		uint8_t *raw = (uint8_t*)(malloc)(l + 1);
		size_t rn = 0;
		for (size_t i = 0; i < l; ) { size_t j = i; uint32_t c = cats_u8_next((const uint8_t*)s, l, &j); int b = cats_bytelevel_byte(c); raw[rn++] = (uint8_t)(b < 0 ? '?' : b); i = j; }
		(free)(s);
		CATS_BPE_SET(id, (char*)raw, rn, 0);
	} while (cats_jr_eat(&r, ','));
	// added tokens are stored literally and matched in the text
	if (added_pos) {
		r.p = added_pos;
		if (cats_jr_eat(&r, '[') && !cats_jr_eat(&r, ']')) do {
			if (!cats_jr_eat(&r, '{')) goto done;
			int id = -1; char *content = NULL; size_t cl = 0;
			do {
				char *k = cats_jr_str(&r, NULL);
				if (!k || !cats_jr_eat(&r, ':')) { (free)(k); goto done; }
				cats_jr_ws(&r);
				if (!strcmp(k, "id")) id = (int)strtol(r.p, (char**)&r.p, 10);
				else if (!strcmp(k, "content")) content = cats_jr_str(&r, &cl);
				else if (cats_jr_skip(&r)) { (free)(k); goto done; }
				(free)(k);
			} while (cats_jr_eat(&r, ','));
			if (!cats_jr_eat(&r, '}') || id < 0 || !content) goto done;
			CATS_BPE_SET(id, content, cl, 1);
		} while (cats_jr_eat(&r, ','));
	}
	{
		const int n = maxid + 1;
		for (int i = 0; i < n; i++) if (!toks[i]) { toks[i] = (char*)(malloc)(1); tl[i] = 0; special[i] = 1; }
		cats_buf_put(&blob, "CBPE", 4);
		cats_buf_u32(&blob, 1); cats_buf_u32(&blob, (uint32_t)n);
		const size_t nm_at = blob.n;
		cats_buf_u32(&blob, 0); cats_buf_u32(&blob, ignore_merges ? 1 : 0);
		for (int i = 0; i < n; i++) {
			cats_buf_u32(&blob, (uint32_t)tl[i]);
			cats_buf_put(&blob, &special[i], 1);
			cats_buf_put(&blob, toks[i], tl[i]);
		}
		while (blob.n % 4) { uint8_t z = 0; cats_buf_put(&blob, &z, 1); }	// aligned merge table
		// token lookup for the merges (regular tokens only)
		size_t vc = 1024;
		while (vc < (size_t)n * 2) vc <<= 1;
		int *slot = (int*)(malloc)(vc * sizeof(int));
		for (size_t i = 0; i < vc; i++) slot[i] = -1;
		for (int i = 0; i < n; i++) if (!special[i]) {
			size_t s = cats_bpe_hash((const uint8_t*)toks[i], tl[i]) & (vc - 1);
			while (slot[s] >= 0) s = (s + 1) & (vc - 1);
			slot[s] = i;
		}
#define CATS_BPE_FIND(res, p, l) do { size_t s_ = cats_bpe_hash((const uint8_t*)(p), (l)) & (vc - 1); (res) = -1; \
		while (slot[s_] >= 0) { int c_ = slot[s_]; if (tl[c_] == (l) && !memcmp(toks[c_], (p), (l))) { (res) = c_; break; } s_ = (s_ + 1) & (vc - 1); } } while (0)
		r.p = merges_pos;
		uint32_t nmerge = 0;
		if (cats_jr_eat(&r, '[') && !cats_jr_eat(&r, ']')) do {
			char *a = NULL, *b = NULL; size_t al = 0, bl = 0;
			if (cats_jr_eat(&r, '[')) {	// ["a","b"]
				a = cats_jr_str(&r, &al); cats_jr_eat(&r, ','); b = cats_jr_str(&r, &bl); cats_jr_eat(&r, ']');
			} else {			// "a b"
				size_t l; char *s = cats_jr_str(&r, &l);
				if (s) {
					char *sp = l > 1 ? (char*)memchr(s + 1, ' ', l - 1) : NULL;
					if (sp) {
						al = (size_t)(sp - s); bl = l - al - 1;
						a = (char*)(malloc)(al + 1); memcpy(a, s, al); a[al] = 0;
						b = (char*)(malloc)(bl + 1); memcpy(b, sp + 1, bl); b[bl] = 0;
					}
					(free)(s);
				}
			}
			if (!a || !b) { (free)(a); (free)(b); (free)(slot); goto done; }
			// the strings are in the byte-level alphabet: convert to raw bytes
			uint8_t *ra = (uint8_t*)(malloc)(al + bl + 2), *rb = ra + al + 1;
			size_t na = 0, nb = 0;
			for (size_t i = 0; i < al; ) { uint32_t c = cats_u8_next((const uint8_t*)a, al, &i); int x = cats_bytelevel_byte(c); ra[na++] = (uint8_t)(x < 0 ? '?' : x); }
			for (size_t i = 0; i < bl; ) { uint32_t c = cats_u8_next((const uint8_t*)b, bl, &i); int x = cats_bytelevel_byte(c); rb[nb++] = (uint8_t)(x < 0 ? '?' : x); }
			int ia, ib, im;
			CATS_BPE_FIND(ia, ra, na); CATS_BPE_FIND(ib, rb, nb);
			uint8_t *cat = (uint8_t*)(malloc)(na + nb + 1);
			memcpy(cat, ra, na); memcpy(cat + na, rb, nb);
			CATS_BPE_FIND(im, cat, na + nb);
			(free)(cat); (free)(ra); (free)(a); (free)(b);
			if (ia < 0 || ib < 0 || im < 0) continue;	// cannot happen for a consistent tokenizer
			cats_buf_u32(&blob, (uint32_t)ia); cats_buf_u32(&blob, (uint32_t)ib); cats_buf_u32(&blob, (uint32_t)im);
			nmerge++;
		} while (cats_jr_eat(&r, ','));
		(free)(slot);
		blob.b[nm_at] = (uint8_t)nmerge; blob.b[nm_at + 1] = (uint8_t)(nmerge >> 8);
		blob.b[nm_at + 2] = (uint8_t)(nmerge >> 16); blob.b[nm_at + 3] = (uint8_t)(nmerge >> 24);
		*out = blob.b; *out_size = blob.n;
		blob.b = NULL;
		ok = 0;
	}
done:
	for (size_t i = 0; i < cap; i++) (free)(toks[i]);
	(free)(toks); (free)(tl); (free)(special); (free)(blob.b);
	return ok;
}

// ---- tiny helpers for config.json style files ----
// Reads a whole file into a malloc'ed, NUL terminated buffer (NULL on failure).
static inline char *cats_read_file(const char *path, size_t *size)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *b = (char*)(malloc)((size_t)n + 1);
	if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); (free)(b); return NULL; }
	b[n] = 0;
	fclose(f);
	if (size) *size = (size_t)n;
	return b;
}
// Finds "key": <number> anywhere in a JSON text (flat lookup: the first occurrence wins). Returns 0 if found.
static inline int cats_json_num(const char *j, const char *key, double *v)
{
	char k[96];
	snprintf(k, sizeof(k), "\"%s\"", key);
	const char *p = strstr(j, k);
	if (!p) return -1;
	p = strchr(p + strlen(k), ':');
	if (!p) return -1;
	char *e;
	double x = strtod(p + 1, &e);
	if (e == p + 1) return -1;
	*v = x;
	return 0;
}
// Reads "key": [n, n, ...] into out (up to cap numbers); returns the count, or -1 if the key is missing.
static inline int cats_json_nums(const char *j, const char *key, double *out, int cap)
{
	char k[96];
	snprintf(k, sizeof(k), "\"%s\"", key);
	const char *p = strstr(j, k);
	if (!p || !(p = strchr(p + strlen(k), '['))) return -1;
	int n = 0;
	for (p++; *p && *p != ']'; ) {
		char *e;
		double x = strtod(p, &e);
		if (e == p) { p++; continue; }
		if (n < cap) out[n] = x;
		n++;
		p = e;
	}
	return n;
}

#endif /* CATSEYE_H */

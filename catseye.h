//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <time.h>
#include <sys/time.h>
inline float time_diff(struct timespec *start, struct timespec *end)
{
	return (end->tv_sec - start->tv_sec) + 1e-9*(end->tv_nsec - start->tv_nsec);
}

#define _debug(...)	{ printf("%s(%d):", __func__, __LINE__); printf(__VA_ARGS__); }

#if 1
#if defined(_MSC_VER) || defined(__MINGW32__)
#define malloc(size)	_aligned_malloc(size, 16)
#define free(p)		_aligned_free(p)
#else
#define malloc(size)	({ void* p; posix_memalign((void**) &p, 16, size)==0 ? p : NULL; })
#define free(p)		free(p)
#endif  /* _MSC_VER */
#define calloc(n, size)	({ void* p = malloc((n*size)); memset(p, 0, (n*size))!=0 ? p : NULL; })
#endif

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
inline void gemm_rnn(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
inline void gemm_rnt(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
inline void gemm_rtn(int M, int N, int K, real alpha, real* restrict A, real* restrict B, real beta, real* restrict C)
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
#if __WORDSIZE == 64
typedef unsigned long int	uint64_t;
#else
__extension__
typedef unsigned long long int	uint64_t;
#endif
// The state must be seeded so that it is not everywhere zero.
uint64_t seed[2];
void xor128_init(uint64_t s)
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

// xoroshiro generator taken from http://vigna.di.unimi.it/xorshift/xoroshiro128plus.c
uint64_t xoroshiro_s[2] = {
    0X922AC4EB35B502D9L,
    0XDA3AA4832B8F1D27L
};
void xoroshiro128plus_init(uint64_t s)
{
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
//#define CATS_USE_XOR128
#ifdef CATS_USE_XOR128
#define xrand()			xor128()
#define frand()			( xor128() / (XOR128_MAX+1.0) )
#else
#define xrand()			xoroshiro128plus()
#define frand()			( xoroshiro128plus() / (XOR128_MAX+1.0) )
#endif
#define _rand(max)		(int)( frand() * max)
#define random(min, max)	( frand() * (max -min) +min )
#define irand(min, max)		( (xrand() % (max -min +1)) +min )
// http://www.natural-science.or.jp/article/20110404234734.php (mt19937ar.h)
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
	int fix;		// training / no training

	int wtype;		// weight init type
	real wrange;		// weight init

	int ksize;		// CNN
	int stride;		// CNN
	int padding;		// CNN
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
		l->w[i] -= l->eta * (l->dw[i] +/*l->p->lambda*/0.0 *l->w[i]); // L2 norm
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
	gemm_rtn(l->outputs, l->inputs, l->p->batch, 1, l->dOut, l->x, 1, l->dw);
//	SOLVER(l);
	for (int n=0; n<l->p->batch; n++) {
		for (int i=0; i<l->outputs; i++) l->w[l->inputs*l->outputs +i] -= l->eta * l->dOut[n*l->outputs +i]; // bias!!
	}
//	for (int i=0; i<10; i++) printf("%f ", l->dOut[i]);
/*	for (int i=0; i<10; i++) printf("%f ", l->w[i]);
	printf("\n");*/

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
			workspace = l->workspace +l->ox*l->oy*l->ksize*l->ksize*l->ich *i;
			im2col(l->x +l->inputs*i, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, workspace);
		}
		// z = W * x [A(m,k) B(k,n) C(m,n)], cnhw
		gemm_rnn(l->ch, l->ox*l->oy*1, l->ksize*l->ksize*l->ich, 1, l->w, workspace, 0, l->z +l->outputs*i);

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
		gemm_rtn(l->ksize*l->ksize*l->ich, l->ox*l->oy*1, l->ch, 1, l->w, l->dOut +l->outputs*i, 0, workspace);
		/*printf("gemm_rtn: ");
		if (i==l->p->batch-1) for (int i=0; i<10; i++) printf("%f ", workspace[i]);
		printf("\n");*/
		if (l->ksize!=1) {
			col2im(workspace, l->ich, l->sy, l->sx, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->dIn +l->inputs*i);
//			for (int n=0; n<10; n++) printf("%f ", l->dIn[l->inputs*i + l->inputs-1-n]);
//			printf("[%d]\n", l->wsize);
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
		real *workspace = l->ksize!=1 ? l->workspace +l->ox*l->oy*l->ksize*l->ksize*l->ich *i : l->x +l->inputs*i;
		// W = W - eta * dOut * x**T [A(m,k) B(k,n) C(m,n)]
//		gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*1, 1, l->dOut +l->outputs*i, workspace, 1, l->dw);
		gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*1, 1.0/l->p->batch, l->dOut +l->outputs*i, workspace, 1, l->dw);
//		for (int i=0; i<100; i++) printf("%f ", l->dw[l->wsize-1-i]);
//		printf("\n");
	}
//	for (int i=0; i<l->wsize; i++) l->dw[i] /= l->p->batch; // avg
//	for (int i=0; i<10; i++) printf("%f ", l->dOut[i +l->outputs*(l->p->batch-1)]);
//	for (int i=0; i<10; i++) printf("%f ", l->dw[i]);
//	printf("w:%d %d\n", l->wsize, l->ch*l->ksize*l->ksize*l->ich);
//	SOLVER(l);
#else
	real *workspace = l->ksize!=1 ? l->workspace : l->x;
	gemm_rnt(l->ch, l->ksize*l->ksize*l->ich, l->ox*l->oy*l->p->batch, -l->eta, l->dOut, workspace, 0, l->dw);
#endif
}

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
		im2col(l->dOut +l->outputs*i, l->ch, l->oy, l->ox, l->ksize, l->ksize, l->padding, l->padding, l->stride, l->stride, l->p->mem);
		gemm_rnn(l->ich, l->sx*l->sy*1, l->ksize*l->ksize*l->ch, 1, l->w, l->p->mem, 0, l->dIn +i*l->inputs);
	}
}
static void CatsEye_deconvolutional_update(CatsEye_layer *l)
{
	for (int i=0; i<l->p->batch; i++) {
		gemm_rnt(l->ich, l->ksize*l->ksize*l->ch, l->sx*l->sy*1, 1, l->dOut +l->outputs*i, l->p->mem, 1, l->dw);
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
static void CatsEye_BatchNormalization_forward(CatsEye_layer *l)
{
//	if (l->p->batch<4) return;

	int len = l->p->batch *l->inputs /l->r/*ch*/;
	real *x = l->x;
	real *z = l->z;

	for (int n=0; n<l->r; n++) {
		// average/mean
		real avg = 0;
		for (int i=0; i<len; i++) avg += x[i];
		avg /= len;

		// variance
		real var = 0;
		for (int i=0; i<len; i++) {
			z[i] = x[i] -avg;
			var += z[i] * z[i];
		}
		var /= len;
		real sigma = sqrt(var + 1e-8);

		// normalize, scale and shift
		for (int i=0; i<len; i++) {
			z[i] = (z[i] / sigma) * l->gamma + l->beta;
		}

		l->mu[n] = avg;
		l->var[n] = var;

		x += len;
		z += len;
	}
}
// https://deepnotes.io/batchnorm
static void CatsEye_BatchNormalization_backward(CatsEye_layer *l)
{
//	if (l->p->batch<4) return;

	int len = l->p->batch *l->inputs /l->r;
	real *x = l->x;
	real *delta = l->dOut;
	real *d = l->dIn;

	real dgamma = 0;
	real dbeta = 0;

	for (int n=0; n<l->r; n++) {
		real sigma = sqrt(l->var[n] + 1e-8);
		real dvar = 0;
		real dmu = 0;
		real dmu2 = 0;

		for (int i=0; i<len; i++) {
			real X_mu = x[i] - l->mu[n];

			real dX_norm = delta[i] * l->gamma;
			dvar += dX_norm * X_mu;
			d[i] = dX_norm / sigma;
			dmu += - d[i]; //dX_norm / -sigma;
			dmu2 += -2.0 * X_mu;

			dbeta += delta[i];
			dgamma = delta[i] * (x[i] -l->mu[n]) /sqrt(l->var[n] + 1e-8);
		}
		dvar *= -0.5 * pow(l->var[n] + 1e-8, -3.0/2.0);
		dmu += dvar * dmu2/len;

		real a = dmu / len;
		real b = dvar * 2.0 / len;
		for (int i=0; i<len; i++) {
			d[i] += a + b * (x[i] - l->mu[n]);
		}

//		l->beta[n] -= l->eta * dbeta;
//		l->gamma[n] -= l->eta * dgamma;
		l->beta -= l->eta * dbeta;
		l->gamma -= l->eta * dgamma;

		x += len;
		delta += len;
		d += len;
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
			d2 += l->inputs;
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
//	for (int b=0; b<l->p->batch; b++) for (int n=0; n<10; n++) printf("%f[%f] ", l->dIn[b*l->inputs +l->inputs-1-n], l->z[b*l->inputs +l->inputs-1-n]);\
//	printf("%d\n", l->wsize);\

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
#define CATS_DACT_log(y, l)		(1.0/y)
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
//	for (int n=0; n<l->inputs; n++) printf(" %f", l->dIn[n]);
//	printf("\n");
//	l->dIn[a] -= 1;
	int16_t *a = l->p->clasify;
	for (int i=0; i<l->p->batch; i++) {
		l->dIn[l->inputs*i + *a++] -= 1;
/*		l->dIn[l->inputs*i + *a] -= 1;
		l->dIn[l->inputs*i + *a++] /= l->p->batch; // FIXME: right?*/
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
//		printf("(%f-%f=%f) ", *y, *t, *y-*t);
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
		loss += - *t * log(*y) - (1 - *t) * log(1 - *y);
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
		*d++ = (*y - *t++) / (*y * (1 - *y));	// -t * log(y) - (1 - t) * log(1 - y);
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
	loss /= l->inputs *l->p->batch;
//	loss /= l->p->batch;
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
	memcpy(l->dIn, l->x, sizeof(real)*l->inputs *l->p->batch);
	int16_t *a = l->p->clasify;
	for (int i=0; i<l->p->batch; i++) {
		l->dIn[l->inputs*i + *a] = -1 / l->x[l->inputs*i + *a];
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
	CATS_LINEAR, CATS_CONV, CATS_DECONV, CATS_MAXPOOL, CATS_AVGPOOL, CATS_GAP,
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
char CatsEye_string[][16] = {
	"dense", "conv", "deconv", "max", "avg", "gap",
	"subpixel", /*"rnn",*/
	"pad", "bn",

	"concat", "shortcut",

	"sigmoid", "softmax", "tanh",
	"relu", "leaky", "elu", "rrelu",
	"log",

	"binary", "mse", "bce", "ce",

	"identity/mse", "sigmoid/bce", "softmax/ce",
};

int CatsEye_getLayer(CatsEye *this, char *name)
{
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

		n[i] = m[i] = b[i] = 0;
		switch (l->type) {
		case CATS_CONV:
			l->ox = /*l->px =*/ (l->sx +2*l->padding -l->ksize) /l->stride +1;
			l->oy = /*l->py =*/ (l->sy +2*l->padding -l->ksize) /l->stride +1;
/*			l->px -= l->padding*2;
			l->py -= l->padding*2;
			l->pz = l->padding +l->padding*l->ox;*/
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ksize * l->ksize;	// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->wsize = n[i] * m[i];
			l->workspace = malloc(sizeof(real)* l->ox*l->oy*l->ksize*l->ksize*l->ich *this->batch);
			printf("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
//			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) printf("\n\t↑ warning: stride is strange!\n");
			break;
		case CATS_DECONV: // https://blog.shikoan.com/pytorch-convtranspose2d/
			l->ox = (l->sx-1) *l->stride -2*l->padding +l->ksize;
			l->oy = (l->sy-1) *l->stride -2*l->padding +l->ksize;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ksize * l->ksize;	// kernel size
			m[i] = l->ch * l->ich;		// channel
			l->wsize = n[i] * m[i];
			printf("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
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
			printf("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			if ((l->sx +2*l->padding -l->ksize) % l->stride > 0) printf("\n\t↑ warning: kernel or stride is strange!\n");
			break;
		case CATS_GAP:
			l->outputs = l->ch = l->ich;
			l->ox = 1;
			l->oy = 1;
			printf("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_PIXELSHUFFLER:
			if (l->ch * l->r*l->r != l->ich) {
				l->sx = l->sy = (int)sqrt(l->inputs/ (l->ch * l->r*l->r));
			}
			l->ich = l->ch * l->r*l->r;
			l->ox = l->sx * l->r;
			l->oy = l->sy * l->r;
			l->outputs = l->ch * l->ox * l->oy;
			printf("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
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
			printf("%3d %-12s %4d %dx%d/%d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->ksize, l->ksize, l->stride, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;*/

		case CATS_PADDING:
			l->ch = l->ich;
			l->ox = l->sx + l->padding*2;
			l->oy = l->sy + l->padding*2;
			l->outputs = l->ch * l->ox * l->oy;
			printf("%3d %-12s %4d     %d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->ch, l->padding, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_BATCHNORMAL:
			l->gamma = 1.0;
			l->beta = 0;
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			l->r = l->sx==1? 1: l->ch;
			n[i] = l->r;
			m[i] = 2; // mu, var, gamma, beta
//			m[i] = 4; // mu, var, gamma, beta
			l->wsize = n[i] * m[i];
			printf("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_SHORTCUT:
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
			l->ch = l->ich;
			l->outputs = l->inputs;
			l->ox = l->sx;
			l->oy = l->sy;
			printf("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
			break;

		case CATS_CONCAT:
			l->l = &this->layer[CatsEye_getLayer(this, l->layer)];
		case CATS_LINEAR:
		case CATS_LOSS_0_1:
		case CATS_LOSS_MSE:
		default:
			if (i<this->layers-1 && !l->outputs) {
				if ((l+1)->inputs>0) l->outputs = (l+1)->inputs;
				else { l->outputs = l->inputs; printf("\t↓ warning: out:%d\n", l->outputs); }
			}
			if (l->type==CATS_LINEAR) {
				l->wsize = l->inputs * l->outputs;
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
			printf("%3d %-12s %10d %4d x%4d x%4d -> %4d x%4d x%4d", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich, l->ox, l->oy, l->ch);
//			printf("%3d %-12s %10d %4d x%4d x%4d -> loss", i+1, CatsEye_string[l->type], l->inputs, l->sx, l->sy, l->ich);
		}
		if (l->name) printf(" <%s>\n", l->name);
		else if (l->layer) printf(" +<%s:%d>\n", l->layer, l->r);
		else printf("\n");
		if (!l->inputs) {
			printf("\t↑ warning: input is strange!\n");
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
			if ((l->type==CATS_SOFTMAX_CE && (l-1)->type==CATS_ACT_SOFTMAX) ||
				(l->type==CATS_SIGMOID_BCE && (l-1)->type==CATS_ACT_SIGMOID)) {
				// FIXME: for loss function
				// https://www.renom.jp/ja/notebooks/tutorial/basic_algorithm/lossfunction/notebook.html
				(l-1)->backward = CatsEye_none;
				(l-2)->dOut = l->dIn;
			}
			break;
		}
/*		if (l->type == CATS_BATCHNORMAL) {
			printf("+!\n");
			for (int n=0; n<l->inputs; n++) {
			printf("%d\n",n);
				l->gamma[n] = 1.0;
				l->beta[n] = 0;
			}
		}*/
		if (!l->wsize) { /*l->dw = 0;*/ continue; }

		// initialize weights, range depends on the research of Y. Bengio et al. (2010)
		// http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
		// https://ntacoffee.com/xavier-initialization/
typedef enum {
	CATS_RAND, /*CATS_UNIFORM,*/ CATS_GLOROT_UNIFORM/*Xavier*/, CATS_GLOROT_NORMAL, CATS_HE_UNIFORM/*Kaiming*/, CATS_HE_NORMAL
} CATS_WEIGHT;
		xor128_init(time(0));
		xoroshiro128plus_init(time(0));
		real range = sqrt(2)/sqrt(l->inputs+l->outputs);
		// https://pytorch.org/docs/stable/nn.init.html
		real gain = 1;
		int fin = l->ksize ? l->ksize*l->ksize*l->ich : l->inputs;
		if (i<this->layers-1) {
			int n = (l+1)->type==CATS_BATCHNORMAL ? 2 : 1;
			switch ((l+n)->type) {
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
			range = gain * sqrt(6 / fin);
//			range = gain * sqrt(3 / (l->inputs/l->ich));
			for (int n=0; n<l->wsize; n++) l->w[n] = 2.0*range*frand()-range; // uniform
			break;
		case CATS_HE_NORMAL: // ReLU
			range = gain * sqrt(2 / fin);
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
	}
	this->clasify = (int16_t*)this->layer[this->layers-1].z;
	this->label = this->layer[this->layers-1].z;
	printf("\n");

	uint64_t max = 0;
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		printf("L%02d in:%d out:%d weight:%d[%f] (z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->wsize, l->wrange, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
//		printf("L%02d in:%d out:%d weight:%d (x:%ld-z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->wsize, l->x-this->odata, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
		//printf("  (dIn:%ld-dOut:%ld)\n", l->dIn, l->dOut);

		uint64_t s = l->ox*l->oy*l->ksize*l->ksize*l->ich; // col2im
		if (max < s) max = s;
	}
	this->mem = calloc(max/**this->batch*/, sizeof(real));
	uint64_t wmem = sizeof(real) * this->osize*this->batch+this->dsize*this->batch+this->wsize*3;
	printf("Memory: %.1f MiB [%lu B], Working Memory: %.1f MiB [%lu B]\n\n", sizeof(real)*this->wsize/1024/1024., sizeof(real)*this->wsize, wmem/1024/1024., wmem);

	this->start = this->stop = 0;
	this->end = this->layers-1;
	this->slide = this->layer[0].inputs;

	sgemm_init(wmem);
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	sgemm_finish();

	// delete arrays
//	printf("%x %x %x %x %x %x %x %x %x\n",this->z,this->odata,this->o,this->ddata,this->d,this->ws,this->wdata,this->w,this->u);
	if (this->shuffle_buffer) free(this->shuffle_buffer);
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

	CatsEye_layer *l = &this->layer[this->start];
	memcpy(l->x, x, l->inputs*sizeof(real));

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

static inline void _CatsEye_DA_zoom(real *p, real *s, int w, int h, int ch, real r)
{
	for (int c=0; c<ch; c++) {
		for (int y=0; y<h; y++) {
			for (int x=0; x<w; x++) {
				*p++ = s[(int)(y/r*w +x/r)];
			}
		}
		s += w*h;
	}
}
static inline void _CatsEye_DA_translationXY(real *p, real *s, int w, int h, int ch, int px, int py)
{
	for (int c=0; c<ch; c++) {
		for (int y=0; y<py; y++) {
//			memcpy(p, s+(py-y)*sx+px, (sx-px)*sizeof(real));
//			for (int x=0; x<px; x++) p[px+x] = s[(py-y)*sx +px-x];
			for (int x=0; x<px; x++) p[x] = s[(py-y)*w +px-x];
			memcpy(p+px, s+(py-y)*w, (w-px)*sizeof(real));
			p += w;
		}
		for (int y=0; y<(h-py); y++) {
//			memcpy(p, s+y*w+px, (w-px)*sizeof(real));
//			for (int x=0; x<px; x++) p[px+x] = s[y*w +px-x];
			for (int x=0; x<px; x++) p[x] = s[(py-y)*w +px-x];
			memcpy(p+px, s+(py-y)*w, (w-px)*sizeof(real));
			p += w;
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
			p[i] = s[i] +rand_normal(0, sigma) -sigma/2;
		}
		break;
	case CATS_ZOOM_IN: // zoom in
		_CatsEye_DA_zoom(p, s, sx, sy, ch, random(1, 2));
		break;
	case CATS_ZOOM_OUT: // zoom out
		_CatsEye_DA_zoom(p, s, sx, sy, ch, random(0, 1));
		break;
	case CATS_TRANSLATION: // translation
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, irand(0, sx), irand(0, sy));
		break;
	case CATS_TRANSLATION_X: // translationX
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, irand(0, sx), sy/*0?*/);
		break;
	case CATS_TRANSLATION_Y: // translationY
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, 0, irand(0, sy));
		break;
	case CATS_FLIP:
		_CatsEye_DA_translationXY(p, s, sx, sy, ch, sx, sy/*0?*/);
		break;

	case CATS_ORIGINAL: // original
	default:
		memcpy(p, s, sx*sy*ch*sizeof(real));
	}
}
static inline void _CatsEye_shuffle(int *array, int n)
{
	for (int i=0; i<n-1; i++) {
//		int j = i + rand() / (RAND_MAX / (n - i) + 1);
		int j = i + (int)(xrand() / (XOR128_MAX / (n-i) +1));
		if (j>=n) printf(" %d\n",j); // FIXME: err in espcn.c at here??
		int t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}
static inline void _CatsEye_data_transfer(CatsEye *this, real *x, void *l, int n)
{
	if (!this->shuffle_buffer) this->shuffle_buffer = malloc(n * sizeof(int));
	for (int i=0; i<n; i++) this->shuffle_buffer[i] = i;
	_CatsEye_shuffle(this->shuffle_buffer, n);
	this->shuffle_base = 0;

	this->label_size = this->layer[this->end].inputs *sizeof(real);
	if (this->layer[this->end].type==CATS_LOSS_0_1 ||
		this->layer[this->end].type==CATS_SOFTMAX_CE /*|| this->layer[this->end].type==CATS_SIGMOID_BCE*/) {
		this->label_size = sizeof(int16_t);
	}

	this->data_num = n;
	this->learning_data = x;
	this->label_data = l;
}
static inline void _CatsEye_forward(CatsEye *this)
{
	CatsEye_layer *l = &this->layer[this->start];

	// create data for every batch
	int lsize = this->label_size;
	int8_t *label = (int8_t*)this->label;
	int8_t *label_data = (int8_t*)this->label_data;
	for (int b=0; b<this->batch; b++) {
		int sample = this->shuffle_buffer[this->shuffle_base];
		memcpy(l->x +l->inputs*b, this->learning_data+sample*this->slide, l->inputs*sizeof(real));
//		_CatsEye_DataAugmentation(l->x +l->inputs*b, this->learning_data+sample*this->slide, l->sx, l->sy, l->ich, this->da);
		memcpy(label +lsize*b, label_data+lsize*sample, lsize/*bytes*/);

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
		if (l->wsize) SOLVER(l);
#ifdef CATS_CHECK
		clock_gettime(CLOCK_REALTIME, &stop);
		l->backward_time = (l->backward_time + time_diff(&start, &stop)) * 0.5;
#endif
		l--;
	}
}
static inline void _CatsEye_update(CatsEye *this)
{
/*	CatsEye_layer *l = &this->layer[this->end];
	for (int i=this->end; i>=this->start; i--) {
		if (l->wsize) SOLVER(l);
		l--;
	}*/
}
static inline void _CatsEye_zero_grad(CatsEye *this)
{
	//memset(this->ddata, 0, this->dsize*this->batch*sizeof(real)); // clear all l->d (zero_grad)
	memset(this->wdata +this->wsize, 0, this->wsize*sizeof(real)); // clear all l->dw
}
int CatsEye_train(CatsEye *this, real *x, void *t, int N, int epoch, int random, int verify)
{
	int a = this->end;	// layers-1
	this->layer[a].z = t;	// FIXME

	if (verify) N -= verify;	// for test
	int repeat = N;			// for random
	if (random) repeat = random;
	if (this->batch>1) {
		repeat /= this->batch;
		printf(" batch: %d, repeat: %d\n", this->batch, repeat);
	}
	printf("epoch    loss     elapsed time\n");

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
		printf("%7d, %f %f [%.2fs]", times, this->loss, err, time_diff(&start, &stop));

		if (verify) {
			int r = CatsEye_accuracy(this, x+this->layer[0].inputs*N, (int16_t*)t+N, verify);
			printf(" %.1f%%", (float)r/verify*100.0);
		}
		printf("\n");
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
			printf("\nnan or inf error!\n");
			CatsEye_layer *l = &this->layer[this->start];
			for (int i=this->start; i<=this->end; i++) {
				for (int n=0; n<this->ws[i]; n++) {
					if (isnan(l->w[n]) || isinf(l->w[n])) {
						printf(" layer[%d] weight[%d]:%f\n", i, n, l->w[n]);
						break;
					}
				}
				for (int n=0; n<l->outputs; n++) {
					if (isnan(l->z[n]) || isinf(l->z[n])) {
						printf(" layer[%d] z[%d]:%f\n", i, n, l->z[n]);
						break;
					}
				}
				for (int n=0; n<l->inputs; n++) {
					if (isnan(l->dIn[n]) || isinf(l->dIn[n])) {
						printf(" layer[%d] d[%d]\n", i, n);
						break;
					}
				}
				l++;
			}
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
			for (i=0; i<this->ws[i]; i++) {
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
	for (int c=0; c<ch; c++) {
		for (int i=0; i<n; i++) {
			p[((i/sx)*width + i%sx)*ch +c] = (uint8_t)((o[i+c*n] - min) / (max - min) * 255.0);
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
	for (int c=0; c<ch; c++) {
		for (int i=0; i<n; i++) {
			p[((i/sx)*width + i%sx)*ch +c] = (uint8_t)((o[i+c*n] - min) / (max - min) * 255.0);
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

#undef SOLVER

//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#define _debug(...)	{ printf("%s(%d):", __func__, __LINE__); printf(__VA_ARGS__); }

#ifdef CATS_USE_FIXED
#define real		short
#elif defined CATS_USE_FLOAT
#define real		float
#else
#define real		double
#warning "using double!!"
#endif

#define CATS_SIGMOID
//#define CATS_SIGMOID_CROSSENTROPY
#ifdef CATS_SIGMOID_CROSSENTROPY
// cross entropy df: (y - t) / (y * (1 - y))
// sigmoid function with cross entropy loss
#define CATS_SIGMOID
#define s_gain			1
#define ACT2(x)			ACT1(x)
#define DACT2(x)		DACT1(x)
// SoftmaxWithLoss
#else
// identify function with mse loss
#define s_gain			1
#define ACT2(x)			(x)
#define DACT2(x)		1
#endif

#ifdef CATS_OPT_ADAGRAD
// AdaGrad (http://qiita.com/ak11/items/7f63a1198c345a138150)
#define eps 1e-8		// 1e-4 - 1e-8
#define OPT_CALC1(x)		this->e##x[i] += this->d[x-2][i]*this->d[x-2][i]
//#define OPT_CALC1(x)		this->e##x[i] += this->d[x-2][i]*this->d[x-2][i] *0.7
//#define OPT_CALC1(x)		this->e##x[i] = this->e##x[i]*(0.99+times/times*0.01) + this->d[x-2][i]*this->d[x-2][i]
//#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->d[y-2][j] /sqrt(this->e##y[j])
#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->d[y-2][j] /sqrt(this->e##y[j]+eps)

#elif defined CATS_OPT_ADAM
// Adam
#define eps 1e-8
#define beta1 0.9
#define beta2 0.999
#define OPT_CALC1(x)		this->m##x[i] = beta1*this->m##x[i] + (1.0-beta1) * this->d[x-2][i]; \
				this->v##x[i] = beta2*this->v##x[i] + (1.0-beta2) * this->d[x-2][i]*this->d[x-2][i]
#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->m##y[j] /sqrt(this->v##y[j]+eps)

#elif defined CATS_OPT_RMSPROP
// RMSprop (http://cs231n.github.io/neural-networks-3/#anneal)
#define eps 1e-8		// 1e-4 - 1e-8
#define decay_rate 0.999	// [0.9, 0.99, 0.999]
#define OPT_CALC1(x)		this->e##x[i] = decay_rate * this->e##x[i] + (1.0-decay_rate)*this->d[x-2][i]*this->d[x-2][i]
#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->d[y-2][j] /sqrt(this->e##y[j]+eps)

#elif defined CATS_OPT_RMSPROPGRAVES
// RMSpropGraves (https://github.com/pfnet/chainer/blob/master/chainer/optimizers/rmsprop_graves.py)
#define eps 1e-4
#define beta1 0.95
#define beta2 0.95
#define momentum 0.9
#define OPT_CALC1(x)		this->m##x[i] = beta1*this->m##x[i] + (1.0-beta1) * this->d[x-2][i]; \
				this->v##x[i] = beta2*this->v##x[i] + (1.0-beta2) * this->d[x-2][i]*this->d[x-2][i]
#define OPT_CALC2(n, x, y)	this->dl##x[j] = this->dl##x[j] * momentum - eta*this->o[x-1][i]*this->d[y-2][j] /sqrt(this->v##y[j] - this->m##y[j]*this->m##y[j]+eps); \
				this->w[x-1][i*n+j] += this->dl##x[j]

#elif defined CATS_OPT_MOMENTUM
// Momentum update (http://cs231n.github.io/neural-networks-3/#anneal)
#define momentum 0.9		// [0.5, 0.9, 0.95, 0.99]
#define OPT_CALC1(x)
#define OPT_CALC2(n, x, y)	this->dl##x[i] = momentum * this->dl##x[i] - eta*this->o[x-1][i] *this->d[y-2][j]; \
				this->w[x-1][i*n+j] += this->dl##x[i]

#else
// SGD (Vanilla update)
#define CATS_OPT_SGD
#define OPT_CALC1(x)
//#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->d[y-2][j]
// SVM (http://d.hatena.ne.jp/echizen_tm/20110627/1309188711)
// ∂loss(w, x, t) / ∂w = ∂(λ - twx + α * w^2 / 2) / ∂w = - tx + αw
#define OPT_CALC2(n, x, y)	this->w[x-1][i*n+j] -= eta*this->o[x-1][i] *this->d[y-2][j] +this->w[x-1][i*n+j]*1e-8
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
uint64_t xor128()
{
	uint64_t s1 = seed[0];
	const uint64_t s0 = seed[1];
	seed[0] = s0;
	s1 ^= s1 << 23;
	return ( seed[1] = ( s1 ^ s0 ^ ( s1 >> 17 ) ^ ( s0 >> 26 ) ) ) + s0;
}
#define frand()		( xor128() / ((double)XOR128_MAX + 1.0f) )
int binomial(/*int n, */real p)
{
//	if (p<0 || p>1) return 0;
	int c = 0;
//	for (int i=0; i<n; i++) {
		real r = frand();
		if (r < p) c++;
//	}
	return c;
}

// identity function (output only)
real CatsEye_act_identity(real x)
{
	return (x);
}
real CatsEye_dact_identity(real x)
{
	return (1.0);
}
// softmax function (output only)
real CatsEye_act_softmax(real x/* *x, int n, int len*/)
{
/*	real alpha = x[0];
	for (int i=1; i<len; i++) if (alpha<x[i]) alpha = x[i];
	real numer = exp(x[n] - alpha);
	real denom = 0.0;
	for (int i=0; i<len; i++) denom += exp(x[i] - alpha);
	return (numer / denom);*/
	return (x);
}
real CatsEye_dact_softmax(real x)
{
	return (x * (1.0 - x));
}
// sigmoid function
real CatsEye_act_sigmoid(real x)
{
	return (1.0 / (1.0 + exp(-x * s_gain)));
}
real CatsEye_dact_sigmoid(real x)
{
	return ((1.0-x)*x * s_gain);	// ((1.0-sigmod(x))*sigmod(x))
}
// tanh function
// https://github.com/nyanp/tiny-cnn/blob/master/tiny_cnn/activations/activation_function.h
real CatsEye_act_tanh(real x)
{
	return (tanh(x));

/*	real ep = exp(x);
	real em = exp(-x);
	return (ep-em) / (ep+em);*/

	// error at paint.c
	// fast approximation of tanh (improve 2-3% speed in LeNet-5)
/*	real x1 = x;
	real x2 = x1 * x1;
	x1 *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
	return x1 / sqrt(1.0 + x2);*/
}
real CatsEye_dact_tanh(real x)
{
	return (1.0-x*x);		// (1.0-tanh(x)*tanh(x))
}
// scaled tanh function
real CatsEye_act_scaled_tanh(real x)
{
	return (1.7159 * tanh(2.0/3.0 * x));
}
real CatsEye_dact_scaled_tanh(real x)
{
	return ((2.0/3.0)/1.7159 * (1.7159-x)*(1.7159+x));
}
// rectified linear unit function
real CatsEye_act_ReLU(real x)
{
	return (x>0 ? x : 0.0);
}
real CatsEye_dact_ReLU(real x)
{
	return (x>0 ? 1.0 : 0.0);
}
// leaky rectified linear unit function
#define leaky_alpha	0.01	// 0 - 1
real CatsEye_act_LeakyReLU(real x)
{
	return (x>0 ? x : x*leaky_alpha);
}
real CatsEye_dact_LeakyReLU(real x)
{
	return (x>0 ? 1.0 : leaky_alpha);
}
// exponential rectified linear unit function
// http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/elu.html
real CatsEye_act_ELU(real x)
{
	return (x>0 ? x : exp(x)-1.0);
}
real CatsEye_dact_ELU(real x)
{
	return (x>0 ? 1.0 : 1.0+x);
}
// abs function
real CatsEye_act_abs(real x)
{
	return (x / (1.0 + fabs(x)));
}
real CatsEye_dact_abs(real x)
{
	return (1.0 / (1.0 + fabs(x))*(1.0 + fabs(x)));
}

// activation function and derivative of activation function
real (*CatsEye_act[])(real x) = {
	CatsEye_act_identity,
	CatsEye_act_softmax,
	CatsEye_act_sigmoid,
	CatsEye_act_tanh,
	CatsEye_act_scaled_tanh,
	CatsEye_act_ReLU,
	CatsEye_act_LeakyReLU,
	CatsEye_act_ELU,
	CatsEye_act_abs
};
real (*CatsEye_dact[])(real x) = {
	CatsEye_dact_identity,
	CatsEye_dact_softmax,
	CatsEye_dact_sigmoid,
	CatsEye_dact_tanh,
	CatsEye_dact_scaled_tanh,
	CatsEye_dact_ReLU,
	CatsEye_dact_LeakyReLU,
	CatsEye_dact_ELU,
	CatsEye_dact_abs
};
typedef enum {
	CATS_ACT_IDENTITY, CATS_ACT_SOFTMAX, CATS_ACT_SIGMOID, CATS_ACT_TANH, CATS_ACT_SCALED_TANH,
	CATS_ACT_RELU, CATS_ACT_LEAKY_RELU, CATS_ACT_ELU, CATS_ACT_ABS
} CATS_ACTIVATION;
typedef real (*CATS_ACT)(real x);

void CatsEye_act_array(CATS_ACT act, real *z, real *x, int num)
{
	#pragma omp parallel for
	for (int n=num; n>0; n--) *z++ = act(*x++);
}
void CatsEye_dact_array(CATS_ACT dact, real *z, real *x, int num)
{
	#pragma omp parallel for
	for (int n=num; n>0; n--) *z++ = dact(*x++);
}

enum CATS_LP {
	TYPE,		// MLP, CONV, MAXPOOL
	ACT,		// activation function type
	CHANNEL,
	SIZE,		// input size (ch * x * y)
	XSIZE,		// width
	YSIZE,		// height
	KSIZE,		// kernel size
	STRIDE,
	LPLEN		// length of layer params
};
#define TYPE(i)		this->u[LPLEN*(i)+TYPE]
#define SIZE(i)		this->u[LPLEN*(i)+SIZE]
//#define CH(i)		this->u[LPLEN*(i)+CH]
#define RANDOM		this->u[STRIDE]

#ifdef CATS_SSE
#include "catseye_simd.h"	// deprecated!
#else
// vec1 * vec2
real dotvv(real *vec1, real *vec2, int n)
{
	real s = 0;
	int i;
	#pragma omp simd reduction(+:s)
	for (i=n; i>=8; i-=8) {
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
		s += (*vec1++) * (*vec2++);
	}
	for (; i>=0; i--) {
		s += (*vec1++) * (*vec2++);
	}
	return s;
}
// mat * vec
void dotmv(real *s, real *mat, real *vec, int n, int m)
{
	#pragma omp parallel for simd
	for (int i=m; i>0; i--) {
		*s++ = dotvv(mat, vec, n);
		mat += n;
	}
	return;
}
// vec += mat * vec
void dotamv(real *s, real *mat, real *vec, int n, int m)
{
	#pragma omp parallel for simd
	for (int i=m; i>0; i--) {
		*s++ += dotvv(mat, vec, n);
		mat += n;
	}
	return;
}
void muldot(real *s, real *mat, real *vec, int n, int m)
{
	#pragma omp parallel for simd
	for (int i=m; i>0; i--) {
		*s++ *= dotvv(mat, vec, n);
		mat += n;
	}
	return;
}
real dotTv(real *mat1, real *vec1, int r, int c)
{
	real s = 0;
	#pragma omp simd reduction(+:s)
	for (int i=r; i>0; i--) {
		s += *mat1 * (*vec1++);
		mat1 += c;
	}
	return s;
}
void muladd(real *vec1, real *vec2, real a, int n)
{
	#pragma omp for simd
	for (int i=n; i>0; i--) {
		*vec1++ += a * (*vec2++);
	}
}
void outeradd(real *s, real *vec1, real *vec2, int n, int m)
{
	#pragma omp for simd
	for (int i=m; i>0; i--) {
		muladd(s, vec2, *vec1++, n);
	}
}
void muladdx(real *vec1, real *vec2, real a, int n)
{
	#pragma omp for simd
	for (int i=n; i>0; i--) {
		*vec1 += a * (*vec2++) + (*vec1) * 1e-8;
		vec1++;
	}
}
#endif

typedef struct layer {
	int inputs;		// input size

	int type;
	int activation;

	real eta;

	int ksize;		// CNN
	int stride;		// CNN
	int padding;		// CNN
	int px, py, pz;	// CNN (AUTO: padding)
	int ich, ch;		// CNN
	int sx, sy, ox, oy;	// CNN (AUTO: i/o size)

	int hiddens;		// RNN
	int truncatedTime;	// RNN

	// auto config
	int outputs;		// output size
	real *x;		// input
	real *z;		// output
	real *W, *dW, *prev_dw;
	real *Wi, *dWi;		// RNN [hidden * time](input -> hidden) = W, dW
	real *Wr, *dWr;		// RNN [hidden * hidden]
	real *Wo, *dWo;		// RNN [output * hidden](hidden -> output)
//	real *U, *dU;		// RNN [hidden * time](input -> hidden)
//	real *V, *dV;		// RNN [output * hidden](hidden -> output)
	real *eh;		// RNN [time * hidden]
	real *s;		// RNN [time * hidden]
	real *u;		// RNN [time * hidden]
//	real *y;		// RNN [time * output]
	real *v;		// RNN [time * output]
	real (*act2)(real x);	// RNN
	real (*dact2)(real x);	// RNN

	real (*act)(real x);
	real (*dact)(real x);
//	void (*act)(real *z, real *x, int n);
//	void (*dact)(real *z, real *x, int n);
	void (*forward)(struct layer*);
	void (*backward)(struct layer*);
	void (*update)(struct layer*);
	void (*loss)(struct layer *l, void *t, int n);
} CatsEye_layer;

// FNN
void _CatsEye_linear_layer_forward(CatsEye_layer *l)
{
	real *o = l->z;
	real *w = l->W;
	for (int i=l->outputs; i>0; i--) {
		*o++ = l->act(dotTv(w++, l->x, l->inputs+1, l->outputs));	// bias!!
	}
}
void _CatsEye_linear_layer_backward(CatsEye_layer *l)
{
	real *o = l->x;
	real *d = l->prev_dw;
	for (int i=0; i<=l->inputs; i++) {	// bias!!
		*d++ = dotvv(&l->W[i*l->outputs], l->dW, l->outputs) * l->dact(*o++);
	}
}
void _CatsEye_linear_layer_update(CatsEye_layer *l)
{
	real *o = l->x;
	real *w = l->W;
	for (int i=0; i<=l->inputs; i++) {	// bias!!
		real a = -l->eta * (*o++);
		muladd(w, l->dW, a, l->outputs);
		w += l->outputs;
	}
}
// RNN
void CatsEye_rnn_layer_forward(CatsEye_layer *l)
{
	real *u = l->u;
	real *s = l->s;
	real *v = l->v;
	real *y = l->z;

	// t=0
	dotmv(u, l->Wi, l->x, l->inputs, l->hiddens);	// l->inputs+1
	CatsEye_act_array(l->act, s, u, l->hiddens);
	dotmv(v, l->Wo, s, l->hiddens, l->outputs);
	CatsEye_act_array(l->act2, y, v, l->outputs);

	for (int t=1; t<l->inputs; t++) {
		u += l->hiddens;
		v += l->outputs;
		y += l->outputs;

		dotmv(u, l->Wi, l->x, l->inputs, l->hiddens);	// l->inputs+1
		dotamv(u, l->Wr, s, l->inputs, l->hiddens);	// s[t-1]	// l->inputs+1
		s += l->hiddens;
		CatsEye_act_array(l->act, s, u, l->hiddens);
		dotmv(v, l->Wo, s, l->hiddens, l->outputs);
		CatsEye_act_array(l->act2, y, v, l->outputs);
	}
}
void CatsEye_rnn_layer_backward(CatsEye_layer *l)
{
	real *d = l->dW + (l->inputs-1)*l->outputs;
	real *s = l->s + (l->inputs-1)*l->hiddens;
	real *u = l->u + (l->inputs-1)*l->hiddens;
	real *eh = l->eh + (l->inputs-1)*l->hiddens;

	for (int t=l->inputs-1; t>=0; t--) {
		//muladd(dV, s, *d, l->hiddens);		// dV[] += d * s[]
		outeradd(l->dWo, d, s, l->hiddens, l->outputs);
		CatsEye_dact_array(l->dact, eh, u, l->hiddens);
		//dotmv(eh, d, l->V, l->outputs, l->hiddens);
		muldot(eh, d, l->Wo, l->outputs, l->hiddens);

		s -= l->hiddens;
		u -= l->hiddens;
		real *_s = s;
		real *_u = u;
		real *_eh = eh;
		for (int z=0; z<l->truncatedTime; z++) {
			if (t-z < 0) break;
			outeradd(l->dWi, _eh, l->x+(t-z)*l->inputs, l->inputs, l->hiddens);	// eh[t-z]

			if (t-z-1 >= 0) {
				outeradd(l->dWr, _eh, _s, l->outputs, l->hiddens);		// s[t-z-1]
				CatsEye_dact_array(l->dact, _eh-l->hiddens, _u, l->hiddens);	// u[t-z-1]
				muldot(_eh-l->hiddens, _eh, l->Wr, l->outputs, l->hiddens);
			}
			_s -= l->hiddens;
			_u -= l->hiddens;
			_eh -= l->hiddens;
		}

		d -= l->outputs;
		eh -= l->hiddens;
	}
}
void CatsEye_rnn_layer_update(CatsEye_layer *l)
{
	muladd(l->Wi, l->dWi, -l->eta, l->hiddens * l->inputs);
	muladd(l->Wo, l->dWo, -l->eta, l->outputs * l->hiddens);
	muladd(l->Wr, l->dWr, -l->eta, l->outputs * l->hiddens);
}

// calculate forward propagation
void _CatsEye_convolutional_layer_forward(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;

	// c[out], c[in], ksize, h, w
	real *s, *z;
	real *w = l->W;
	real *o = l->z;	// need!
	memset(o, 0, sizeof(real)*l->ch*l->ox*l->oy);
	for (int c=l->ch; c>0; c--) {	// out
		real *r = s = l->x;
//		o = l->z + (l->ch-c)*l->ox*l->oy;
		o = l->z + l->pz + (l->ch-c)*l->ox*l->oy;
		for (int cc=l->ich; cc>0; cc--) {	// in
//			s = l->x + (l->ich-cc)*l->sx*l->sy;
			for (int wy=l->ksize; wy>0; wy--) {
				for (int wx=l->ksize; wx>0; wx--) {
					real *p = s++;	// in
					z = o;		// out
					for (int y=l->py; y>0; y--) {
						muladd(z, p, *w, l->px);	// *z++ += (*p++) * (*w); p += m;
						p += l->sx;
						z += l->ox;
					}
					w++;
				}
				s += step;
			}
			r += l->sx * l->sy;
			s = r;
		}
//		o = z + l->pz;
	}
	o = l->z;
	for (int c=l->ch*l->ox*l->oy; c>0; c--) {	// out
		*o = l->act(*o);
		o++;
	}
//	CatsEye_act_array(l->act, l->z, l->z, l->ch*ox*oy);
}
// calculate back propagation
void _CatsEye_convolutional_layer_backward(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;

	// calculate the error
	memset(l->prev_dw, 0, sizeof(real)*l->ich*l->sx*l->sy);

	// c[out], c[in], ksize, h, w
	real *d, *prev_delta;
	real *w = l->W;
	real *delta = l->dW;
	for (int c=l->ch; c>0; c--) {	// out
		real *r = prev_delta = l->prev_dw;
		for (int cc=l->ich; cc>0; cc--) {	// in
			for (int wy=l->ksize; wy>0; wy--) {
				for (int wx=l->ksize; wx>0; wx--) {
					real *p = prev_delta++;	// in
					d = delta;		// out
					for (int y=l->oy; y>0; y--) {
						muladd(p, d, *w, l->ox);	// *p++ += (*d++) * (*w);
						p += l->sx;
						d += l->ox;
					}
					w++;
				}
				prev_delta += step;
			}
			r += l->sx * l->sy;
			prev_delta = r;
		}
		delta = d;
//		prev_delta = l->prev_dw;
	}

	real *prev_out = l->x;
	for (int i=l->ich*l->sx*l->sy; i>0; i--) {
		*prev_delta++ *= l->dact(*prev_out++);
	}
}
// update the weights
void _CatsEye_convolutional_layer_update(CatsEye_layer *l)
{
	int step = l->sx - l->ksize;

	// c[out], c[in], ksize, h, w
	real *w = l->W;
	real *d, *prev_out;
	real *curr_delta = l->dW;
	for (int c=l->ch; c>0; c--) {	// out
		real *r = prev_out = l->x;
		for (int cc=l->ich; cc>0; cc--) {	// in
			for (int wy=l->ksize; wy>0; wy--) {
				for (int wx=l->ksize; wx>0; wx--) {
					real *p = prev_out++;	// in
					d = curr_delta;		// out
					real a = 0;
					for (int y=l->oy; y>0; y--) {
						a += dotvv(d, p, l->ox);	// a += d * p;
						p += l->sx;
						d += l->ox;
					}
					*w++ += -l->eta * a;
				}
				prev_out += step;
			}
			r += l->sx * l->sy;
			prev_out = r;
		}
		curr_delta = d;
//		prev_out = l->x;
	}
}

// calculate forward propagation
void _CatsEye_maxpooling_layer_forward(CatsEye_layer *l)
{
	int sx = l->sx;
	int sy = l->sy;
	int *max = (int*)l->W; // temp
	real *o = l->z;

	for (int c=0; c<l->ch; c++) {
		for (int y=0; y<sy-1; y+=l->stride) {
			for (int x=0; x<sx-1; x+=l->stride) {
				int n = c*sx*sy + y*sx+x;
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
					n += sx-l->ksize;
				}
				max++;
				*o++ = a;
			}
		}
	}
}
// calculate back propagation
void _CatsEye_maxpooling_layer_backward(CatsEye_layer *l)
{
	int *max = (int*)l->W; // temp
	real *delta = l->dW;
	real *d = l->prev_dw;
	memset(d, 0, sizeof(real)*l->inputs);
	for (int i=0; i<l->outputs; i++) {
		d[*max] = (*delta++) * l->dact(l->x[*max]);
		max++;
	}
}

void CatsEye_none(CatsEye_layer *l)
{
}
void (*_CatsEye_layer_forward[])(CatsEye_layer *l) = {
	_CatsEye_linear_layer_forward,
	_CatsEye_convolutional_layer_forward,
	_CatsEye_maxpooling_layer_forward,
	CatsEye_rnn_layer_forward,
};
void (*_CatsEye_layer_backward[])(CatsEye_layer *l) = {
	_CatsEye_linear_layer_backward,
	_CatsEye_convolutional_layer_backward,
	_CatsEye_maxpooling_layer_backward,
	CatsEye_rnn_layer_backward,
};
void (*_CatsEye_layer_update[])(CatsEye_layer *l) = {
	_CatsEye_linear_layer_update,
	_CatsEye_convolutional_layer_update,
	CatsEye_none,
	CatsEye_rnn_layer_update,
};
typedef enum {
	CATS_LINEAR, CATS_CONV, CATS_MAXPOOL, CATS_RECURRENT, CATS_LOSS
} CATS_LAYER_TYPE;

// calculate the error of output layer
void _CatsEye_loss_0_1(CatsEye_layer *l, void *t, int n)
{
	real *d = l->prev_dw;
	real *o = l->x;

	int a = ((int16_t*)t)[n];
	for (int i=0; i<l->inputs; i++) {
		// 0-1 loss function
		*d++ = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
	// http://d.hatena.ne.jp/echizen_tm/20110606/1307378609
	// E = max(0, -twx), ∂E / ∂w = max(0, -tx)
}
// loss function for mse with identity and cross entropy with sigmoid
void _CatsEye_loss_mse(CatsEye_layer *l, void *t, int n)
{
	real *d = l->prev_dw;
	real *o = l->x;

	real *a = &((real*)t)[n * l->inputs];
	for (int i=0; i<l->inputs; i++) {
		// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
		// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
		*d++ = *o++ - *a++;
	}
}
void (*_CatsEye_loss[])(CatsEye_layer *l, void *t, int n) = {
	_CatsEye_loss_0_1,
	_CatsEye_loss_mse,
};
typedef enum {
	CATS_LOSS_0_1, CATS_LOSS_MSE, //CATS_LOSS_MSESPARSE
} CATS_LOSS_TYPE;

typedef struct {
	// number of each layer
	int layers, *u;
	CatsEye_layer *layer;

	// input layers
	real *xdata;
	int xsize;
	// output layers [o = f(z)]
	real **z, **o, *odata;
	int osize;
	// error value
	real **d, *ddata;
	int dsize;
	// weights
	real **w, *wdata;
	int *ws, wsize;

	// gradient value
//	real *e2, *e3, *m2, *v2, *m3, *v3;
//	real *dl1, *dl2;

	// deprecated!
	int in;
	real *o3;
} CatsEye;

#define _CatsEye__construct(t, p)	__CatsEye__construct(t, p, sizeof(p)/sizeof(CatsEye_layer))
void __CatsEye__construct(CatsEye *this, CatsEye_layer *layer, int layers)
{
	this->u = 0;
	this->z = 0;
	this->layers = layers;
	this->layer = calloc(this->layers, sizeof(CatsEye_layer));
	memcpy(this->layer, layer, this->layers*sizeof(CatsEye_layer));

	// calculate parameters
	int osize[this->layers], dsize[this->layers], wsize[this->layers];
	this->o = malloc(sizeof(real*)*(this->layers));	// outputs
	this->osize = 0;
	this->d = malloc(sizeof(real*)*(this->layers/*-1*/));	// errors
	this->dsize = 0;
	this->w = malloc(sizeof(real*)*(this->layers/*-1*/));	// weights
	this->ws = malloc(sizeof(int)*(this->layers/*-1*/));
	this->wsize = 0;

	int n[this->layers], m[this->layers];
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];

		if (i<this->layers-1) {
			l->forward = _CatsEye_layer_forward[l->type];
			l->backward = _CatsEye_layer_backward[l->type];
			if (!i && l->type!=CATS_RECURRENT) l->backward = CatsEye_none;
			l->update = _CatsEye_layer_update[l->type];
		} else { // last layer
			if (!l->outputs) l->outputs = 1;
			l->forward = CatsEye_none;
			l->backward = CatsEye_none;
			l->update = CatsEye_none;
			l->loss = _CatsEye_loss[l->activation];
		}
		// activation
		l->act = CatsEye_act[l->activation];
		if (i>0) l->dact = CatsEye_dact[(l-1)->activation];
		else l->dact = CatsEye_dact[l->activation];

		osize[i] = 0;
		if (i>0) {
			l->ich = (l-1)->ch;
			if (!l->inputs) l->inputs = (l-1)->outputs;
			else if (l->inputs<0) { // select the layer
				l->inputs = (l+l->inputs)->outputs;
				osize[i] = osize[i+l->inputs];
			}
//			if (l->inputs<=0) l->inputs = (l-1+l->inputs)->outputs;
		} else { // first layer
			if (!l->ch) l->ch = 1;
			if (!l->ich) l->ich = 1;
		}
		if (!l->sx && l->ich) l->sx = l->sy = sqrt(l->inputs/l->ich);

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
			printf("L%02d: CONV%d-%d i/o:%d/%d[%dx%d/%dx%d] (%d[ksize^2]x%d[ch])\n", i+1, l->ksize, l->ch, l->inputs, l->outputs, l->sx, l->sy, l->ox, l->oy, n[i], m[i]);
			break;
		case CATS_MAXPOOL:
			l->ch = l->ich;
			l->ox = (l->sx +2*l->padding -l->ksize) /l->stride +1;
			l->oy = (l->sy +2*l->padding -l->ksize) /l->stride +1;
			l->outputs = l->ch * l->ox * l->oy;
			n[i] = l->ch * l->sx * l->sy;
			m[i] = 1;
			printf("L%02d: POOL%d-%d (w:%d h:%d [%d])\n", i+1, l->ksize, l->ch, l->ox, l->oy, n[i]);
			break;
		case CATS_RECURRENT:
			l->Wi = calloc(l->inputs * l->hiddens, sizeof(real));
//			l->Wi = l->W;
			l->Wr = calloc(l->hiddens * l->hiddens, sizeof(real));
			l->Wo = calloc(l->hiddens * l->outputs, sizeof(real));
			l->dWi = calloc(l->inputs * l->hiddens, sizeof(real));
			l->dWr = calloc(l->hiddens * l->hiddens, sizeof(real));
			l->dWo = calloc(l->hiddens * l->outputs, sizeof(real));
			l->eh = calloc(l->inputs * l->hiddens, sizeof(real));
			l->s = calloc(l->inputs * l->hiddens, sizeof(real));
			l->u = calloc(l->inputs * l->hiddens, sizeof(real));
			l->v = calloc(l->inputs * l->outputs, sizeof(real));
//			l->y = calloc(l->inputs * l->outputs, sizeof(real));
//			l->y = l->z;
			//l->act2 = CatsEye_act[CATS_ACT_IDENTITY];		//FIXME
			//l->dact2 = CatsEye_dact[CATS_ACT_IDENTITY];	//FIXME
			l->act2 = CatsEye_act[CATS_ACT_SIGMOID];
			l->dact2 = CatsEye_dact[CATS_ACT_SIGMOID];
//			n[i] = l->hiddens;
			n[i] = l->inputs;
			m[i] = l->hiddens;
			printf("L%02d: RECURRENT %d %d\n", i+1, n[i], m[i]);
			break;
		default: // LINEAR
//			if (i>0 && !l->inputs) l->inputs = this->layer[i-1].outputs;
			if (i<this->layers-1) l->outputs = (l+1)->inputs;
			n[i] = l->inputs;
			m[i] = l->outputs;
			printf("L%02d: LINEAR %d %d\n", i+1, n[i], m[i]);
		}
		wsize[i] = this->wsize;
		this->ws[i] = (n[i]+1)*m[i];
		this->wsize += this->ws[i];

		if (!osize[i]) {
			osize[i] = this->osize;
			this->osize += l->inputs+1;	// bias
		}

		dsize[i] = this->dsize;
		this->dsize += l->outputs+1;	// bias
	}
//	this->osize--;	// bias
	this->odata = calloc(this->osize, sizeof(real));
//	this->dsize--;
	this->ddata = calloc(this->dsize, sizeof(real));
	this->wdata = calloc(this->wsize, sizeof(real));
	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];

		l->x = this->odata + osize[i];
		if (i<this->layers-1) l->z = this->odata + osize[i+1];	// output
//		else l->z = this->odata + osize[i];
		l->dW = this->ddata + dsize[i];
		l->W = this->wdata + wsize[i];
		if (i>0) l->prev_dw = this->ddata + dsize[i-1];

		this->o[i] = this->odata + osize[i];	// input
		this->d[i] = this->ddata + dsize[i];
		this->w[i] = this->wdata + wsize[i];

		if (i==this->layers-1) break;
		this->o[i][l->outputs] = 1;	// bias

		// initialize weights (http://aidiary.hatenablog.com/entry/20150618/1434628272)
		// range depends on the research of Y. Bengio et al. (2010)
		xor128_init(time(0));
		real range = sqrt(6)/sqrt(n[i]+m[i]+2);
		for (int j=0; j<this->ws[i]; j++) {
			this->w[i][j] = 2.0*range*frand()-range;
		}
//		memcpy(&this->wdata[this->wsize], this->wdata, this->wsize*sizeof(real));	// for debug
	}

	for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		printf("L%02d in:%d out:%d (x:%ld-z:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, l->x-this->odata, l->z-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
	}
#ifdef CATS_OPENCL
	CatsEye_clSetup(this);
#endif
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
#ifdef CATS_OPENCL
	CatsEye_clFinish();
#endif
	// delete arrays
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

void _CatsEye_forward(CatsEye *this, real *x)
{
//	this->layer[0].x = x;	//FIXME
	memcpy(this->o[0], x, this->layer[0].inputs*sizeof(real));
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<this->layer[0].inputs; i++) {
		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
	}
#endif
	for (int i=0; i<this->layers-1; i++) {
		this->o[i][this->layer[i].inputs] = 1;	// for bias
		this->layer[i].forward(&this->layer[i]);
	}
}

void _CatsEye_train(CatsEye *this, real *x, void *t, int N, int repeat, int random)
{
	int a = this->layers-1;
	this->layer[a].z = t;	//FIXME

	this->xdata = x;
	this->xsize = N;

	int batch = N;			// for random
	if (random) batch = random;

	struct timeval start, stop;
	gettimeofday(&start, NULL);
	for (int times=0; times<repeat; times++) {
		for (int n=0; n<batch; n++) {
			int sample = random ? (frand()*N) : n;

			// forward propagation
			_CatsEye_forward(this, x+sample*this->layer[0].inputs);

			// calculate the error of output layer
			this->layer[a].loss(&this->layer[a], t, sample);
			// calculate the error of hidden layer
			for (int i=this->layers-2; i>=0; i--) { // i=0 -> RNN
				this->layer[i].backward(&this->layer[i]);
			}

			// update the weights
			for (int i=this->layers-2; i>=0; i--) {
				this->layer[i].update(&this->layer[i]);
			}
#ifdef CATS_AUTOENCODER
			// tied weight
			/*real *dst = this->w[1];
			for (int i=0; i<SIZE(1); i++) {
				for (int j=0; j<SIZE(0); j++) {
					this->w[1][j + SIZE(1)*i] = this->w[0][SIZE(1)*j + i];
				}
			}*/
#endif
		}
		real err = 0;
		{
			// calculate the mean squared error
			real mse = 0;
			for (int i=0; i<this->layer[a-1].outputs; i++) {
				mse += 0.5 * (this->d[a-1][i] * this->d[a-1][i]);
			}
			err = 0.5 * (err + mse);
		}
		printf("epochs %d, mse %f", times, err);
		gettimeofday(&stop, NULL);
		printf(" [%.2fs]", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);
		printf("\n");
	}
}
// return most probable label to the input x
int _CatsEye_predict(CatsEye *this, real *x)
{
	// forward propagation
	_CatsEye_forward(this, x);

	// biggest output means most probable label
	CatsEye_layer *l = &this->layer[this->layers-1];
//	int a = this->layers-1;
//	real max = this->o[a][0];
	real max = l->x[0];
	int ans = 0;
/*	for (int i=1; i<this->layer[a].inputs; i++) {
		if (this->o[a][i] > max) {
			max = this->o[a][i];
		}
	}*/
	for (int i=1; i<l->inputs; i++) {
		if (l->x[i] > max) {
			max = l->x[i];
			ans = i;
		}
	}
	return ans;
}

// https://www.cs.toronto.edu/~kriz/cifar.html
real *_CatsEye_loadCifar(char *name, int size, int lsize, int sample, int16_t **label)
{
	unsigned char *data = malloc((size+lsize)*sample);	// +1 for label
	if (!data) return 0;
	int16_t *t = malloc(sizeof(int16_t)*sample);
	if (!t) return 0;
//	real *x = malloc(sizeof(real)*(size+1)*(sample+1));	// +1 for bias
	real *x = malloc(sizeof(real)*size*sample);
	if (!x) return 0;

	FILE *fp = fopen(name, "rb");
	if (!fp) return 0;
	fread(data, (size+lsize)*sample, 1, fp);
	for (int n=0; n<sample; n++) {
		t[n] = data[n*(size+lsize)];
		for (int i=0; i<size; i++) x[n*size+i] = data[n*(size+lsize)+lsize+i] * (1.0/255.0);
	}
	fclose(fp);
	free(data);

	*label = t;
	return x;
}




#define CATS_MBATCH	8
//#define CATS_OPENCL
#ifdef CATS_OPENCL
//#define CL_DEBUG
#include "catseye_cl.h"
#endif
/* constructor
 * n_in:  number of input layer
 * n_hid: number of hidden layer
 * n_out: number of output layer */
void CatsEye__construct(CatsEye *this, int n_in, int n_hid, int n_out, void *param)
{
	FILE *fp;
	if (!n_out && param) {
		// load
		fp = fopen(param, "r");
		if (fp==NULL) return;
		fscanf(fp, "%d %d %d\n", &SIZE(0), &SIZE(1), &SIZE(2));
	} else if (!n_in && n_out>0 && param) {
		// deep neural network
		this->layers = n_out;
		this->u = malloc(sizeof(int)*LPLEN*this->layers);
		memcpy(this->u, param, sizeof(int)*LPLEN*this->layers);
		param = 0;

		// calculate parameters
		for (int i=1; i<this->layers; i++) {
			int *u = &this->u[LPLEN*i];
			if (/*u[TYPE]!=CATS_LINEAR &&*/ !u[XSIZE]) {
				u[XSIZE] = u[YSIZE] = sqrt(u[SIZE-LPLEN]/u[CHANNEL-LPLEN]);
			}

			if (!u[SIZE]) {
				switch (u[TYPE]) {
				case CATS_CONV:
					u[SIZE] = u[CHANNEL] * (u[XSIZE]-u[KSIZE]/2*2) * (u[YSIZE]-u[KSIZE]/2*2);
					break;
				case CATS_MAXPOOL:
					u[CHANNEL] = u[CHANNEL-LPLEN];
					u[SIZE] = u[CHANNEL] * (u[XSIZE]/u[KSIZE]) * (u[YSIZE]/u[KSIZE]);
				}
			}
			printf("L%02d in:[%dx%dx%d] out:[%d]\n", i, u[CHANNEL-LPLEN], u[XSIZE], u[YSIZE], u[SIZE]);
		}
	} else {
		// multilayer perceptron
		int u[] = {
			0, 0, 1, n_in,    0, 0, 0, n_in>1500?1500:n_in,
			0, 2, 1, n_hid,   0, 0, 0, 0,
			0, 0, 1, n_out,   0, 0, 0, 0,
		};
		this->layers = sizeof(u)/sizeof(int)/LPLEN;
		this->u = malloc(sizeof(int)*LPLEN*this->layers);
		memcpy(this->u, u, sizeof(int)*LPLEN*this->layers);
	}
	this->in = SIZE(0);	// deprecated!

	// allocate inputs
	this->z = malloc(sizeof(real*)*(this->layers-1));
/*	for (int i=0; i<this->layers-1; i++) {
		this->z[i] = malloc(sizeof(real)*(SIZE(i+1)+1));
	}*/

	// allocate outputs
	int size[this->layers];
	this->o = malloc(sizeof(real*)*(this->layers));
	this->osize = 0;
	for (int i=0; i<this->layers; i++) {
		size[i] = this->osize;
		this->osize += SIZE(i)+1;	// bias
	}
	this->osize--;
	this->odata = malloc(sizeof(real)*this->osize *CATS_MBATCH);
	memset(this->odata, 0, sizeof(real)*this->osize *CATS_MBATCH);	// for debug
	for (int i=0; i<this->layers; i++) {
		this->o[i] = this->odata + size[i];
		this->o[i][SIZE(i)] = 1;	// bias
		for (int j=0; j<CATS_MBATCH; j++) this->odata[this->osize*j + size[i]+SIZE(i)] = 1;
	}
	this->o3 = this->o[2];		// deprecated!

	// allocate errors
	this->d = malloc(sizeof(real*)*(this->layers-1));
	this->dsize = 0;
	for (int i=0; i<this->layers-1; i++) {
		size[i] = this->dsize;
		this->dsize += SIZE(i+1)+1;	// bias
	}
	this->dsize--;
	this->ddata = malloc(sizeof(real)*this->dsize *CATS_MBATCH);
	for (int i=0; i<this->layers-1; i++) this->d[i] = this->ddata + size[i];

	// allocate gradient
/*	this->e2 = calloc(1, sizeof(real)*(SIZE(1)+1));
	this->e3 = calloc(1, sizeof(real)*SIZE(2));
	this->m2 = calloc(1, sizeof(real)*(SIZE(1)+1));
	this->m3 = calloc(1, sizeof(real)*SIZE(2));
	this->v2 = calloc(1, sizeof(real)*(SIZE(1)+1));
	this->v3 = calloc(1, sizeof(real)*SIZE(2));
	this->dl1 = malloc(sizeof(real)*(SIZE(0)+1)*SIZE(1));
	this->dl2 = malloc(sizeof(real)*(SIZE(1)+1)*SIZE(2));*/

	// allocate weights
	this->w = malloc(sizeof(real*)*(this->layers-1));
	this->ws = malloc(sizeof(int)*(this->layers-1));
	int n[this->layers-1], m[this->layers-1];
	this->wsize = 0;
	for (int i=0; i<this->layers-1; i++) {
		int *u = &this->u[LPLEN*(i+1)]; 
		switch (u[TYPE]) {
		case CATS_CONV:
			n[i] = u[KSIZE] * u[KSIZE];		// kernel size
			m[i] = u[CHANNEL] * u[CHANNEL-LPLEN];	// channel
			printf("L%02d: CONV%d-%d (%d[ksize]x%d[ch])\n", i+1, u[KSIZE], u[CHANNEL], n[i], m[i]);
			break;
		case CATS_MAXPOOL:
			n[i] = SIZE(i);
			m[i] = 1;
			printf("L%02d: POOL%d [%d]\n", i+1, u[KSIZE], n[i]);
			break;
		default:
			n[i] = SIZE(i);
			m[i] = SIZE(i+1);
			printf("L%02d: LINEAR %d %d\n", i+1, n[i], m[i]);
		}
		this->ws[i] = (n[i]+1)*m[i];
//		this->ws[i] = n[i]*m[i]+1;	// FIXME
		size[i] = this->wsize;
		this->wsize += this->ws[i];
	}
	this->wdata = malloc(sizeof(real)*this->wsize *CATS_MBATCH);
	for (int i=0; i<this->layers-1; i++) {
		this->w[i] = this->wdata + size[i];
//		this->w[i] = malloc(sizeof(real)*(n[i]+1)*m[i]);
//		if (!this->w[i]) printf("memory error at layer %d[%d], size %d!!\n", i+1, u[TYPE], this->ws[i]);

		// initialize weights (http://aidiary.hatenablog.com/entry/20150618/1434628272)
		// range depends on the research of Y. Bengio et al. (2010)
		xor128_init(time(0));
		real range = sqrt(6)/sqrt(n[i]+m[i]+2);
		for (int j=0; j<this->ws[i]; j++) {
			this->w[i][j] = 2.0*range*frand()-range;
//			for (int k=1; k<CATS_MBATCH; k++) this->wdata[this->wsize*k + size[i]+j] = 2.0*range*frand()-range;
		}
		memcpy(&this->wdata[this->wsize], this->wdata, this->wsize*sizeof(real));	// for debug
	}

	if (param) {
		for (int i=0; i<(SIZE(0)+1)*SIZE(1); i++) {
			fscanf(fp, "%f ", &this->w[0][i]);
		}
		for (int i=0; i<(SIZE(1)+1)*SIZE(2); i++) {
			fscanf(fp, "%f ", &this->w[1][i]);
		}
		fclose(fp);
	}
#ifdef CATS_OPENCL
	CatsEye_clSetup(this);
#endif
	/*for (int i=0; i<this->layers; i++) {
		CatsEye_layer *l = &this->layer[i];
		printf("L%02d in:%d out:%d (o:%ld-d:%ld-w:%ld)\n", i+1, l->inputs, l->outputs, this->o[i]-this->odata, this->d[i]-this->ddata, this->w[i]-this->wdata);
//		printf("L%02d in:[%dx%dx%d] out:[%d]\n", i, u[CHANNEL-LPLEN], u[XSIZE], u[YSIZE], u[SIZE]);
	}*/
}

// calculate forward propagation of input x
// f(x) = h(scale*x+bias)
void CatsEye_linear_layer_forward(CatsEye_layer *l, real *x, real *w, real *z/*no use*/, real *o, int u[])
{
	int in = u[SIZE-LPLEN]+1;	// +1 -> for bias
	int out = u[SIZE];

	CATS_ACT act = CatsEye_act[u[ACT]];
//	dotmv(o, w, x, in, out);
//	CatsEye_act_array(act, o, o, out);
	for (int i=out; i>0; i--) {
		*o++ = act(dotTv(w++, x, in, out));
//		*o++ = act(dotvv(w, x, in));
//		w += in;
	}
}
// calculate back propagation
void CatsEye_linear_layer_backward(CatsEye_layer *l, real *o, real *w, real *d, real *delta, int u[])
{
	int in = u[SIZE-LPLEN];
	int out = u[SIZE];

	// calculate the error
	CATS_ACT dact = CatsEye_dact[u[ACT-LPLEN]];
	//real s[in+1];
//	//outeradd(l->dV, d, s, in+1, out);
	//CatsEye_dact_array(dact, s, o, in+1);
//	//outeradd(l->dV, s, delta, in+1, out);
	//muldot(s, delta, w, out, in+1);
	for (int i=0; i<=in; i++) {	// bias!!
		*d++ = dotvv(&w[i*out], delta, out) * dact(*o++);
//		*d++ = dotTv(w++, delta, out, in+1) * dact(*o++);
	}
}
void CatsEye_linear_layer_update(CatsEye_layer *l, real eta, real *o, real *w, real *d, int u[])
{
	int in = u[SIZE-LPLEN]+1;	// +1 -> for bias
	int out = u[SIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		real a = -eta * (*o++);
		muladd(w, d, a, out);
		w += out;
		/*real a = eta * (*o++);
		for (int j=0; j<out; j++) {
			*w++ -= a*d[j];
		}*/
	}
}
void CatsEye_SVM_layer_update(CatsEye_layer *l, real eta, real *o, real *w, real *d, int u[])
{
	int in = u[SIZE-LPLEN]+1;
	int out = u[SIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		// SVM (http://d.hatena.ne.jp/echizen_tm/20110627/1309188711)
		// ∂loss(w, x, t) / ∂w = ∂(λ - twx + α * w^2 / 2) / ∂w = - tx + αw
		real a = -eta * (*o++);
		muladdx(w, d, a, out);	// *w -= a* (*dd++) + (*w)*1e-8; w++;
		w += out;
//		muladd(w++, d, a, out);	// *w -= a* (*dd++) + (*w)*1e-8; w++;
	}
}

// calculate forward propagation
void CatsEye_convolutional_layer_forward(CatsEye_layer *l, real *s, real *w, real *_z/*no use*/, real *o, int u[])
{
	int ks = u[KSIZE];
	int k2 = ks * ks;
	int n = u[CHANNEL] * k2;
	int m = (ks/2)*2;
	int sx = u[XSIZE] - m;	// out
	int sy = u[YSIZE] - m;
	int ch = u[CHANNEL-LPLEN];
	int step = u[XSIZE] - ks;
	CATS_ACT act = CatsEye_act[u[ACT]];

#ifdef CATS_FASTCONV
	// h, w, c[in], c[out]
	m *= ch;
	step *= ch;
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			real *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				real *p = s;	// in
				real a = 0;
				for (int wy=ks; wy>0; wy--) {
					for (int wx=ks; wx>0; wx--) {
						for (int cc=ch; cc>0; cc--) {	// in
							a += (*p++) * (*k++);
						}
					}
					p += step;
				}
				*o++ = act(a);
			}
			s += ch;
		}
		s += m;
	}
#else
	// c[out], k, c[in], h, w
	real *z, *p, *r;
	real *pp = s;
	memset(o, 0, sizeof(real)*u[CHANNEL]*sx*sy);
	for (int c=u[CHANNEL]; c>0; c--) {	// out
		r = s;
		for (int cc=ch; cc>0; cc--) {	// in
			for (int wy=ks; wy>0; wy--) {
				for (int wx=ks; wx>0; wx--) {
					p = s++;	// in
					z = o;		// out
					for (int y=sy; y>0; y--) {
						muladd(z, p, *w, sx);	// *z++ += (*p++) * (*w); p += m;
						p += u[XSIZE];
						z += sx;
					}
					w++;
				}
				s += step;
			}
			r += u[XSIZE] * u[YSIZE];
			s = r;
		}
		o = z;
		s = pp;
	}
	for (int c=u[CHANNEL]*sx*sy; c>0; c--) {	// out
		o--;
		*o = act(*o);
	}
#endif
}
// calculate back propagation
void CatsEye_convolutional_layer_backward(CatsEye_layer *l, real *prev_out, real *w, real *prev_delta, real *delta, int u[])
{
	int ix = u[XSIZE];	// in
	int iy = u[YSIZE];
	int ks = u[KSIZE];	// kernel size
	int m = (ks/2)*2;
	int sx = u[XSIZE] - m;	// out
	int sy = u[YSIZE] - m;
	int ch = u[CHANNEL-LPLEN];	// 'in' channel
	int step = u[XSIZE] - ks;

	// calculate the error
	memset(prev_delta, 0, sizeof(real)*ch*ix*iy);

#ifdef CATS_FASTCONV
	real *p = prev_delta;
	step *= ch;
	m *= ch;
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			real *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				real *d = prev_delta;	// in
				for (int wy=ks; wy>0; wy--) {
					for (int wx=ks; wx>0; wx--) {
						for (int cc=ch; cc>0; cc--) {	// in
							*d++ += (*delta) * (*k++);
						}
					}
					d += step;
				}
				delta++;
			}
			prev_delta += ch;
		}
		prev_delta += m;
	}
	prev_delta = p;
#else
	// c[out], k, c[in], h, w
	real *d, *p, *r;
	real *pp = prev_delta;
	for (int c=u[CHANNEL]; c>0; c--) {	// out
		r = prev_delta;
		for (int cc=ch; cc>0; cc--) {	// in
			for (int wy=ks; wy>0; wy--) {
				for (int wx=ks; wx>0; wx--) {
					p = prev_delta++;	// in
					d = delta;		// out
					for (int y=sy; y>0; y--) {
						muladd(p, d, *w, sx);	// *p++ += (*d++) * (*w);
						p += u[XSIZE];
						d += sx;
					}
					w++;
				}
				prev_delta += step;
			}
			r += u[XSIZE] * u[YSIZE];
			prev_delta = r;
		}
		delta = d;
		prev_delta = pp;
	}
#endif

	CATS_ACT dact = CatsEye_dact[u[ACT-LPLEN]];
	for (int i=ch*ix*iy; i>0; i--) {
		*prev_delta++ *= dact(*prev_out++);
	}
}
// update the weights
void CatsEye_convolutional_layer_update(CatsEye_layer *l, real eta, real *prev_out, real *w, real *curr_delta, int u[])
{
	int ks = u[KSIZE];
	int m = (ks/2)*2;
	int sx = u[XSIZE] - m;	// out
	int sy = u[YSIZE] - m;
	int ch = u[CHANNEL-LPLEN];
	int step = u[XSIZE] - ks;

#ifdef CATS_FASTCONV
	step *= ch;
	m *= ch;
	real *d = curr_delta;	// out
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			real *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				real *p = prev_out;	// in
				for (int wy=ks; wy>0; wy--) {
					for (int wx=ks; wx>0; wx--) {
						for (int cc=ch; cc>0; cc--) {	// in
							*k++ -= eta * (*d) * (*p++);
						}
					}
					p += step;
				}
				d++;
			}
			prev_out += ch;
		}
		prev_out += m;
	}
#else
	// c[out], k, c[in], h, w
	real *d, *p, *r;
	real *pp = prev_out;
	for (int c=u[CHANNEL]; c>0; c--) {	// out
		r = prev_out;
		for (int cc=ch; cc>0; cc--) {	// in
			for (int wy=ks; wy>0; wy--) {
				for (int wx=ks; wx>0; wx--) {
					p = prev_out++;	// in
					d = curr_delta;	// out
					real a = 0;
					for (int y=sy; y>0; y--) {
						a += dotvv(d, p, sx);
//						a -= dotvv(d, p, sx) * eta;
						p += u[XSIZE];
						d += sx;
/*						for (int x=sx; x>0; x--) {
//							*w -= eta * (*d++) * (*p++);
							a += (*d++) * (*p++);
						}
						p += m;*/
					}
//					w++;
					*w++ += -eta * a;
//					*w++ += a;
				}
				prev_out += step;
			}
			r += u[XSIZE] * u[YSIZE];
			prev_out = r;
		}
		curr_delta = d;
		prev_out = pp;
	}
#endif
}

// calculate forward propagation
void CatsEye_maxpooling_layer_forward(CatsEye_layer *l, real *s, real *w, real *z, real *o, int u[])
{
	int sx = u[XSIZE];
	int sy = u[YSIZE];
	int *max = (int*)w;

#ifdef CATS_FASTCONV
	// FIXME
/*	int step = sx * ch;
	m *= ch;
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			real *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				real *d = prev_delta;	// in
				for (int wy=ks; wy>0; wy--) {
					for (int wx=ks; wx>0; wx--) {
						for (int cc=ch; cc>0; cc--) {	// in
							*d++ += (*delta) * (*k++);
						}
					}
					d += step;
				}
				delta++;
			}
			prev_delta += ch;
		}
		prev_delta += m;
	}*/
#else
	for (int c=0; c<u[CHANNEL]; c++) {
		for (int y=0; y<sy-1; y+=u[STRIDE]) {
			for (int x=0; x<sx-1; x+=u[STRIDE]) {
				int n = c*sx*sy + y*sx+x;
				real a = s[n];
				*max = n;
				for (int wy=u[KSIZE]; wy>0; wy--) {
					for (int wx=u[KSIZE]; wx>0; wx--) {
						if (a<s[n]) {
							a = s[n];
							*max = n;
						}
						n++;
					}
					n += sx-u[KSIZE];
				}
				max++;
				*o++ = a;
			}
		}
	}
#endif
}
// calculate back propagation
void CatsEye_maxpooling_layer_backward(CatsEye_layer *l, real *o, real *w, real *d, real *delta, int u[])
{
	CATS_ACT dact = CatsEye_dact[u[ACT-LPLEN]];
	int *max = (int*)w;
	memset(d, 0, sizeof(real)*u[SIZE-LPLEN]);
	for (int i=0; i<u[SIZE]; i++) {
		d[*max] = (*delta++) * dact(o[*max]);
		max++;
	}
}
void CatsEye_none_update(CatsEye_layer *l, real eta, real *s, real *w, real *d, int u[])
{
}

void (*CatsEye_layer_forward[])(CatsEye_layer *l, real *s, real *w, real *z, real *o, int u[]) = {
	CatsEye_linear_layer_forward,
	CatsEye_convolutional_layer_forward,
	CatsEye_maxpooling_layer_forward,
};
void (*CatsEye_layer_backward[])(CatsEye_layer *l, real *o, real *w, real *d, real *delta, int u[]) = {
	CatsEye_linear_layer_backward,
	CatsEye_convolutional_layer_backward,
	CatsEye_maxpooling_layer_backward,
};
void (*CatsEye_layer_update[])(CatsEye_layer *l, real eta, real *s, real *w, real *d, int u[]) = {
	CatsEye_linear_layer_update,
//	CatsEye_SVM_layer_update,
	CatsEye_convolutional_layer_update,
	CatsEye_none_update,
};

// calculate the error of output layer
void CatsEye_loss_0_1(CatsEye *this, int c, void *t, int n)
{
	real *d = this->d[c-1];
	real *o = this->o[c];
	int size = SIZE(c);

	int a = ((int*)t)[n];
	for (int i=0; i<size; i++) {
		// 0-1 loss function
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
	// http://d.hatena.ne.jp/echizen_tm/20110606/1307378609
	// E = max(0, -twx), ∂E / ∂w = max(0, -tx)
}
// loss function for mse with identity and cross entropy with sigmoid
void CatsEye_loss_mse(CatsEye *this, int c, void *t, int n)
{
	real *d = this->d[c-1];
	real *o = this->o[c];
	int size = SIZE(c);

	real *a = &((real*)t)[n*size];
	for (int i=0; i<size; i++) {
		// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
		// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
		d[i] = o[i] - a[i];
//		this->d[a-1][i] = this->o[a][i]-((real*)t)[sample*SIZE(a)+i] +avg;
//		this->d[a-1][i] = this->o[a][i]-((real*)t)[sample*SIZE(a)+i] +fabs(this->o[a-1][i])*0.01;
//		this->e3[i] += this->d[1][i]*this->d[1][i];
//		OPT_CALC1(3);
	}
}
void CatsEye_loss_mse_with_sparse(CatsEye *this, int c, void *t, int n)
{
	real *d = this->d[c-1];
	real *o = this->o[c];
	int size = SIZE(c);

	// http://www.vision.is.tohoku.ac.jp/files/9313/6601/7876/CVIM_tutorial_deep_learning.pdf P40
	// http://www.slideshare.net/at_grandpa/chapter5-50042838 P105
	// http://www.slideshare.net/takashiabe338/auto-encoder20140526up P11
/*	real avg = 0;
	for (int i=0; i<SIZE(c-1); i++) {
		avg += this->o[c-1][i];
	}
	avg = avg / SIZE(c-1) *1e-2;//0.001;

	real *a = &((real*)t)[n*size];
	for (int i=0; i<size; i++) {
//		this->d[a-1][i] = this->o[a][i]-a[i] +avg;
		d[i] = o[i]-a[i] + avg;
	}*/

	// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
	// http://www.slideshare.net/YumaMatsuoka/auto-encoder P15
/*	real l1 = 0;
	for (int l=1; l<this->layers-1; l++) {
		for (int i=0; i<SIZE(l); i++) {
			l1 += fabs(this->o[l][i]);
		}
	}
	l1 *= 0.001;*/

	real *a = &((real*)t)[n*size];
	for (int i=0; i<size; i++) {
//		d[i] = o[i]-a[i] - l1;
//		d[i] = o[i]-a[i] - (o[i]>0 ? 1.0 : o[i]<0 ? -1.0 : 0.0)*0.1;
		d[i] = o[i]-a[i] - fabs(o[i])*0.1;
	}
}
void (*CatsEye_loss[])(CatsEye *this, int c, void *t, int n) = {
	CatsEye_loss_0_1,
	CatsEye_loss_mse,
	CatsEye_loss_mse_with_sparse,
};

void CatsEye_backpropagate(CatsEye *this, int n)	// FIXME
{
	for (int i=/*this->layers-2*/n; i>0; i--) {
//		CatsEye_layer_backward[TYPE(i+1)](this->d[i], this->w[i], this->o[i-1], this->o[i], &this->u[LPLEN*(i+1)]);
		CatsEye_layer_backward[TYPE(i+1)](&this->layer[i-1], this->d[i-1], this->w[i], this->o[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
//		CatsEye_layer_backward[TYPE(i+1)](this->d[i-1], this->w[i], this->o[i-1], this->o[i], &this->u[LPLEN*(i+1)]);
	}
}
void CatsEye_propagate(CatsEye *this, int n)	// FIXME
{
	for (int i=n; i<this->layers-1; i++) {
		this->o[i][SIZE(i)] = 1;	// for bias
		CatsEye_layer_forward[TYPE(i+1)](&this->layer[i], this->o[i], this->w[i], this->z[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
	}
}

// calculate forward propagation of input x
void CatsEye_forward(CatsEye *this, real *x)
{
	// calculation of input layer
	memcpy(this->o[0], x, SIZE(0)*sizeof(real));
	this->o[0][SIZE(0)] = 1;	// for bias
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<SIZE(0); i++) {
		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
	}
#endif

	// caluculation of hidden and output layer [z = wx, o = f(z)]
	// z[hidden] += w[in * hidden] * o[0][in]
	// o[1][hidden] = act(z[hidden])
	CatsEye_layer_forward[TYPE(1)](&this->layer[0], this->o[0], this->w[0], this->z[0], this->o[1], &this->u[LPLEN*1]);
	for (int i=1; i<this->layers-1; i++) {
		this->o[i][SIZE(i)] = 1;	// for bias
		CatsEye_layer_forward[TYPE(i+1)](&this->layer[i], this->o[i], this->w[i], this->z[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
	}
}

#ifndef CATS_OPENCL
/* train: multi layer perceptron
 * x: train data (number of elements is in*N)
 * t: correct label (number of elements is N)
 * N: data size
 * repeat: repeat times
 * eta: learning rate (1e-6 to 1) */
void CatsEye_train(CatsEye *this, real *x, void *t, int N, int repeat, real eta)
{
	this->xdata = x;
	this->xsize = N;

	int batch = N;			// for random
	if (RANDOM) batch = RANDOM;

	int a = this->layers-1;
	int loss = this->u[a*LPLEN+STRIDE];
	if (!loss && x==t) loss = CATS_LOSS_MSE;

	struct timeval start, stop;
	gettimeofday(&start, NULL);
	for (int times=0; times<repeat; times++) {
/*#ifndef CATS_OPT_SGD
		memset(this->e3, 0, sizeof(real)*SIZE(2));
		memset(this->e2, 0, sizeof(real)*SIZE(1));
#endif*/
		for (int n=0; n<batch; n++) {
			int sample = RANDOM ? (frand()*N) : n;

			// forward propagation
			CatsEye_forward(this, x+sample*SIZE(0));

			// calculate the error of output layer
			CatsEye_loss[loss](this, a, t, sample);
			// calculate the error of hidden layer
			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_backward[TYPE(i+1)](&this->layer[i], this->o[i], this->w[i], this->d[i-1], this->d[i], &this->u[LPLEN*(i+1)]);
			}

			// update the weights of hidden layer
			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_update[TYPE(i)](&this->layer[i-1], eta, this->o[i-1], this->w[i-1], this->d[i-1], &this->u[LPLEN*i]);
			}
			// update the weights of output layer
			CatsEye_layer_update[TYPE(a)](&this->layer[a-1], eta, this->o[a-1], this->w[a-1], this->d[a-1], &this->u[LPLEN*a]);
#ifdef CATS_AUTOENCODER
			// tied weight
			real *dst = this->w[1];
			for (int i=0; i<SIZE(1); i++) {
				for (int j=0; j<SIZE(0); j++) {
					this->w[1][j + SIZE(1)*i] = this->w[0][SIZE(1)*j + i];
				}
			}
#endif
		}
		real err = 0;
		{
			// calculate the mean squared error
			real mse = 0;
			for (int i=0; i<SIZE(2); i++) {
				mse += 0.5 * (this->d[1][i] * this->d[1][i]);
			}
			err = 0.5 * (err + mse);
		}
		printf("epochs %d, mse %f", times, err);
		gettimeofday(&stop, NULL);
		printf(" [%.2fs]", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);
		printf("\n");
	}
}
#endif

// return most probable label to the input x
int CatsEye_predict(CatsEye *this, real *x)
{
	// forward propagation
	CatsEye_forward(this, x);

	// biggest output means most probable label
	int a = this->layers-1;
	real max = this->o[a][0];
	int ans = 0;
	for (int i=1; i<SIZE(a); i++) {
		if (this->o[a][i] > max) {
			max = this->o[a][i];
			ans = i;
		}
	}
	return ans;
}

// save weights to csv file
int CatsEye_save(CatsEye *this, char *filename)
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
}

// save weights to json file
int CatsEye_saveJson(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (fp==NULL) return -1;

	int *u = this->u;
	for (int n=0; n<this->layers; n++) {
		fprintf(fp, "var u%d = [%d,%d,%d,%d,%d,%d,%d,%d];\n", n, u[TYPE], u[ACT],
			u[CHANNEL], u[SIZE], u[XSIZE], u[YSIZE], u[KSIZE], u[STRIDE]);
		u += LPLEN;
	}
	fprintf(fp, "var u = [u0");
	for (int n=1; n<this->layers; n++) {
		fprintf(fp, ",u%d", n);
	}
	fprintf(fp, "];\n");

	for (int n=0; n<this->layers-1; n++) {
		int i;
		fprintf(fp, "var w%d = [", n+1);
//		if (this->u[TYPE+LPLEN*n] != CATS_MAXPOOL) {
			for (i=0; i<this->ws[n]; i++) {
				fprintf(fp, "%lf,", this->w[n][i]);
			}
			fprintf(fp, "%lf", this->w[n][i]);
//		}
		fprintf(fp, "];\n");
	}
	fprintf(fp, "var w = [w1");
	for (int n=1; n<this->layers-1; n++) {
		fprintf(fp, ",w%d", n+1);
	}
	fprintf(fp, "];\n");

	fclose(fp);
	return 0;
}

// save weights to binary file
int CatsEye_saveBin(CatsEye *this, char *filename)
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
}

// visualize weights [w1]
void CatsEye_visualizeWeights(CatsEye *this, int n, int size, unsigned char *p, int width)
{
	real *w = &this->w[0][n];
	real max = w[0];
	real min = w[0];
	for (int i=1; i<SIZE(0); i++) {
		if (max < w[i * SIZE(1)]) max = w[i * SIZE(1)];
		if (min > w[i * SIZE(1)]) min = w[i * SIZE(1)];
	}
	for (int i=0; i<SIZE(0); i++) {
		p[(i/size)*width + i%size] = ((w[i * SIZE(1)] - min) / (max - min)) * 255.0;
	}
}

// visualize
void CatsEye_visualize(real *o, int n, int size, unsigned char *p, int width)
{
	real max = o[0];
	real min = o[0];
	for (int i=1; i<n; i++) {
		if (max < o[i]) max = o[i];
		if (min > o[i]) min = o[i];
	}
	for (int i=0; i<n; i++) {
		p[(i/size)*width + i%size] = ((o[i] - min) / (max - min)) * 255.0;
	}
}

// visualizeUnits
void CatsEye_visualizeUnits(CatsEye *this, int n, int l, int ch, unsigned char *p, int width)
{
	int *u = &this->u[(l+1)*LPLEN];
	real *s;
	int size, w;
#ifdef CATS_FASTCONV
	switch (n) {
	case 0:
		size = u[XSIZE]*u[YSIZE];
		w = u[XSIZE];
		s = &this->o[l][ch];
		break;
	case 1:
		size = u[KSIZE]*u[KSIZE];
		w = u[KSIZE];
		s = &this->w[l][ch];
	}
	ch = u[CHANNEL-LPLEN];

	real max = s[0];
	real min = s[0];
	for (int i=1; i<size; i++) {
		if (max < s[i*ch]) max = s[i*ch];
		if (min > s[i*ch]) min = s[i*ch];
	}
	for (int i=0; i<size; i++) {
		p[(i/w)*width + i%w] = ((s[i*ch] - min) / (max - min)) * 255.0;
	}
#else
	switch (n) {
	case 0:
		size = u[XSIZE]*u[YSIZE];
		w = u[XSIZE];
		s = &this->o[l][ch * size];
		break;
	case 1:
		size = u[KSIZE]*u[KSIZE];
		w = u[KSIZE];
		s = &this->w[l][ch * size];
	}
	CatsEye_visualize(s, size, w, p, width);
#endif
}

// https://www.cs.toronto.edu/~kriz/cifar.html
real *CatsEye_loadCifar(char *name, int sample, int **label)
{
	unsigned char *data = malloc((32*32*3+1)*sample);		// +1 for label
	if (!data) return 0;
	int *t = malloc(sizeof(int)*sample);
	if (!t) return 0;
	real *x = malloc(sizeof(real)*(32*32*3+1)*(sample+1));	// +1 for bias
	if (!x) return 0;

	FILE *fp = fopen(name, "rb");
	if (!fp) return 0;
	fread(data, (32*32*3+1)*sample, 1, fp);
	for (int n=0; n<sample; n++) {
		t[n] = data[n*(32*32*3+1)];
#ifdef CATS_FASTCONV
		for (int i=0; i<32*32; i++) {
			x[n*32*32*3+i*3  ] = data[n*(32*32*3+1)+1        +i] * (1.0/255.0);	// r
			x[n*32*32*3+i*3+1] = data[n*(32*32*3+1)+1+32*32  +i] * (1.0/255.0);	// g
			x[n*32*32*3+i*3+2] = data[n*(32*32*3+1)+1+32*32*2+i] * (1.0/255.0);	// b
		}
//		x[n*32*32*3+1] = 1;
#else
		for (int i=0; i<32*32*3; i++) x[n*32*32*3+i] = data[n*(32*32*3+1)+1+i] * (1.0/255.0);
#endif
	}
	fclose(fp);
	free(data);

	*label = t;
	return x;
}

real *CatsEye_loadMnist(char *name, char *name2, int sample, int **label)
{
	int size = 784;
	unsigned char *data = malloc((size+1)*sample);		// +1 for label
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

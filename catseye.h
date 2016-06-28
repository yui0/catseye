//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CATS_TIME
#ifdef CATS_TIME
#include <sys/time.h>
#endif

#ifdef CATS_USE_FIXED
#define numerus		short
#elif defined CATS_USE_FLOAT
#define numerus		float
#else
#define numerus		double
#endif

#define CATS_SIGMOID
//#define CATS_TANH
//#define CATS_SCALEDTANH
//#define CATS_RELU
//#define CATS_ABS

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
typedef unsigned long long	uint64_t;
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
int binomial(/*int n, */numerus p)
{
//	if (p<0 || p>1) return 0;
	int c = 0;
//	for (int i=0; i<n; i++) {
		numerus r = frand();
		if (r < p) c++;
//	}
	return c;
}

typedef struct {
	// number of each layer
	int layers, *u;

	// input layers
	numerus *xdata;
	int xsize;
	// output layers [o = f(z)]
	numerus **z, **o, *odata;
	int osize;
	// error value
	numerus **d, *ddata;
	int dsize;
	// weights
	numerus **w, *wdata;
	int *ws, wsize;

	// gradient value
//	numerus *e2, *e3, *m2, *v2, *m3, *v3;
//	numerus *dl1, *dl2;

	// deprecated!
	int in;
	numerus *o3;
} CatsEye;

// identity function (output only)
numerus CatsEye_act_identity(numerus x)
{
	return (x);
}
numerus CatsEye_dact_identity(numerus x)
{
	return (1.0);
}
// softmax function (output only)
numerus CatsEye_act_softmax(numerus x/* *x, int n, int len*/)
{
/*	numerus alpha = x[0];
	for (int i=1; i<len; i++) if (alpha<x[i]) alpha = x[i];
	numerus numer = exp(x[n] - alpha);
	numerus denom = 0.0;
	for (int i=0; i<len; i++) denom += exp(x[i] - alpha);
	return (numer / denom);*/
	return (x);
}
numerus CatsEye_dact_softmax(numerus x)
{
	return (x * (1.0 - x));
}
// sigmoid function
numerus CatsEye_act_sigmoid(numerus x)
{
	return (1.0 / (1.0 + exp(-x * s_gain)));
}
numerus CatsEye_dact_sigmoid(numerus x)
{
	return ((1.0-x)*x * s_gain);	// ((1.0-sigmod(x))*sigmod(x))
}
// tanh function
// https://github.com/nyanp/tiny-cnn/blob/master/tiny_cnn/activations/activation_function.h
numerus CatsEye_act_tanh(numerus x)
{
//	return (tanh(x));

/*	numerus ep = exp(x[n]);
	numerus em = exp(-x[n]);
	return (ep-em) / (ep+em);*/

	// fast approximation of tanh (improve 2-3% speed in LeNet-5)
	numerus x1 = x;
	numerus x2 = x1 * x1;
	x1 *= 1.0 + x2 * (0.1653 + x2 * 0.0097);
	return x1 / sqrt(1.0 + x2);
}
numerus CatsEye_dact_tanh(numerus x)
{
	return (1.0-x*x);		// (1.0-tanh(x)*tanh(x))
}
// scaled tanh function
numerus CatsEye_act_scaled_tanh(numerus x)
{
	return (1.7159 * tanh(2.0/3.0 * x));
}
numerus CatsEye_dact_scaled_tanh(numerus x)
{
	return ((2.0/3.0)/1.7159 * (1.7159-x)*(1.7159+x));
}
// rectified linear unit function
numerus CatsEye_act_ReLU(numerus x)
{
	return (x>0 ? x : 0.0);
}
numerus CatsEye_dact_ReLU(numerus x)
{
	return (x>0 ? 1.0 : 0.0);
}
// leaky rectified linear unit function
#define leaky_alpha	0.01	// 0 - 1
numerus CatsEye_act_LeakyReLU(numerus x)
{
	return (x>0 ? x : x*leaky_alpha);
}
numerus CatsEye_dact_LeakyReLU(numerus x)
{
	return (x>0 ? 1.0 : leaky_alpha);
}
// exponential rectified linear unit function
// http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/elu.html
numerus CatsEye_act_ELU(numerus x)
{
	return (x>0 ? x : exp(x)-1.0);
}
numerus CatsEye_dact_ELU(numerus x)
{
	return (x>0 ? 1.0 : 1.0+x);
}
// abs function
numerus CatsEye_act_abs(numerus x)
{
	return (x / (1.0 + fabs(x)));
}
numerus CatsEye_dact_abs(numerus x)
{
	return (1.0 / (1.0 + fabs(x))*(1.0 + fabs(x)));
}

// activation function and derivative of activation function
numerus (*CatsEye_act[])(numerus x) = {
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
numerus (*CatsEye_dact[])(numerus x) = {
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
enum CATS_ACTIVATION_FUNCTION {
	CATS_ACT_IDENTITY,
	CATS_ACT_SOFTMAX,
	CATS_ACT_SIGMOID,
	CATS_ACT_TANH,
	CATS_ACT_SCALED_TANH,
	CATS_ACT_RELU,
	CATS_ACT_LEAKY_RELU,
	CATS_ACT_ELU,
	CATS_ACT_ABS,
};
typedef numerus (*CATS_ACT)(numerus x);

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
numerus dot(numerus *vec1, numerus *vec2, int n)
{
	numerus s = 0.0;
	#pragma omp simd reduction(+:s)
	for (int i=n; i>0; i--) {
		s += (*vec1++) * (*vec2++);
	}
	return s;
}
numerus dotT(numerus *mat1, numerus *vec1, int r, int c)
{
	numerus s = 0.0;
	#pragma omp simd reduction(+:s)
	for (int i=r; i>0; i--) {
		s += *mat1 * (*vec1++);
		mat1 += c;
	}
	return s;
}
void muladd(numerus *vec1, numerus *vec2, numerus a, int n)
{
	#pragma omp for simd
	for (int i=n; i>0; i--) {
		*vec1++ += a * (*vec2++);
	}
}
void muladdx(numerus *vec1, numerus *vec2, numerus a, int n)
{
	#pragma omp for simd
	for (int i=n; i>0; i--) {
		*vec1 += a * (*vec2++) + (*vec1) * 1e-8;
		vec1++;
	}
}
#endif

// calculate forward propagation of input x
// f(x) = h(scale*x+bias)
void CatsEye_linear_layer_forward(numerus *x, numerus *w, numerus *z/*no use*/, numerus *o, int u[])
{
	int in = u[SIZE-LPLEN]+1;	// +1 -> for bias
	int out = u[SIZE];

	CATS_ACT act = CatsEye_act[u[ACT]];
	for (int i=out; i>0; i--) {
		*o++ = act(dotT(w++, x, in, out));
//		*o++ = act(dot(w, x, in));
//		w += in;
	}
}
// calculate back propagation
void CatsEye_linear_layer_backward(numerus *o, numerus *w, numerus *d, numerus *delta, int u[])
{
	int in = u[SIZE-LPLEN];
	int out = u[SIZE];

	// calculate the error
	CATS_ACT dact = CatsEye_dact[u[ACT-LPLEN]];
	for (int i=0; i<=in; i++) {	// bias!!
		*d++ = dot(&w[i*out], delta, out) * dact(*o++);
//		*d++ = dotT(w++, delta, out, in+1) * dact(*o++);
	}
}
void CatsEye_linear_layer_update(numerus eta, numerus *o, numerus *w, numerus *d, int u[])
{
	int in = u[SIZE-LPLEN]+1;	// +1 -> for bias
	int out = u[SIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		numerus a = eta * (*o++);
		for (int j=0; j<out; j++) {
			*w++ -= a*d[j];
		}
	}
}
void CatsEye_SVM_layer_update(numerus eta, numerus *o, numerus *w, numerus *d, int u[])
{
	int in = u[SIZE-LPLEN]+1;
	int out = u[SIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		// SVM (http://d.hatena.ne.jp/echizen_tm/20110627/1309188711)
		// ∂loss(w, x, t) / ∂w = ∂(λ - twx + α * w^2 / 2) / ∂w = - tx + αw
		numerus a = -eta * (*o++);
		muladdx(w, d, a, out);	// *w -= a* (*dd++) + (*w)*1e-8; w++;
		w += out;
//		muladd(w++, d, a, out);	// *w -= a* (*dd++) + (*w)*1e-8; w++;
	}
}

// calculate forward propagation
void CatsEye_convolutional_layer_forward(numerus *s, numerus *w, numerus *_z/*no use*/, numerus *o, int u[])
{
	int ks = u[KSIZE];
	int k2 = ks * ks;
	int n = u[CHANNEL] * k2;
	int m = (ks/2)*2;
	int sx = u[XSIZE] - m;	// out
	int sy = u[YSIZE] - m;
	int ch = u[CHANNEL-LPLEN];
	int size = u[SIZE-LPLEN] / ch;
	int step = u[XSIZE] - ks;
	CATS_ACT act = CatsEye_act[u[ACT]];

#ifdef CATS_FASTCONV
	// h, w, c[in], c[out]
	m *= ch;
	step *= ch;
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			numerus *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				numerus *p = s;	// in
				numerus a = 0;
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
	numerus *z, *p, *r;
	numerus *pp = s;
	memset(o, 0, sizeof(numerus)*u[CHANNEL]*sx*sy);
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
void CatsEye_convolutional_layer_backward(numerus *prev_out, numerus *w, numerus *prev_delta, numerus *delta, int u[])
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
	memset(prev_delta, 0, sizeof(numerus)*ch*ix*iy);

#ifdef CATS_FASTCONV
	numerus *p = prev_delta;
	step *= ch;
	m *= ch;
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			numerus *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				numerus *d = prev_delta;	// in
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
	numerus *d, *p, *r;
	numerus *pp = prev_delta;
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
void CatsEye_convolutional_layer_update(numerus eta, numerus *prev_out, numerus *w, numerus *curr_delta, int u[])
{
	int ks = u[KSIZE];
	int m = (ks/2)*2;
	int sx = u[XSIZE] - m;	// out
	int sy = u[YSIZE] - m;
	int ch = u[CHANNEL-LPLEN];
	int size = u[SIZE-LPLEN]/ch;
	int step = u[XSIZE] - ks;

#ifdef CATS_FASTCONV
	step *= ch;
	m *= ch;
	numerus *d = curr_delta;	// out
	for (int y=sy; y>0; y--) {
		for (int x=sx; x>0; x--) {
			numerus *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				numerus *p = prev_out;	// in
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
	numerus *d, *p, *r;
	numerus *pp = prev_out;
	for (int c=u[CHANNEL]; c>0; c--) {	// out
		r = prev_out;
		for (int cc=ch; cc>0; cc--) {	// in
			for (int wy=ks; wy>0; wy--) {
				for (int wx=ks; wx>0; wx--) {
					p = prev_out++;	// in
					d = curr_delta;	// out
					numerus a = 0;
					for (int y=sy; y>0; y--) {
						a += dot(d, p, sx);
//						a -= dot(d, p, sx) * eta;
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
void CatsEye_maxpooling_layer_forward(numerus *s, numerus *w, numerus *z, numerus *o, int u[])
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
			numerus *k = w;
			for (int c=u[CHANNEL]; c>0; c--) {	// out
				numerus *d = prev_delta;	// in
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
				numerus a = s[n];
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
void CatsEye_maxpooling_layer_backward(numerus *o, numerus *w, numerus *d, numerus *delta, int u[])
{
	CATS_ACT dact = CatsEye_dact[u[ACT-LPLEN]];
	int *max = (int*)w;
	memset(d, 0, sizeof(numerus)*u[SIZE-LPLEN]);
	for (int i=0; i<u[SIZE]; i++) {
		d[*max] = (*delta++) * dact(o[*max]);
		max++;
	}
}
void CatsEye_none_update(numerus eta, numerus *s, numerus *w, numerus *d, int u[])
{
}

void (*CatsEye_layer_forward[])(numerus *s, numerus *w, numerus *z, numerus *o, int u[]) = {
	CatsEye_linear_layer_forward,
	CatsEye_convolutional_layer_forward,
	CatsEye_maxpooling_layer_forward,
};
void (*CatsEye_layer_backward[])(numerus *o, numerus *w, numerus *d, numerus *delta, int u[]) = {
	CatsEye_linear_layer_backward,
	CatsEye_convolutional_layer_backward,
	CatsEye_maxpooling_layer_backward,
};
void (*CatsEye_layer_update[])(numerus eta, numerus *s, numerus *w, numerus *d, int u[]) = {
//	CatsEye_linear_layer_update,
	CatsEye_SVM_layer_update,
	CatsEye_convolutional_layer_update,
	CatsEye_none_update,
};
enum CATS_LAYER_TYPE {
	CATS_LINEAR,
	CATS_CONV,
	CATS_MAXPOOL,
};

//#define CATS_OPENCL
#ifdef CATS_OPENCL
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
	this->z = malloc(sizeof(numerus*)*(this->layers-1));
/*	for (int i=0; i<this->layers-1; i++) {
		this->z[i] = malloc(sizeof(numerus)*(SIZE(i+1)+1));
	}*/

	// allocate outputs
	int size[this->layers];
	this->o = malloc(sizeof(numerus*)*(this->layers));
	this->osize = 0;
	for (int i=0; i<this->layers; i++) {
		size[i] = this->osize;
		this->osize += SIZE(i);//+1;
	}
	this->odata = malloc(sizeof(numerus)*this->osize);
	for (int i=0; i<this->layers; i++) this->o[i] = this->odata + size[i];
	this->o3 = this->o[2];	// deprecated!

	// allocate errors
	this->d = malloc(sizeof(numerus*)*(this->layers-1));
	this->dsize = 0;
	for (int i=0; i<this->layers-1; i++) {
		size[i] = this->dsize;
		this->dsize += SIZE(i+1)+1;
	}
	this->ddata = malloc(sizeof(numerus)*this->dsize);
	for (int i=0; i<this->layers-1; i++) this->d[i] = this->ddata + size[i];

	// allocate gradient
/*	this->e2 = calloc(1, sizeof(numerus)*(SIZE(1)+1));
	this->e3 = calloc(1, sizeof(numerus)*SIZE(2));
	this->m2 = calloc(1, sizeof(numerus)*(SIZE(1)+1));
	this->m3 = calloc(1, sizeof(numerus)*SIZE(2));
	this->v2 = calloc(1, sizeof(numerus)*(SIZE(1)+1));
	this->v3 = calloc(1, sizeof(numerus)*SIZE(2));
	this->dl1 = malloc(sizeof(numerus)*(SIZE(0)+1)*SIZE(1));
	this->dl2 = malloc(sizeof(numerus)*(SIZE(1)+1)*SIZE(2));*/

	// allocate weights
	this->w = malloc(sizeof(numerus*)*(this->layers-1));
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
		size[i] = this->wsize;
		this->wsize += this->ws[i];
	}
	this->wdata = malloc(sizeof(numerus)*this->wsize);
	for (int i=0; i<this->layers-1; i++) {
		this->w[i] = this->wdata + size[i];
//		this->w[i] = malloc(sizeof(numerus)*(n[i]+1)*m[i]);
//		if (!this->w[i]) printf("memory error at layer %d[%d], size %d!!\n", i+1, u[TYPE], this->ws[i]);

		// initialize weights (http://aidiary.hatenablog.com/entry/20150618/1434628272)
		// range depends on the research of Y. Bengio et al. (2010)
		xor128_init(time(0));
		numerus range = sqrt(6)/sqrt(n[i]+m[i]+2);
		for (int j=0; j<this->ws[i]; j++) {
			this->w[i][j] = 2.0*range*frand()-range;
		}
	}

	if (param) {
		for (int i=0; i<(SIZE(0)+1)*SIZE(1); i++) {
			fscanf(fp, "%lf ", &this->w[0][i]);
		}
		for (int i=0; i<(SIZE(1)+1)*SIZE(2); i++) {
			fscanf(fp, "%lf ", &this->w[1][i]);
		}
		fclose(fp);
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
	free(this->z);
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
	free(this->u);
}

void CatsEye_backpropagate(CatsEye *this, int n)
{
	for (int i=/*this->layers-2*/n; i>0; i--) {
//		CatsEye_layer_backward[TYPE(i+1)](this->d[i], this->w[i], this->o[i-1], this->o[i], &this->u[LPLEN*(i+1)]);
		CatsEye_layer_backward[TYPE(i+1)](this->d[i-1], this->w[i], this->o[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
//		CatsEye_layer_backward[TYPE(i+1)](this->d[i-1], this->w[i], this->o[i-1], this->o[i], &this->u[LPLEN*(i+1)]);
	}
}
void CatsEye_propagate(CatsEye *this, int n)
{
	for (int i=n; i<this->layers-1; i++) {
		this->o[i][SIZE(i)] = 1;	// for bias
		CatsEye_layer_forward[TYPE(i+1)](this->o[i], this->w[i], this->z[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
	}
}
#ifndef CATS_OPENCL
// calculate forward propagation of input x
void CatsEye_forward(CatsEye *this, numerus *x, int n)
{
	// calculation of input layer
	memcpy(this->o[0], x+n, SIZE(0)*sizeof(numerus));
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
	CatsEye_layer_forward[TYPE(1)](this->o[0], this->w[0], this->z[0], this->o[1], &this->u[LPLEN*1]);
	for (int i=1; i<this->layers-1; i++) {
		this->o[i][SIZE(i)] = 1;	// for bias
		CatsEye_layer_forward[TYPE(i+1)](this->o[i], this->w[i], this->z[i], this->o[i+1], &this->u[LPLEN*(i+1)]);
	}
}
#endif

#define CATS_NO_MINIBATCH
// calculate the error of output layer
void CatsEye_loss_0_1(CatsEye *this, int c, void *t, int n)
{
	numerus *d = this->d[c-1];
	numerus *o = this->o[c];
	int size = SIZE(c);

	int a = ((int*)t)[n];
	for (int i=0; i<size; i++) {
		// 0-1 loss function
#ifdef CATS_NO_MINIBATCH
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
#else
		numerus e = a==i ? o[i]-1 : o[i];	// 1-of-K
		e += d[i];
		d[i] = e * 0.5;
#endif
	}
	// Ref.
	// http://d.hatena.ne.jp/echizen_tm/20110606/1307378609
	// E = max(0, -twx), ∂E / ∂w = max(0, -tx)
}
// loss function for mse with identity and cross entropy with sigmoid
void CatsEye_loss_mse(CatsEye *this, int c, void *t, int n)
{
	numerus *d = this->d[c-1];
	numerus *o = this->o[c];
	int size = SIZE(c);

	numerus *a = &((numerus*)t)[n*size];
	for (int i=0; i<size; i++) {
		// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
		// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
		d[i] = o[i] - a[i];
//		this->d[a-1][i] = this->o[a][i]-((numerus*)t)[sample*SIZE(a)+i] +avg;
//		this->d[a-1][i] = this->o[a][i]-((numerus*)t)[sample*SIZE(a)+i] +fabs(this->o[a-1][i])*0.01;
//		this->e3[i] += this->d[1][i]*this->d[1][i];
//		OPT_CALC1(3);
	}
}
void CatsEye_loss_mse_with_sparse(CatsEye *this, int c, void *t, int n)
{
	numerus *d = this->d[c-1];
	numerus *o = this->o[c];
	int size = SIZE(c);

	// http://www.vision.is.tohoku.ac.jp/files/9313/6601/7876/CVIM_tutorial_deep_learning.pdf P40
	// http://www.slideshare.net/at_grandpa/chapter5-50042838 P105
	// http://www.slideshare.net/takashiabe338/auto-encoder20140526up P11
/*	numerus avg = 0;
	for (int i=0; i<SIZE(c-1); i++) {
		avg += this->o[c-1][i];
	}
	avg = avg / SIZE(c-1) *1e-2;//0.001;

	numerus *a = &((numerus*)t)[n*size];
	for (int i=0; i<size; i++) {
//		this->d[a-1][i] = this->o[a][i]-a[i] +avg;
		d[i] = o[i]-a[i] + avg;
	}*/

	// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
	// http://www.slideshare.net/YumaMatsuoka/auto-encoder P15
/*	numerus l1 = 0;
	for (int l=1; l<this->layers-1; l++) {
		for (int i=0; i<SIZE(l); i++) {
			l1 += fabs(this->o[l][i]);
		}
	}
	l1 *= 0.001;*/

	numerus *a = &((numerus*)t)[n*size];
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
/* train: multi layer perceptron
 * x: train data (number of elements is in*N)
 * t: correct label (number of elements is N)
 * N: data size
 * repeat: repeat times
 * eta: learning rate (1e-6 to 1) */
void CatsEye_train(CatsEye *this, numerus *x, void *t, int N, int repeat, numerus eta)
{
	this->xdata = x;
	this->xsize = N;

	int batch = N;			// for random
	if (RANDOM) batch = RANDOM;

	int a = this->layers-1;
	int loss = this->u[a*LPLEN+STRIDE];
	if (!loss && x==t) loss = 1;

#ifdef CATS_TIME
	struct timeval start, stop;
	gettimeofday(&start, NULL);
#endif
	for (int times=0; times<repeat; times++) {
		numerus err = 0;
#ifndef CATS_OPT_SGD
		memset(this->e3, 0, sizeof(numerus)*SIZE(2));
		memset(this->e2, 0, sizeof(numerus)*SIZE(1));
#endif
//#pragma omp parallel
		for (int n=0; n<batch; n++) {
			int sample = RANDOM ? (frand()*N) : n;

			// forward propagation
			CatsEye_forward(this, x, sample*SIZE(0));

			// calculate the error of output layer
			CatsEye_loss[loss](this, a, t, sample);
#ifndef CATS_NO_MINIBATCH
		}
		{
#endif
			// calculate the error of hidden layer
			// t[hidden] += w[1][hidden * out] * d[1][out]
			// d[hidden] = t[hidden] * dact(o[hidden])
			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_backward[TYPE(i+1)](this->o[i], this->w[i], this->d[i-1], this->d[i], &this->u[LPLEN*(i+1)]);
			}
			// update the weights of hidden layer
			// w[0][in] -= eta * o[0][in] * d[0][in * hidden]
			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_update[TYPE(i)](eta, this->o[i-1], this->w[i-1], this->d[i-1], &this->u[LPLEN*i]);
			}

			// update the weights of output layer
			CatsEye_layer_update[TYPE(a)](eta, this->o[a-1], this->w[a-1], this->d[a-1], &this->u[LPLEN*a]);
#ifdef CATS_AUTOENCODER
			// tied weight
			numerus *dst = this->w[1];
			for (int i=0; i<SIZE(1); i++) {
				for (int j=0; j<SIZE(0); j++) {
					this->w[1][j + SIZE(1)*i] = this->w[0][SIZE(1)*j + i];
				}
			}
#endif
		}
		{
			// calculate the mean squared error
			numerus mse = 0;
			for (int i=0; i<SIZE(2); i++) {
				mse += 0.5 * (this->d[1][i] * this->d[1][i]);
			}
			err = 0.5 * (err + mse);
		}
		printf("epochs %d, mse %f", times, err);
#ifdef CATS_TIME
		gettimeofday(&stop, NULL);
		printf(" [%.2fs]", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);
#endif
		printf("\n");
	}
}

// return most probable label to the input x
int CatsEye_predict(CatsEye *this, numerus *x)
{
	// forward propagation
	CatsEye_forward(this, x+1, -1);	// FIXME

	// biggest output means most probable label
	int a = this->layers-1;
	numerus max = this->o[a][0];
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

	//fwrite(this->w[0], sizeof(numerus)*(SIZE(0)+1)*SIZE(1), 1, fp);
	//fwrite(this->w[1], sizeof(numerus)*(SIZE(1)+1)*SIZE(2), 1, fp);
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
	numerus *w = &this->w[0][n];
	numerus max = w[0];
	numerus min = w[0];
	for (int i=1; i<SIZE(0); i++) {
		if (max < w[i * SIZE(1)]) max = w[i * SIZE(1)];
		if (min > w[i * SIZE(1)]) min = w[i * SIZE(1)];
	}
	for (int i=0; i<SIZE(0); i++) {
		p[(i/size)*width + i%size] = ((w[i * SIZE(1)] - min) / (max - min)) * 255.0;
	}
}

// visualize
void CatsEye_visualize(numerus *o, int n, int size, unsigned char *p, int width)
{
	numerus max = o[0];
	numerus min = o[0];
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
	numerus *s;
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

	numerus max = s[0];
	numerus min = s[0];
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
numerus *CatsEye_loadCifar(char *name, int sample, int **label)
{
	unsigned char *data = malloc((32*32*3+1)*sample);		// +1 for label
	if (!data) return 0;
	int *t = malloc(sizeof(int)*sample);
	if (!t) return 0;
	numerus *x = malloc(sizeof(numerus)*(32*32*3+1)*(sample+1));	// +1 for bias
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

numerus *CatsEye_loadMnist(char *name, char *name2, int sample, int **label)
{
	int size = 784;
	unsigned char *data = malloc((size+1)*sample);		// +1 for label
	if (!data) return 0;
	int *t = malloc(sizeof(int)*sample);
	if (!t) return 0;
	numerus *x = malloc(sizeof(numerus)*(size+1)*(sample+1));	// +1 for bias
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

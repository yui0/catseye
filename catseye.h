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

#define CATS_RANDOM

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

// activation function and derivative of activation function
#ifdef CATS_SIGMOID
// sigmoid function
#define ACT1(x)			(1.0 / (1.0 + exp(-x * s_gain)))
#define DACT1(x)		((1.0-x)*x * s_gain)	// ((1.0-sigmod(x))*sigmod(x))
#elif defined CATS_TANH
// tanh function
#define ACT1(x)			(tanh(x))
#define DACT1(x)		(1.0-x*x)		// (1.0-tanh(x)*tanh(x))
#elif defined CATS_SCALEDTANH
// scaled tanh function
#define ACT1(x)			(1.7159 * tanh(2.0/3.0 * x))
#define DACT1(x)		((2.0/3.0)/1.7159 * (1.7159-x)*(1.7159+x))
#elif defined CATS_RELU
// rectified linear unit function
#define ACT1(x)			(x>0 ? x : 0.0)
#define DACT1(x)		(x>0 ? 1.0 : 0.0)
#elif defined CATS_ABS
// abs function
#define ACT1(x)			(x / (1.0 + fabs(x)))
#define DACT1(x)		(1.0 / (1.0 + fabs(x))*(1.0 + fabs(x)))
#else
// identity function (output only)
#define ACT1(x)			(x)
#define DACT1(x)		(1.0)
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

void muladd(double *vec1, double *vec2, double a, int n)
{
	for (int i=0; i<n; i++) {
		vec1[i] += vec2[i] * a;
	}
}
double dot(double *vec1, double *vec2, int n)
{
	double s = 0.0;
	for (int i=0; i<n; i++) {
		s += vec1[i] * vec2[i];
	}
	return s;
}
double dotT(double *mat1, double *vec1, int r, int c)
{
	double s = 0.0;
	for (int i=0; i<r; i++) {
		s += mat1[i*c] * vec1[i];
	}
	return s;
}
/*void transpose(double *src, double *dst, int N, int M)
{
//	#pragma omp parallel for
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			*dst++ = src[M*j + i];
		}
	}
}*/
int binomial(/*int n, */double p)
{
//	if (p<0 || p>1) return 0;
	int c = 0;
//	for (int i=0; i<n; i++) {
		double r = rand() / (RAND_MAX + 1.0);
		if (r < p) c++;
//	}
	return c;
}

typedef struct {
	// number of each layer
	int layers, *u;
	// input layer
//	double *x;
	// output layers [o = f(z)]
	double **z, **o;
	// error value
	double **d;
	// gradient value
	double *e2, *e3, *m2, *v2, *m3, *v3;
	double *dl1, *dl2;
	// weights
	double **w;
} CatsEye;

// identity function (output only)
double CatsEye_act_identity(double *x, int n, int len)
{
	return (x[n]);
}
double CatsEye_dact_identity(double *x, int n, int len)
{
	return (1.0);
}
// softmax function (output only)
double CatsEye_act_softmax(double *x, int n, int len)
{
	double alpha = x[0];
	for (int i=1; i<len; i++) if (alpha<x[i]) alpha = x[i];
	double numer = exp(x[n] - alpha);
	double denom = 0.0;
	for (int i=0; i<len; i++) denom += exp(x[i] - alpha);
	return (numer / denom);
}
double CatsEye_dact_softmax(double *x, int n, int len)
{
	return (x[n] * (1.0 - x[n]));
}
// sigmoid function
double CatsEye_act_sigmoid(double *x, int n, int len)
{
	return (1.0 / (1.0 + exp(-x[n] * s_gain)));
}
double CatsEye_dact_sigmoid(double *x, int n, int len)
{
	return ((1.0-x[n])*x[n] * s_gain);	// ((1.0-sigmod(x))*sigmod(x))
}
// tanh function
double CatsEye_act_tanh(double *x, int n, int len)
{
	return (tanh(x[n]));
}
double CatsEye_dact_tanh(double *x, int n, int len)
{
	return (1.0-x[n]*x[n]);		// (1.0-tanh(x)*tanh(x))
}
// scaled tanh function
double CatsEye_act_scaled_tanh(double *x, int n, int len)
{
	return (1.7159 * tanh(2.0/3.0 * x[n]));
}
double CatsEye_dact_scaled_tanh(double *x, int n, int len)
{
	return ((2.0/3.0)/1.7159 * (1.7159-x[n])*(1.7159+x[n]));
}
// rectified linear unit function
double CatsEye_act_ReLU(double *x, int n, int len)
{
	return (x[n]>0 ? x[n] : 0.0);
}
double CatsEye_dact_ReLU(double *x, int n, int len)
{
	return (x[n]>0 ? 1.0 : 0.0);
}
// abs function
double CatsEye_act_abs(double *x, int n, int len)
{
	return (x[n] / (1.0 + fabs(x[n])));
}
double CatsEye_dact_abs(double *x, int n, int len)
{
	return (1.0 / (1.0 + fabs(x[n]))*(1.0 + fabs(x[n])));
}

// activation function and derivative of activation function
double (*CatsEye_act[])(double *x, int n, int len) = {
	CatsEye_act_identity,
	CatsEye_act_softmax,
	CatsEye_act_sigmoid,
	CatsEye_act_tanh,
	CatsEye_act_scaled_tanh,
	CatsEye_act_ReLU,
	CatsEye_act_abs
};
double (*CatsEye_dact[])(double *x, int n, int len) = {
	CatsEye_dact_identity,
	CatsEye_dact_softmax,
	CatsEye_dact_sigmoid,
	CatsEye_dact_tanh,
	CatsEye_dact_scaled_tanh,
	CatsEye_dact_ReLU,
	CatsEye_dact_abs
};

#define CATS_LPARAM	7
enum CATS_LP {
	TYPE,
	ACT,
	CHANNEL,
	XSIZE,
	YSIZE,
	KSIZE,
	STRIDE,
};
//#define CATS_U(i, x)	this->u[CATS_LPARAM*i+x]
#define CATS_SIZE(i)	this->u[CATS_LPARAM*(i)+XSIZE]
// caluculate forward propagation of input x
void CatsEye_layer_forward(double *x, double *w, double *z, double *o, int u[])
{
	int in = u[XSIZE-CATS_LPARAM]+1;
	int out = u[XSIZE];
	for (int i=0; i<out; i++) {
		z[i] = dotT(&w[i], x, in, out);
		o[i] = CatsEye_act[u[ACT]](z, i, out);
	}
//	o[out] = 1;	// for bias
}
// caluculate back propagation
void CatsEye_layer_backward(double *o, double *w, double *d, double *delta, int u[])
{
	int in = u[XSIZE];
	int out = u[XSIZE+CATS_LPARAM];

	// calculate the error
	for (int i=0; i<in; i++) {
		d[i] = dot(&w[i*out], delta, out) * CatsEye_dact[u[ACT]](o, i, in);
		//OPT_CALC1(2);
	}
	d[in] = dot(&w[in*out], delta, out) * CatsEye_dact[u[ACT]](o, in, in);
}
void CatsEye_layer_update(double eta, double *o, double *w, double *d, int u[])
/*{
	int in = u[XSIZE-CATS_LPARAM]+1;
	int out = u[XSIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		for (int j=0; j<out; j++) {
			w[i*out+j] -= eta*o[i]*d[j];
			//OPT_CALC2(out, 1, 2);
		}
	}
}
void CatsEye_SVM_update(double eta, double *o, double *w, double *d, int u[])*/
{
	int in = u[XSIZE-CATS_LPARAM]+1;
	int out = u[XSIZE];

	// update the weights
	for (int i=0; i<in; i++) {
		for (int j=0; j<out; j++) {
			// SVM (http://d.hatena.ne.jp/echizen_tm/20110627/1309188711)
			// ∂loss(w, x, t) / ∂w = ∂(λ - twx + α * w^2 / 2) / ∂w = - tx + αw
			w[i*out+j] -= eta*o[i]*d[j] + w[i*out+j]*1e-8;
		}
	}
}

// caluculate forward propagation of input x
void CatsEye_convolutional_layer_forward(double *xx, double *w, double *z, double *o, int u[])
{
	int sx = u[XSIZE] - u[KSIZE];
	int sy = u[YSIZE] - u[KSIZE];

	for (int y=0; y<sy; y++) {
		for (int x=0; x<sx; x++) {
			double *p = &xx[y*sx+x];
			double a = 0;
			double *k = w;
			for (int wy=0; wy<u[KSIZE]; wy++) {
				for (int wx=0; wx<u[KSIZE]; wx++) {
					a += p[wx] * (*k++);
				}
				p += sx;
			}
//			*o++ = a + bias;
//			*o++ = CatsEye_act[act](a);
		}
	}
}

/* constructor
 * n_in:  number of input layer
 * n_hid: number of hidden layer
 * n_out: number of output layer */
void CatsEye__construct(CatsEye *this, int n_in, int n_hid, int n_out, char *filename)
{
	int u[] = {
		0, 0, 1, n_in,    0, 0, 0,
		0, 2, 1, n_hid,   0, 0, 0,
//		0, 2, 1, n_hid/2, 0, 0, 0,
		0, 2, 1, n_out,   0, 0, 0,
//		0, 0, 1, n_out,   0, 0, 0,
	};
	this->layers = 3;
	this->u = malloc(sizeof(int)*CATS_LPARAM*this->layers);
	memcpy(this->u, u, sizeof(int)*CATS_LPARAM*this->layers);

	FILE *fp;
	if (filename) {
		fp = fopen(filename, "r");
		if (fp==NULL) return;
		fscanf(fp, "%d %d %d\n", &CATS_SIZE(0), &CATS_SIZE(1), &CATS_SIZE(2));
	} else {
//		CATS_SIZE(0) = n_in;
//		CATS_SIZE(1) = n_hid;
//		CATS_SIZE(2) = n_out;
		for (int i=0; i<this->layers; i++) CATS_SIZE(i) = this->u[CATS_LPARAM*i+XSIZE];
	}

	// allocate inputs
	this->z = malloc(sizeof(double*)*(this->layers-1));
	for (int i=0; i<this->layers-1; i++) {
		this->z[i] = malloc(sizeof(double)*(CATS_SIZE(i+1)+1));
	}

	// allocate outputs
	this->o = malloc(sizeof(double*)*(this->layers));
	for (int i=0; i<this->layers; i++) {
		this->o[i] = malloc(sizeof(double)*(CATS_SIZE(i)+1));
	}

	// allocate errors
	this->d = malloc(sizeof(double*)*(this->layers-1));
	for (int i=0; i<this->layers-1; i++) {
		this->d[i] = malloc(sizeof(double)*(CATS_SIZE(i+1)+1));
	}

	// allocate gradient
	this->e2 = calloc(1, sizeof(double)*(CATS_SIZE(1)+1));
	this->e3 = calloc(1, sizeof(double)*CATS_SIZE(2));
	this->m2 = calloc(1, sizeof(double)*(CATS_SIZE(1)+1));
	this->m3 = calloc(1, sizeof(double)*CATS_SIZE(2));
	this->v2 = calloc(1, sizeof(double)*(CATS_SIZE(1)+1));
	this->v3 = calloc(1, sizeof(double)*CATS_SIZE(2));
	this->dl1 = malloc(sizeof(double)*(CATS_SIZE(0)+1)*CATS_SIZE(1));
	this->dl2 = malloc(sizeof(double)*(CATS_SIZE(1)+1)*CATS_SIZE(2));

	// allocate memories
	this->w = malloc(sizeof(double*)*(this->layers-1));
	for (int i=0; i<this->layers-1; i++) {
		this->w[i] = malloc(sizeof(double)*(CATS_SIZE(i)+1)*CATS_SIZE(i+1));
	}

	if (filename) {
		for (int i=0; i<(CATS_SIZE(0)+1)*CATS_SIZE(1); i++) {
			fscanf(fp, "%lf ", &this->w[0][i]);
		}
		for (int i=0; i<(CATS_SIZE(1)+1)*CATS_SIZE(2); i++) {
			fscanf(fp, "%lf ", &this->w[1][i]);
		}
		fclose(fp);
	} else {
		// initialize weights (http://aidiary.hatenablog.com/entry/20150618/1434628272)
		// range depends on the research of Y. Bengio et al. (2010)
		srand((unsigned)(time(0)));
		double range = sqrt(6)/sqrt(CATS_SIZE(0)+CATS_SIZE(1)+2);
		srand((unsigned)(time(0)));
		for (int i=0; i<this->layers-1; i++) {
			for (int j=0; j<(CATS_SIZE(i)+1)*CATS_SIZE(i+1); j++) {
				this->w[i][j] = 2.0*range*rand()/RAND_MAX-range;
			}
		}
	}
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	// delete arrays
	for (int i=0; i<this->layers-1; i++) free(this->z[i]);
	free(this->z);
	for (int i=0; i<this->layers; i++) free(this->o[i]);
	free(this->o);
	for (int i=0; i<this->layers-1; i++) free(this->d[i]);
	free(this->d);
	free(this->e2);
	free(this->e3);
	free(this->m2);
	free(this->m3);
	free(this->v2);
	free(this->v3);
	for (int i=0; i<this->layers-1; i++) free(this->w[i]);
	free(this->w);
	free(this->u);
}

// caluculate forward propagation of input x
void CatsEye_forward(CatsEye *this, double *x)
{
	// calculation of input layer
	memcpy(this->o[0], x, CATS_SIZE(0)*sizeof(double));
	this->o[0][CATS_SIZE(0)] = 1;	// for bias
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<CATS_SIZE(0); i++) {
		this->o[0][i] *= binomial(/*0.8(20%)*//*0.2*/0.5);
	}
#endif

	// caluculation of hidden and output layer [z = wx, o = f(z)]
	// z[hidden] += w[in * hidden] * o[0][in]
	// o[1][hidden] = act(z[hidden])
	CatsEye_layer_forward(this->o[0], this->w[0], this->z[0], this->o[1], &this->u[CATS_LPARAM*1]);
	for (int i=1; i<this->layers-1; i++) {
		this->o[i][CATS_SIZE(i)] = 1;	// for bias
		CatsEye_layer_forward(this->o[i], this->w[i], this->z[i], this->o[i+1], &this->u[CATS_LPARAM*(i+1)]);
	}
}

/* train: multi layer perceptron
 * x: train data (number of elements is in*N)
 * t: correct label (number of elements is N)
 * N: data size
 * repeat: repeat times
 * eta: learning rate (1e-6 to 1) */
void CatsEye_train(CatsEye *this, double *x, void *t, int N, int repeat, double eta)
{
	for (int times=0; times<repeat; times++) {
		double err = 0;
#ifndef CATS_OPT_SGD
		memset(this->e3, 0, sizeof(double)*CATS_SIZE(2));
		memset(this->e2, 0, sizeof(double)*CATS_SIZE(1));
#endif
#ifndef CATS_RANDOM
		for (int sample=0; sample<N; sample++) {
#else
		for (int n=0; n<1500/*N*/; n++) {
			int sample = (rand()/(RAND_MAX+1.0)) * N;	// 0 <= rand < 1
#endif
			// forward propagation
			CatsEye_forward(this, x+sample*CATS_SIZE(0));

			// calculate the error of output layer
			int a = this->layers-1;
			for (int i=0; i<CATS_SIZE(a); i++) {
#ifndef CATS_LOSS_MSE
				// http://d.hatena.ne.jp/echizen_tm/20110606/1307378609
				// E = max(0, -twx), ∂E / ∂w = max(0, -tx)
				if (((int*)t)[sample] == i) {	// 1-of-K
					this->d[a-1][i] = this->o[a][i]-1;
				} else {
					this->d[a-1][i] = this->o[a][i];
				}
#else
				// http://qiita.com/Ugo-Nama/items/04814a13c9ea84978a4c
				// https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88
				this->d[a-1][i] = this->o[a][i]-((double*)t)[sample*CATS_SIZE(a)+i];
//				this->d[1][i] = (this->o[2][i]-((double*)t)[sample*CATS_SIZE(2)+i]) * DACT2(this->o[2][i]);
#endif
//				this->e3[i] += this->d[1][i]*this->d[1][i];
				OPT_CALC1(3);
			}

			for (int i=this->layers-2; i>0; i--) {
				// calculate the error of hidden layer
				// t[hidden] += w[1][hidden * out] * d[1][out]
				// d[hidden] = t[hidden] * dact(o[hidden])
				CatsEye_layer_backward(this->o[i], this->w[i], this->d[i-1], this->d[i], &this->u[CATS_LPARAM*i]);
//			}
//			for (int i=this->layers-2; i>0; i--) {
				// update the weights of hidden layer
				// w[0][in] -= eta * o[0][in] * d[0][in * hidden]
				CatsEye_layer_update(eta, this->o[i-1], this->w[i-1], this->d[i-1], &this->u[CATS_LPARAM*i]);
			}

			// update the weights of output layer
			CatsEye_layer_update(eta, this->o[a-1], this->w[a-1], this->d[a-1], &this->u[CATS_LPARAM*a]);
#ifdef CATS_AUTOENCODER
			// tied weight
			double *dst = this->w[1];
			for (int i=0; i<CATS_SIZE(1); i++) {
				for (int j=0; j<CATS_SIZE(0); j++) {
					this->w[1][j + CATS_SIZE(1)*i] = this->w[0][CATS_SIZE(1)*j + i];
				}
			}
#endif

			// calculate the mean squared error
			double mse = 0;
			for (int i=0; i<CATS_SIZE(2); i++) {
				mse += 0.5 * (this->d[1][i] * this->d[1][i]);
			}
			err = 0.5 * (err + mse);

/*			if (isnan(this->d[1][0])) {
				printf("epochs %d, samples %d, mse %f\n", times, sample, err);
				break;
			}*/
		}
		printf("epochs %d, mse %f\n", times, err);
	}
}

// return most probable label to the input x
int CatsEye_predict(CatsEye *this, double *x)
{
	// forward propagation
	CatsEye_forward(this, x);

	// biggest output means most probable label
	int a = this->layers-1;
	double max = this->o[a][0];
	int ans = 0;
	for (int i=1; i<CATS_SIZE(a); i++) {
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

	fprintf(fp, "%d %d %d\n", CATS_SIZE(0), CATS_SIZE(1), CATS_SIZE(2));

	int i;
	for (i=0; i<(CATS_SIZE(0)+1)*CATS_SIZE(1)-1; i++) {
		fprintf(fp, "%lf ", this->w[0][i]);
	}
	fprintf(fp, "%lf\n", this->w[0][i]);

	for (i=0; i<(CATS_SIZE(1)+1)*CATS_SIZE(2)-1; i++) {
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

	fprintf(fp, "var config = [%d,%d,%d];\n", CATS_SIZE(0), CATS_SIZE(1), CATS_SIZE(2));

	int i;
	fprintf(fp, "var w1 = [");
	for (i=0; i<(CATS_SIZE(0)+1)*CATS_SIZE(1)-1; i++) {
		fprintf(fp, "%lf,", this->w[0][i]);
	}
	fprintf(fp, "%lf];\n", this->w[0][i]);

	fprintf(fp, "var w2 = [");
	for (i=0; i<(CATS_SIZE(1)+1)*CATS_SIZE(2)-1; i++) {
		fprintf(fp, "%lf,", this->w[1][i]);
	}
	fprintf(fp, "%lf];\n", this->w[1][i]);

	fclose(fp);
	return 0;
}

// save weights to binary file
int CatsEye_saveBin(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp==NULL) return -1;

	fwrite(&CATS_SIZE(0), sizeof(int), 1, fp);
	fwrite(&CATS_SIZE(1), sizeof(int), 1, fp);
	fwrite(&CATS_SIZE(2), sizeof(int), 1, fp);

	//fwrite(this->w[0], sizeof(double)*(CATS_SIZE(0)+1)*CATS_SIZE(1), 1, fp);
	//fwrite(this->w[1], sizeof(double)*(CATS_SIZE(1)+1)*CATS_SIZE(2), 1, fp);
	for (int i=0; i<(CATS_SIZE(0)+1)*CATS_SIZE(1); i++) {
		float a = this->w[0][i];
		fwrite(&a, sizeof(float), 1, fp);
	}
	for (int i=0; i<(CATS_SIZE(1)+1)*CATS_SIZE(2); i++) {
		float a = this->w[1][i];
		fwrite(&a, sizeof(float), 1, fp);
	}

	fclose(fp);
	return 0;
}

// w1
void CatsEye_visualizeWeights(CatsEye *this, int n, int size, unsigned char *p, int width)
{
	double *w = &this->w[0][n];
	double max = w[0];
	double min = w[0];
	for (int i=1; i<CATS_SIZE(0); i++) {
		if (max < w[i * CATS_SIZE(1)]) max = w[i * CATS_SIZE(1)];
		if (min > w[i * CATS_SIZE(1)]) min = w[i * CATS_SIZE(1)];
	}
	for (int i=0; i<CATS_SIZE(0); i++) {
		p[(i/size)*width + i%size] = ((w[i * CATS_SIZE(1)] - min) / (max - min)) * 255.0;
	}
}

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

#define CATS_SIGMOID
//#define CATS_TANH
//#define CATS_SCALEDTANH
//#define CATS_RELU
//#define CATS_ABS

// activation function and derivative of activation function
#ifdef CATS_SIGMOID
// sigmoid function
#define ACTIVATION_FUNCTION(x)		(1.0 / (1.0 + exp(-x)))
#define DACTIVATION_FUNCTION(x)		((1.0-x)*x)	// ((1.0-sigmod(x))*sigmod(x))
#elif defined CATS_TANH
// tanh function
#define ACTIVATION_FUNCTION(x)		(tanh(x))
#define DACTIVATION_FUNCTION(x)		(1.0 - x*x)	// (1.0 - tanh(x)*tanh(x))
#elif defined CATS_SCALEDTANH
// scaled tanh function
#define ACTIVATION_FUNCTION(x)		(1.7159 * tanh(2.0/3.0 * x))
#define DACTIVATION_FUNCTION(x)		((2.0/3.0)/1.7159 * (1.7159-x)*(1.7159+x))
#elif defined CATS_RELU
// rectified linear unit function
#define ACTIVATION_FUNCTION(x)		(x>0 ? x : 0.0)
#define DACTIVATION_FUNCTION(x)		(x>0 ? 1.0 : 0.0)
#elif defined CATS_ABS
// abs function
#define ACTIVATION_FUNCTION(x)		(x / (1.0 + fabs(x)))
#define DACTIVATION_FUNCTION(x)		(1.0 / (1.0 + fabs(x))*(1.0 + fabs(x)))
#else
// identity function (output only)
#define ACTIVATION_FUNCTION(x)		(x)
#define DACTIVATION_FUNCTION(x)		(1.0)
#endif

typedef struct {
	// number of each layer
	int in, hid, out;
	// input layer
	double *xi2, *xi3;
	// output layer
	double *o1, *o2, *o3;
	// error value
	double *d2, *d3;
	// gradient value
	double *e2, *e3;
	// weights
	double *w1, *w2;
} CatsEye;

/* constructor
 * n_in:  number of input layer
 * n_hid: number of hidden layer
 * n_out: number of output layer */
void CatsEye__construct(CatsEye *this, int n_in, int n_hid, int n_out, char *filename)
{
	FILE *fp;
	if (filename) {
		fp = fopen(filename, "r");
		if (fp==NULL) return;
		fscanf(fp, "%d %d %d\n", &this->in, &this->hid, &this->out);
	} else {
		// unit number
		this->in = n_in;
		this->hid = n_hid;
		this->out = n_out;
	}

	// allocate inputs
	this->xi2 = malloc(sizeof(double)*(this->hid+1));
	this->xi3 = malloc(sizeof(double)*this->out);

	// allocate outputs
	this->o1 = malloc(sizeof(double)*(this->in+1));
	this->o2 = malloc(sizeof(double)*(this->hid+1));
	this->o3 = malloc(sizeof(double)*this->out);

	// allocate errors
	this->d2 = malloc(sizeof(double)*(this->hid+1));
	this->d3 = malloc(sizeof(double)*this->out);

	// allocate gradient
	this->e2 = calloc(1, sizeof(double)*(this->hid+1));
	this->e3 = calloc(1, sizeof(double)*this->out);

	// allocate memories
	this->w1 = malloc(sizeof(double)*(this->in+1)*this->hid);
	this->w2 = malloc(sizeof(double)*(this->hid+1)*this->out);

	if (filename) {
		for(int i=0; i<(this->in+1)*this->hid; i++) {
			fscanf(fp, "%lf ", &this->w1[i]);
		}
		for(int i=0; i<(this->hid+1)*this->out; i++) {
			fscanf(fp, "%lf ", &this->w2[i]);
		}
		fclose(fp);
	} else {
		// initialize weights
		// range depends on the research of Y. Bengio et al. (2010)
		srand((unsigned)(time(0)));
		double range = sqrt(6)/sqrt(this->in+this->hid+2);
		srand((unsigned)(time(0)));
		for (int i=0; i<(this->in+1)*this->hid; i++) {
			this->w1[i] = 2.0*range*rand()/RAND_MAX-range;
		}
		for (int i=0; i<(this->hid+1)*this->out; i++) {
			this->w2[i] = 2.0*range*rand()/RAND_MAX-range;
		}
	}
}

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	// delete arrays
	free(this->xi2);
	free(this->xi3);
	free(this->o1);
	free(this->o2);
	free(this->o3);
	free(this->d2);
	free(this->d3);
	free(this->e2);
	free(this->e3);
	free(this->w1);
	free(this->w2);
}

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

// caluculate forward propagation of input x
void CatsEye_forward(CatsEye *this, double *x)
{
	// calculation of input layer
//	memcpy(this->xi1, x, this->in*sizeof(double));
	memcpy(this->o1, x, this->in*sizeof(double));
	this->o1[this->in] = 1;

	// caluculation of hidden layer
//	#pragma omp parallel for
	for (int j=0; j<this->hid; j++) {
/*		this->xi2[j] = 0;
		for (int i=0; i<this->in+1; i++) {
			this->xi2[j] += this->w1[i*this->hid+j]*this->o1[i];
		}*/
		this->xi2[j] = dotT(&this->w1[j], this->o1, this->in+1, this->hid);
		this->o2[j] = ACTIVATION_FUNCTION(this->xi2[j]);
	}
	this->o2[this->hid] = 1;

	// caluculation of output layer
	for (int j=0; j<this->out; j++) {
/*		this->xi3[j] = 0;
		for (int i=0; i<this->hid+1; i++) {
			this->xi3[j] += this->w2[i*this->out+j]*this->o2[i];
		}*/
		this->xi3[j] = dotT(&this->w2[j], this->o2, this->hid+1, this->out);
		this->o3[j] = this->xi3[j];
	}
}

/* train: multi layer perceptron
 * x: train data (number of elements is in*N)
 * t: correct label (number of elements is N)
 * N: data size
 * repeat: repeat times
 * eta: learning rate */
void CatsEye_train(CatsEye *this, double *x, void *t, int N, int repeat/*=1000*/, double eta/*=0.1*/)
{
	for (int times=0; times<repeat; times++) {
		double err = 0;
#ifndef CATS_MINI_BATCH
		for (int sample=0; sample<N; sample++) {
			// forward propagation
			CatsEye_forward(this, x+sample*this->in);
#else
		for (int sample=0; sample<100; sample++) {
			// forward propagation
			srand((unsigned)time(NULL));
			int n = ((double)rand()+1.0)/((double)RAND_MAX+2.0) * N;
			CatsEye_forward(this, x+n*this->in);
#endif

			// calculate the error of output layer
			for (int i=0; i<this->out; i++) {
#ifndef CATS_LOSS_MSE
				if (((int*)t)[sample] == i) {	// classifying
					this->d3[i] = this->o3[i]-1;
				} else {
					this->d3[i] = this->o3[i];
				}
#else
				this->d3[i] = this->o3[i]-((double*)t)[sample*this->out+i];
#endif
			}
			// update the weights of output layer
//			#pragma omp parallel for
#ifdef CATS_ADAGRAD
			// AdaGrad (http://qiita.com/ak11/items/7f63a1198c345a138150)
			for (int j=0; j<this->out; j++) {
//				this->e3[j] += this->d3[j]*this->d3[j]*0.7;
				this->e3[j] = this->e3[j]*(0.99+times/times*0.01) + this->d3[j]*this->d3[j];
			}
			for (int i=0; i<this->hid+1; i++) {
				for (int j=0; j<this->out; j++) {
					//this->w2[i*this->out+j] -= eta*this->d3[j]*this->o2[i];
					// AdaGrad (http://qiita.com/ak11/items/7f63a1198c345a138150)
//					this->e3[j] += this->d3[j]*this->d3[j];
					//this->w2[i*this->out+j] -= eta*this->o2[i] /sqrt(this->e3[j]) *this->d3[j];
					this->w2[i*this->out+j] -= eta*this->o2[i] /sqrt(1.0+this->e3[j]) *this->d3[j];
				}
			}
#else
			for (int i=0; i<this->hid+1; i++) {
				muladd(&this->w2[i*this->out], this->d3, -eta*this->o2[i], this->out);
			}
#endif
			// calculate the error of hidden layer
			for (int i=0; i<this->hid+1; i++) {
/*				double tmp = 0;
				for (int l=0; l<this->out; l++) {
					tmp += this->w2[j*this->out+l]*this->d3[l];
				}*/
				//this->d2[j] = tmp * DACTIVATION_FUNCTION(this->xi2[j]);	// xi2 = z
				this->d2[i] = dot(&this->w2[i*this->out], this->d3, this->out) * DACTIVATION_FUNCTION(this->o2[i]);	// o2 = f(z)
			}
			// update the weights of hidden layer
#ifdef CATS_ADAGRAD
			for (int j=0; j<this->hid; j++) {
//				this->e2[j] += this->d2[j]*this->d2[j]*0.7;
				this->e2[j] = this->e2[j]*(0.99+times/times*0.01) + this->d2[j]*this->d2[j];
			}
			for (int i=0; i<this->in+1; i++) {
				for (int j=0; j<this->hid; j++) {
					//this->w1[i*this->hid+j] -= eta*this->d2[j]*this->o1[i];
					// AdaGrad
//					this->e2[j] += this->d2[j]*this->d2[j];
					//this->w1[i*this->hid+j] -= eta*this->o1[i] /sqrt(this->e2[j]) *this->d2[j];
					this->w1[i*this->hid+j] -= eta*this->o1[i] /sqrt(1.0+this->e2[j]) *this->d2[j];
				}
			}
#else
			for (int i=0; i<this->in+1; i++) {
				muladd(&this->w1[i*this->hid], this->d2, -eta*this->o1[i], this->hid);
			}
#endif

			// calculate the mean squared error
			double mse = 0;
			for (int i=0; i<this->out; i++) {
				mse += 0.5 * (this->d3[i] * this->d3[i]);
			}
			err = 0.5 * (err + mse);

/*			if (isnan(this->d3[0])) {
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
	double max = this->o3[0];
	int ans = 0;
	for (int i=1; i<this->out; i++) {
		if (this->o3[i] > max) {
			max = this->o3[i];
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

	fprintf(fp, "%d %d %d\n", this->in, this->hid, this->out);

	int i;
	for (i=0; i<(this->in+1)*this->hid-1; i++) {
		fprintf(fp, "%lf ", this->w1[i]);
	}
	fprintf(fp, "%lf\n", this->w1[i]);

	for (i=0; i<(this->hid+1)*this->out-1; i++) {
		fprintf(fp, "%lf ", this->w2[i]);
	}
	fprintf(fp, "%lf\n", this->w2[i]);

	fclose(fp);
	return 0;
}

// save weights to json file
int CatsEye_saveJson(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "w");
	if (fp==NULL) return -1;

	fprintf(fp, "var config = [%d,%d,%d];\n", this->in, this->hid, this->out);

	int i;
	fprintf(fp, "var w1 = [");
	for (i=0; i<(this->in+1)*this->hid-1; i++) {
		fprintf(fp, "%lf,", this->w1[i]);
	}
	fprintf(fp, "%lf];\n", this->w1[i]);

	fprintf(fp, "var w2 = [");
	for (i=0; i<(this->hid+1)*this->out-1; i++) {
		fprintf(fp, "%lf,", this->w2[i]);
	}
	fprintf(fp, "%lf];\n", this->w2[i]);

	fclose(fp);
	return 0;
}

// save weights to binary file
int CatsEye_saveBin(CatsEye *this, char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp==NULL) return -1;

	fwrite(&this->in, sizeof(this->in), 1, fp);
	fwrite(&this->hid, sizeof(this->hid), 1, fp);
	fwrite(&this->out, sizeof(this->out), 1, fp);

	//fwrite(this->w1, sizeof(double)*(this->in+1)*this->hid, 1, fp);
	//fwrite(this->w2, sizeof(double)*(this->hid+1)*this->out, 1, fp);
	for (int i=0; i<(this->in+1)*this->hid; i++) {
		float a = this->w1[i];
		fwrite(&a, sizeof(float), 1, fp);
	}
	for (int i=0; i<(this->hid+1)*this->out; i++) {
		float a = this->w2[i];
		fwrite(&a, sizeof(float), 1, fp);
	}

	fclose(fp);
	return 0;
}

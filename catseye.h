//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include <stdlib.h>
#include <time.h>
#include <math.h>

// sigmoid function
double sigmoid(double x)
{
	return 1/(1+exp(-x));
}

// derivative of sigmoid function
double d_sigmoid(double x)
{
	double a = sigmoid(x);
	return (1-a)*a;
}

typedef struct {
	// number of each layer
	int in, hid, out;
	// input layer
	double *xi1, *xi2, *xi3;
	// output layer
	double *o1, *o2, *o3;
	// error value
	double *d2, *d3;
	// wait
	double *w1, *w2;
} CatsEye;

/* constructor
 * n_in:  number of input layer
 * n_hid: number of hidden layer
 * n_out: number of output layer */
void CatsEye__construct(CatsEye *this, int n_in, int n_hid, int n_out)
{
	// unit number
	this->in = n_in;
	this->hid = n_hid;
	this->out = n_out;

	// allocate inputs
	this->xi1 = malloc(sizeof(double)*(this->in+1));
	this->xi2 = malloc(sizeof(double)*(this->hid+1));
	this->xi3 = malloc(sizeof(double)*this->out);

	// allocate outputs
	this->o1 = malloc(sizeof(double)*(this->in+1));
	this->o2 = malloc(sizeof(double)*(this->hid+1));
	this->o3 = malloc(sizeof(double)*this->out);

	// allocate errors
	this->d2 = malloc(sizeof(double)*(this->hid+1));
	this->d3 = malloc(sizeof(double)*this->out);

	// allocate memories
	this->w1 = malloc(sizeof(double)*(this->in+1)*this->hid);
	this->w2 = malloc(sizeof(double)*(this->hid+1)*this->out);

	// initialize wait
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

// deconstructor
void CatsEye__destruct(CatsEye *this)
{
	// delete arrays
	free(this->xi1);
	free(this->xi2);
	free(this->xi3);
	free(this->o1);
	free(this->o2);
	free(this->o3);
	free(this->d2);
	free(this->d3);
	free(this->w1);
	free(this->w2);
}

// caluculate forward propagation of input x
void CatsEye_forward(CatsEye *this, double *x)
{
	// calculation of input layer
	for (int j=0; j<this->in; j++) {
		this->xi1[j] = x[j];
		this->xi1[this->in] = 1;
		this->o1[j] = this->xi1[j];
	}
	this->o1[this->in] = 1;

	// caluculation of hidden layer
	for (int j=0; j<this->hid; j++) {
		this->xi2[j] = 0;
		for (int i=0; i<this->in+1; i++) {
			this->xi2[j] += this->w1[i*this->hid+j]*this->o1[i];
		}
		this->o2[j] = sigmoid(this->xi2[j]);
	}
	this->o2[this->hid] = 1;

	// caluculation of output layer
	for (int j=0; j<this->out; j++) {
		this->xi3[j] = 0;
		for (int i=0; i<this->hid+1; i++) {
			this->xi3[j] += this->w2[i*this->out+j]*this->o2[i];
		}
		this->o3[j] = this->xi3[j];
	}
}

/* train: multi layer perceptron
 * x: train data(number of elements is in*N)
 * t: correct label(number of elements is N)
 * N: data size
 * repeat: repeat times
 * eta: learning rate */
void CatsEye_train(CatsEye *this, double *x, int *t, double N, int repeat/*=1000*/, double eta/*=0.1*/)
{
	for (int times=0; times<repeat; times++) {
		for (int sample=0; sample<N; sample++) {
			// forward propagation
			CatsEye_forward(this, x+sample*this->in);

			// calculate the error of output layer
			for (int j=0; j<this->out; j++) {
				if (t[sample] == j) {
					this->d3[j] = this->o3[j]-1;
				} else {
					this->d3[j] = this->o3[j];
				}
			}
			// update the wait of output layer
			for (int i=0; i<this->hid+1; i++) {
				for (int j=0; j<this->out; j++) {
					this->w2[i*this->out+j] -= eta*this->d3[j]*this->o2[i];
				}
			}
			// calculate the error of hidden layer
			for (int j=0; j<this->hid+1; j++) {
				double tmp = 0;
				for (int l=0; l<this->out; l++) {
					tmp += this->w2[j*this->out+l]*this->d3[l];
				}
				this->d2[j] = tmp * d_sigmoid(this->xi2[j]);
			}
			// update the wait of hidden layer
			for (int i=0; i<this->in+1; i++) {
				for (int j=0; j<this->hid; j++) {
					this->w1[i*this->hid+j] -= eta*this->d2[j]*this->o1[i];
				}
			}
		}
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

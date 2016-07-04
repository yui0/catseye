//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

char kernel_code[] =
#include "catseye.cl"

cl_mem d_mem[5];
unsigned int param[8];
args_t args[] = {
	{ CL_MEM_READ_WRITE, 0, &d_mem[0], 0, -1, 0 },	// x
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 1 },	// w
	{ CL_MEM_READ_WRITE, 0, &d_mem[2], 0, -1, 1 },	// o
	{ CL_MEM_READ_WRITE, 0, &d_mem[3], 0, 1, 1 },	// d
	{ CL_MEM_READ_WRITE, 0, &d_mem[4], 0, -1, 0 },	// t
	{ 0, sizeof(param), &param, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "forward",	0, {256,0,0,},{256,0,0,}, args },
//	{ "train",	0, {256,0,0,},{256,0,0,}, args },
	{ "train",	0, {1024,0,0,},{0,0,0,}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

void CatsEye_clSetup(CatsEye *this)
{
	args[0].size = sizeof(numerus)*(SIZE(0)+1)*60000;
	//args[0].s = this->xdata;
	args[1].size = sizeof(numerus)*this->wsize;
	args[1].s = this->wdata;
	args[2].size = sizeof(numerus)*this->osize;
	args[2].s = this->odata;
	args[3].size = sizeof(numerus)*this->dsize;
	args[3].s = this->ddata;
	args[4].size = sizeof(numerus)*60000;
	//args[4].s = this->ddata;
//	printf("%d %d %d\n", this->u[SIZE], this->wsize, this->osize);
//	printf("%d %d %d\n", in, hid, out);

	// http://dhruba.name/2012/12/24/opencl-cookbook-10-tips-for-high-performance-kernels/
	oclSetup(0, 0);
	oclKernel(kernel, ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", kernel_code);
}


void CatsEye_clFinish()
{
	oclReleaseKernel(kernel, ksz);
	oclFinish();
}

void CatsEye_forward(CatsEye *this, numerus *x, int n)
{
#if 0
	// calculation of input layer
	memcpy(this->o[0], x+n, SIZE(0)*sizeof(numerus));
	this->o[0][SIZE(0)] = 1;	// for bias
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<SIZE(0); i++) {
		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
	}
#endif
#endif


	if (n<0) { n = x-1 - this->xdata; x = this->xdata; }
	args[0].s = this->xdata;
	//args[0].size = sizeof(numerus)*(SIZE(0)+1)*60000;
	param[5] = n;

	oclKernelArgsWrite(args);
	oclRun(&kernel[0]);
	oclKernelArgsRead(args);

	memcpy(this->o[0], x+n, SIZE(0)*sizeof(numerus));


/*	for (int i=0; i<200; i++) printf("%f ", this->o[1][i]);
	printf("\n%d %f\n",SIZE(0),this->o[0][0]);
	CatsEye_layer_forward[TYPE(1)](this->o[0], this->w[0], this->z[0], this->o[1], &this->u[LPLEN*(1)]);
	for (int i=0; i<200; i++) printf("%f ", this->o[1][i]);
	printf("\n");
	exit(0);*/
/*	for (int i=0; i<10; i++) printf("%f ", this->o[2][i]);
	printf("\n%d %f\n",SIZE(0),this->o[1][0]);
	CatsEye_layer_forward[TYPE(2)](this->o[1], this->w[1], this->z[1], this->o[2], &this->u[LPLEN*(2)]);
	for (int i=0; i<10; i++) printf("%f ", this->o[2][i]);
	printf("\n");
	exit(0);*/
}

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
		for (int n=0; n<batch; n++) {
			int sample = RANDOM ? (frand()*N) : n;

	args[0].s = this->xdata;
	args[4].s = t;
	param[0] = sample;
	param[1] = eta;
	param[5] = sample*SIZE(0);
	oclKernelArgsWrite(args);
	oclRun(&kernel[1]);
	oclKernelArgsRead(args);
	memcpy(this->o[0], x+param[5], SIZE(0)*sizeof(numerus));
	this->o[0][SIZE(0)] = 1;

			// forward propagation
//			CatsEye_forward(this, x, sample*SIZE(0));

			// calculate the error of output layer
//			CatsEye_loss_0_1(this, a, t, sample);

			// calculate the error of hidden layer
			// t[hidden] += w[1][hidden * out] * d[1][out]
			// d[hidden] = t[hidden] * dact(o[hidden])
/*			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_backward[TYPE(i+1)](this->o[i], this->w[i], this->d[i-1], this->d[i], &this->u[LPLEN*(i+1)]);
			}*/
			// update the weights of hidden layer
			// w[0][in] -= eta * o[0][in] * d[0][in * hidden]
/*			for (int i=this->layers-2; i>0; i--) {
				CatsEye_layer_update[TYPE(i)](eta, this->o[i-1], this->w[i-1], this->d[i-1], &this->u[LPLEN*i]);
			}
		for (int n=784*200; n<784*200+100; n++) {
			printf("%f ", this->w[0][n]);
		}
		printf("\n");*/

			// update the weights of output layer
//			CatsEye_layer_update[TYPE(a)](eta, this->o[a-1], this->w[a-1], this->d[a-1], &this->u[LPLEN*a]);
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

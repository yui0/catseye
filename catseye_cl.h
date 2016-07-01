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
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 0 },	// w
	{ CL_MEM_READ_WRITE, 0, &d_mem[2], 0, -1, 1 },	// o
	{ CL_MEM_READ_WRITE, 0, &d_mem[3], 0, -1, 0 },	// d
	{ CL_MEM_READ_WRITE, 0, &d_mem[4], 0, -1, 0 },	// t
	{ 0, sizeof(param), &param, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "forward",	0, {0,0,0,}, args },
	{ "train",	0, {0,0,0,}, args },
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
	//args[4].s = this->ddata;//FIXME
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

	kernel[0].global_size[0] = 1024;
	oclRun(&kernel[0]);

	oclKernelArgsRead(args);

//	memcpy(this->o[0], x+n, (SIZE(0)+1)*sizeof(numerus));
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

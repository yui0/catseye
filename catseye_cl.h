//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

char kernel_code[] =
#include "catseye.cl"

cl_mem d_mem[2];
int in, hid, out;
unsigned int param[8];
args_t args[] = {
	{ CL_MEM_READ_WRITE, 0, &d_mem[0], 0, 1, 1 },	// x, o
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 0 },	// w
	{ 0, sizeof(param), &param, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "linear_forward_identity", 0, {0,0,0,}, args },
	{ "linear_forward_softmax", 0, {0,0,0,}, args },
	{ "linear_forward_sigmoid", 0, {0,0,0,}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

void CatsEye_clSetup(CatsEye *this)
{
	args[0].size = sizeof(numerus)*this->osize;
	args[0].s = this->odata;
	args[1].size = sizeof(numerus)*this->wsize;
	args[1].s = this->wdata;
	in = this->u[SIZE];
	hid = this->u[SIZE+LPLEN];
	out = this->u[SIZE+LPLEN*2];
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

void CatsEye_clForward(CatsEye *this)
{
	oclKernelArgsWrite(args);

/*	param[3] = param[4] = 0;
	for (int i=0; i<this->layers-1; i++) {
		int *u = &this->u[LPLEN*i];
		int act = u[TYPE+LPLEN];
		param[0] = u[SIZE];		// in
		param[1] = u[SIZE+LPLEN];	// out
		param[2] = param[4];		// xoff
		param[4] += u[SIZE]+1;		// ooff
		kernel[act].global_size[0] = u[SIZE];
		oclRun(&kernel[act]);
printf("%d %d %d %d %d [%d]\n",param[0],param[1],param[2],param[3],param[4],act);
		param[3] = this->ws[i];	// woff
	}*/

	param[0] = in;			// in
	param[1] = hid;			// out
	param[2] = 0;			// xoff
	param[3] = 0;			// woff
	param[4] = this->u[SIZE]+1;	// ooff
	kernel[2].global_size[0] = hid;
	oclRun(&kernel[2]);
	param[0] = hid;			// in
	param[1] = out;			// out
	param[2] = this->u[SIZE]+1;	// xoff
	param[3] = (in+1) * hid;	// woff
	param[4] = in+1+hid+1;		// ooff
	kernel[0].global_size[0] = out;
	oclRun(&kernel[0]);

	oclKernelArgsRead(args);
}

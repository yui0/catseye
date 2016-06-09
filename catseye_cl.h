//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

char kernel_code[] =
#include "catseye.cl"

cl_mem d_mem[3];
int in, hid, out;
args_t args[] = {
	{ CL_MEM_READ_WRITE, 0, &d_mem[0], 0, 1, 0 },	// x
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 0 },	// w
	{ CL_MEM_READ_WRITE, 0, &d_mem[2], 0, 0, 1 },	// o
	{ 0, sizeof(int), &in, 0, 0, 0 },
	{ 0, sizeof(int), &hid, 0, 0, 0 },
//	{ 0, sizeof(int), &out, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "gemv1_act", 0, args },
	{ "gemv1", 0, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

void CatsEye_clSetup(CatsEye *this)
{
	args[0].size = sizeof(numerus)*(this->u[SIZE]+1);
	args[0].s = this->odata;
	args[1].size = sizeof(numerus)*this->wsize;
	args[1].s = this->wdata;
	args[2].size = sizeof(numerus)*(this->osize - (this->u[SIZE]+1));
	args[2].s = this->o[1];
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

void CatsEye_clForward(CatsEye *this, numerus *x)
{
	int N = 200;
	int lsz = 256;
	int gsz = ((N + lsz - 1) / lsz) * lsz;
	size_t local_item_size[] = { lsz };
	size_t global_item_size[] = { gsz };

	oclKernelArgsWrite(args);
	oclRun(&kernel[0], 1, global_item_size, local_item_size);
	oclKernelArgsRead(args);
}

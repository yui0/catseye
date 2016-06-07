//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

#define kernel_name	"forward"
char kernel_code[] =
#include "catseye.cl"

cl_kernel kernel;
cl_mem d_mem[3];
int in, hid, out;
args_t args[] = {
	{ CL_MEM_READ_WRITE/* | CL_MEM_ALLOC_HOST_PTR*/, 0, &d_mem[0], 0, 1, 0 },
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 0 },
	{ CL_MEM_READ_WRITE, 0, &d_mem[2], 0, /*1*/0, 1 },
	{ 0, sizeof(int), &in, 0, 0, 0 },
	{ 0, sizeof(int), &hid, 0, 0, 0 },
	{ 0, sizeof(int), &out, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};

void CatsEye_clSetup(CatsEye *this)
{
	args[0].size = sizeof(numerus)*(this->u[SIZE]+1);
//	args[0].s = this->odata;
	args[1].size = sizeof(numerus)*this->wsize;
	args[1].s = this->wdata;
	args[2].size = sizeof(numerus)*this->osize;
	args[2].s = this->odata;
//	args[2].size = sizeof(numerus)*(this->osize - (this->u[SIZE]+1));
//	args[2].s = &this->o[this->u[SIZE]+1];
	in = this->u[SIZE];
	hid = this->u[SIZE+LPLEN];
	out = this->u[SIZE+LPLEN*2];
	printf("%d %d %d\n", this->u[SIZE], this->wsize, this->osize);
	printf("%d %d %d\n", in, hid, out);

	oclSetup(0, 0);
	kernel = oclKernel(kernel_name, /*"-cl-fast-relaxed-math -Werror"*/0, kernel_code, args);
}
void CatsEye_clFinish()
{
	oclReleaseKernel(kernel, args);
	oclFinish();
}
void CatsEye_clForward(CatsEye *this, numerus *x)
{
	int lsz = 256;
	int gsz = (785*200/256*256+256);//256;//((N + lsz - 1) / lsz) * lsz;
	size_t local_item_size[] = { lsz };
	size_t global_item_size[] = { gsz };

	args[0].s = x;
/*	for (int i=0; i<785; i++) {
		if (x[i]>100) printf("<%d>%f ",i,x[i]);
		//printf("<%d>%f ",i,this->o[0][i]);
//		printf("<%d>%f ",i,((float*)args[2].s)[i]);
		printf("<%d>%f ",i,((float*)args[0].s)[i]);
//		printf("<%d>%f ",i,this->odata[i]);
	}*/

	oclKernelArgsWrite(args);
	oclRun(kernel, args, 1, global_item_size, local_item_size);
	oclKernelArgsRead(args);
}

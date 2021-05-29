//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define _STRGF(x)	# x
#define OCLSTRINGIFY(x)	_STRGF(x)

#define MAX_PLATFORMS	10
#define MAX_DEVICES	10

typedef struct {
	int type;
	int size;
	void *p;
	void *s;
	int write, read;
} args_t;

typedef struct {
	char *f;
	cl_kernel k;
	size_t global_size[3];
	size_t local_size[3];
	args_t *a;
} ocl_t;

int ocl_device;
cl_device_id device_id[MAX_DEVICES];
cl_context context;
cl_command_queue command_queue;

void oclSetup(int platform, int device)
{
	cl_platform_id platform_id[MAX_PLATFORMS];
	cl_uint num_devices;
	cl_uint num_platforms;
	cl_int ret;

	ocl_device = device;

	int type = CL_DEVICE_TYPE_ALL;
	if (getenv("FORCE_GPU")) {
		type = CL_DEVICE_TYPE_GPU;
	} else if (getenv("FORCE_CPU")) {
		type = CL_DEVICE_TYPE_CPU;
	} else if (getenv("FORCE_ACCELERATOR")) {
		type = CL_DEVICE_TYPE_ACCELERATOR;
	}

	ret = clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms);
	ret = clGetDeviceIDs(platform_id[platform], type, MAX_DEVICES, device_id, &num_devices);

	// device name (option)
	size_t size;
	char str[256];
	clGetDeviceInfo(device_id[device], CL_DEVICE_NAME, sizeof(str), str, &size);
	printf("%s (platform %d, device %d)\n", str, platform, device);

	context = clCreateContext(NULL, 1, &device_id[device], NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device_id[device], 0, &ret);

	/*cl_ulong maxMemAlloc;
	clGetDeviceInfo(device_id[device], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAlloc, NULL);
	printf("Maximum memory allocation size is %llu bytes\n", maxMemAlloc);*/
}

void oclKernel(ocl_t *kernel, int n, char *opt, char *kernel_code)
{
	cl_int ret;
	const char* src[1] = { kernel_code };

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&src, 0, &ret);
	ret = clBuildProgram(program, 1, &device_id[ocl_device], NULL, NULL, NULL);
	if (ret) {
		size_t len = 0;
		cl_int ret = CL_SUCCESS;
		ret = clGetProgramBuildInfo(program, device_id[ocl_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer = calloc(len, sizeof(char));
		ret = clGetProgramBuildInfo(program, device_id[ocl_device], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		printf("\n%s\n", buffer);
	}
	for (int i=0; i<n; i++) {
		kernel->k = clCreateKernel(program, kernel->f, &ret);

		if (!kernel->global_size[0]) {
			size_t *local = kernel->global_size;
			clGetKernelWorkGroupInfo(kernel->k, device_id[ocl_device], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t)*3, local, NULL);
			printf("CL_KERNEL_WORK_GROUP_SIZE: %zu\n", local[0]);
		}
		kernel++;
	}
	clReleaseProgram(program);
}

void oclKernelArgs(ocl_t *kernel, int n)
{
	cl_int ret;
	for (int i=0; i<n; i++) {
		args_t *args = kernel->a;
		while (args->size) {
			if (args->type>0) {
				cl_mem *p = args->p;
				if (!*p) *p = clCreateBuffer(context, args->type, args->size, NULL, &ret);
				if (!*p) printf("clCreateBuffer error!!\n");
			}
			args++;
		}
		kernel++;
	}
}

void oclKernelArgsWrite(args_t *args)
{
	while (args->size) {
		if (args->write) {
			clEnqueueWriteBuffer(command_queue, *(cl_mem*)(args->p), CL_TRUE, 0, args->size, args->s, 0, 0, 0);
			if (args->write<0) args->write++;
		}
		args++;
	}
}

void oclKernelArgsRead(args_t *args)
{
	while (args->size) {
		if (args->read) {
			clEnqueueReadBuffer(command_queue, *(cl_mem*)(args->p), CL_TRUE, 0, args->size, args->s, 0, 0, 0);
		}
		args++;
	}
}

void oclRun(ocl_t *kernel)
{
	int n = 0;
	args_t *args = kernel->a;
	while (args->size) {
		if (args->type>0) clSetKernelArg(kernel->k, n++, sizeof(cl_mem), (void*)args->p);
		else clSetKernelArg(kernel->k, n++, args->size, (void*)args->p);
		args++;
	}

	size_t *local = kernel->local_size[0] ? kernel->local_size : 0;
	clEnqueueNDRangeKernel(command_queue, kernel->k, 1, NULL, kernel->global_size, local, 0, NULL, NULL);
	//cl_event e;
	//clEnqueueNDRangeKernel(command_queue, kernel->k, 1, NULL, kernel->global_size, local, 0, NULL, &e);
}

void oclReleaseKernel(ocl_t *kernel, int n)
{
	for (int i=0; i<n; i++) {
		args_t *args = kernel->a;
		while (args->size) {
			if (args->type>0 && args->p) {
				clReleaseMemObject(*(cl_mem*)(args->p));
				*(cl_mem*)args->p = 0;
			}
			args++;
		}
		clReleaseKernel(kernel->k);
		kernel++;
	}
}

void oclWait()
{
	clFinish(command_queue);
}

void oclFinish()
{
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

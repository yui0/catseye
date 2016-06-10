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

	ret = clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms);
	ret = clGetDeviceIDs(platform_id[platform], CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_id, &num_devices);

	// device name (option)
	size_t size;
	char str[256];
	clGetDeviceInfo(device_id[device], CL_DEVICE_NAME, sizeof(str), str, &size);
	printf("%s (platform %d, device %d)\n", str, platform, device);

	context = clCreateContext(NULL, 1, &device_id[device], NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device_id[device], 0, &ret);
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

		args_t *args = kernel->a;
		while (args->size) {
			if (args->type>0) *(cl_mem*)(args->p) = clCreateBuffer(context, args->type, args->size, NULL, &ret);
			args++;
		}

		kernel++;
	}
	clReleaseProgram(program);
}

void oclKernelArgsWrite(args_t *args)
{
	while (args->size) {
		if (args->write) {
			clEnqueueWriteBuffer(command_queue, *(cl_mem*)(args->p), CL_TRUE, 0, args->size, args->s, 0, 0, 0);
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

/*	int r = */clEnqueueNDRangeKernel(command_queue, kernel->k, 1, NULL, kernel->global_size, 0, 0, NULL, NULL);
//	if (r<0) printf("Kernel error!! %d\n", r);
}

void oclReleaseKernel(ocl_t *kernel, int n)
{
	for (int i=0; i<n; i++) {
		args_t *args = kernel->a;
		while (args->size) {
			if (args->type>0 && args->p) {
				clReleaseMemObject(*(cl_mem*)(args->p));
				args->p = 0;
			}
			args++;
		}
		clReleaseKernel(kernel->k);
		kernel++;
	}
}

void oclFinish()
{
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

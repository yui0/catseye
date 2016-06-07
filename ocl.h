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

cl_kernel oclKernel(char *k, char *opt, char *kernel_code, args_t *args)
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
	cl_kernel kernel = clCreateKernel(program, k, &ret);
	clReleaseProgram(program);

	while (args->size) {
		if (args->type>0) *(cl_mem*)(args->p) = clCreateBuffer(context, args->type, args->size, NULL, &ret);
		args++;
	}

	return kernel;
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

void oclRun(cl_kernel kernel, args_t *args, int dim, size_t *global_work_size, size_t *local_work_size)
{
	int n = 0;
	while (args->size) {
		if (args->type>0) clSetKernelArg(kernel, n++, sizeof(cl_mem), (void*)args->p);
		else clSetKernelArg(kernel, n++, sizeof(int), (void*)args->p);
		args++;
	}

	clEnqueueNDRangeKernel(command_queue, kernel, dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

void oclReleaseKernel(cl_kernel kernel, args_t *args)
{
	while (args->size) {
		if (args->type>0) clReleaseMemObject(*(cl_mem*)(args->p));
		args++;
	}
	clReleaseKernel(kernel);
}

void oclFinish()
{
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

OCLSTRINGIFY(

#define identity(a)	(a)
#define softmax(a)	(a)	// FIXME
#define sigmoid(a)	(1.0f / (1 + exp(-a)))
#define normal_tanh(a)	(tanh(a))
#define scaled_tanh(a)	(1.7159f * tanh(0.66667f * a))
#define relu(a)		(a > 0 ? a : 0)
#define LeakyReLU(a)	(a > 0 ? a : a * 0.01)

#define LINEAR_FORWARD(act) \
__kernel void linear_forward_##act(__global const float *x, __global const float *a, __global float *y, uint8 pa)\
{\
	int gid = get_global_id(0);\
	if (gid < pa[1]) {\
		/*__global float *x = y + pa[2];*/\
		if (pa[2]) x = y + pa[2];\
		else x += pa[5];\
		a += pa[3];\
		y += pa[4];\
		float sum = 0;\
		for (int k=0; k<pa[0]; k++) {\
			sum += a[gid + pa[1]*k] * x[k];\
		}\
		sum += a[gid + pa[1]*pa[0]];\
		y[gid] = act(sum);\
	}\
}
LINEAR_FORWARD(identity);
LINEAR_FORWARD(softmax);	// FIXME
LINEAR_FORWARD(sigmoid);
LINEAR_FORWARD(normal_tanh);
LINEAR_FORWARD(scaled_tanh);
LINEAR_FORWARD(relu);
LINEAR_FORWARD(LeakyReLU);


void loss_0_1(__global const float *x, __global const float *a, __global float *y, uint8 pa)
{
}


__kernel void memset_uint4(__global uint4 *mem, __private uint4 val)
{
	mem[get_global_id(0)] = val;
}
__kernel void memset_float(__global float *mem, __private float val)
{
	mem[get_global_id(0)] = val;
}
// pa[0]: in
// pa[1]: out
__kernel void linear_forward(__global float *x, __global float *a, __global float *y, uint8 pa)
{
	int gid = get_global_id(0);
	if (gid <= pa[0]) {
		if (pa[2]) x = y + pa[2];
		else x += pa[5];
		a += pa[3];
		y += pa[4];
		int c = pa[0]*pa[1];
		for (int k=0; k<pa[1]; k++) {
			y[k] += a[k + c] * x[gid];
		}
	}
}



#define ROW_DIM 0
#define COL_DIM 1

// http://www.bealto.com/gpu-gemv_v3.html
// P threads per row compute 1/P-th of each dot product.
// WORK has N/P entries.
__kernel void gemv(__global const float *a, __global const float *x, __global float *y,
	__local float *work, int m, int n)
{
	// Load a slice of X in WORK, using all available threads
	int ncols = n / get_global_size(COL_DIM); // nb values to load
	int col0 = ncols * get_global_id(COL_DIM); // first value to load
	for (int k=0; k<ncols; k+=get_local_size(ROW_DIM)) {
		int col = k+get_local_id(ROW_DIM);
		if (col < ncols) {
			work[col] = x[col0+col];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE); // sync group

	// Compute partial dot product
	float sum = (float)0;
	for (int k=0; k<ncols; k++) {
		sum += a[get_global_id(ROW_DIM)+m*(col0+k)] * work[k];
	}

	// Store in Y (P columns per row)
	y[get_global_id(ROW_DIM)+m*get_global_id(COL_DIM)] = sum;
}

// Reduce M = get_global_size(0) rows of P values in matrix Y.
// Stores the result in first column of Y.
__kernel void reduce_rows(__global float *y, int m, int p)
{
	int row = get_global_id(0);
	float sum = (float)0;
	for (int col=0; col<p; col++) {
		sum += y[row + m*col];
	}
	y[row] = sum;
}

// http://stackoverflow.com/questions/15597299/matrix-vector-multiplications-using-opencl
/*__kernel void matrixVectorMul(__global float* resultVector,
    __global float* matrixA,
    __global float* vectorB, 
    int width_A)
{
    int tx = get_global_id(0);
    __local float vectB[4096*2];

    event_t copy_event = async_work_group_copy(vectB, vectorB, 4096*2, 0);
    wait_group_events(1,copy_event);

    float value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectB[k];
    }

    resultVector[tx] = value;
}*/

);

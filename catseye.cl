OCLSTRINGIFY(

#define identity(a)	(a)
#define softmax(a)	(a)	// FIXME
#define sigmoid(a)	(1.0f / (1 + exp(-a)))
#define normal_tanh(a)	(tanh(a))
#define scaled_tanh(a)	(1.7159f * tanh(0.66667f * a))
#define relu(a)		(a > 0 ? a : 0)
#define LeakyReLU(a)	(a > 0 ? a : a * 0.01)

#define LINEAR_FORWARD(act) \
void linear_forward_##act(__global const float *x, __global const float *w, __global float *o, uint is, uint os)\
{\
	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {\
		w += i;\
		float sum = 0;\
		for (int k=0; k<is; k++) {\
			/*sum += w[os*k] * x[k];*/\
			sum = fma(w[k*os], x[k], sum);\
		}\
		sum += w[is*os];\
		o[i] = act(sum);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
LINEAR_FORWARD(identity);
LINEAR_FORWARD(softmax);	// FIXME
LINEAR_FORWARD(sigmoid);
LINEAR_FORWARD(normal_tanh);
LINEAR_FORWARD(scaled_tanh);
LINEAR_FORWARD(relu);
LINEAR_FORWARD(LeakyReLU);

__kernel void forward(__global const float *x, __global float *w, __global float *o, __global float *d, uint8 args)
{
	linear_forward_sigmoid(x+args[5], w, o+784, 784, 200);
//	barrier(CLK_GLOBAL_MEM_FENCE);
/*if (!get_global_id(0)) {
	//for (int i=0; i<784; i++) printf("%f ", x[i+a[5]]);
	for (int i=0; i<200; i++) printf("%f ", o[784+i]);
	printf("\n");
}*/
	linear_forward_identity(o+784, w+785*200, o+784+200, 200, 10);
}


// a[0]: out
// a[1]: in+1 (bias)
// a[2]: offset of o
// a[3]: offset of w
// a[4]: offset of current delta
// a[5]: offset of prev delta
__kernel void linear_backward(__global const float *o, __global const float *w, __global float *d, uint8 a)
{
	int gid = get_global_id(0);
	if (gid < a[1]) {
		o += a[2];
		w += a[3] + gid*a[0];
		__global float *delta = d + a[4];
		d += a[5];
		float s = 0;
		for (int k=0; k<a[0]; k++) {
			s += w[k] * delta[k];
		}
		d[gid] = s /* * dact(o[gid]) */;
	}
}

#define ETA 1e-8
// a[0]: out
// a[1]: in+1 (bias)
// a[2]: offset of w
// a[3]: offset of d
__kernel void linear_update(__global const float *o, __global float *w, __global const float *d, uint8 a)
{
	int gid = get_global_id(0);
	if (gid < a[1]) {
		w += a[2] + gid*a[0];
		d += a[3];

		float s = ETA * o[gid];
		for (int k=0; k<a[0]; k++) {
			w[k] -= s * d[k];
		}
	}
}


__kernel void loss_0_1(__global const float *o, __global float *d, int a)
{
	int gid = get_global_id(0);
	d[gid] = a==gid ? o[gid]-1 : o[gid];	// 1-of-K
}
__kernel void loss_mse(__global const float *o, __global float *d, __global const float *a)
{
	int gid = get_global_id(0);
	d[gid] = o[gid] - a[gid];
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
/*__kernel void linear_forward(__global float *x, __global float *a, __global float *y, uint8 pa)
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
}*/


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

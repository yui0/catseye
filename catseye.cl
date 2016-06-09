OCLSTRINGIFY(

#define SIGMOID
//#define LINEAR
#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif defined SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif
#define act(output)	ACTIVATION_FUNCTION(output)

//__kernel void gemv1(__global const float *a, __global const float *x, __global float *y, int m, int n)
__kernel void gemv1_act(__global const float *x, __global const float *a, __global float *y, int n, int m)
{
	int i = get_global_id(0); // row index
	if (i < m) {

	float sum = 0;
	for (int k=0; k<=n; k++) {
		sum += a[i + m*k] * x[k];
	}
	y[i] = act(sum);

	}
}
__kernel void gemv1(__global const float *x, __global const float *a, __global float *y, int n, int m)
{
	int i = get_global_id(0); // row index
	if (i < m) {

	float sum = 0;
	for (int k=0; k<=n; k++) {
		sum += a[i + m*k] * x[k];
	}
	y[i] = sum;

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

);

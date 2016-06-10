OCLSTRINGIFY(

#define identity(a)	(a)
#define softmax(a)	(a)	// FIXME
#define sigmoid(a)	(1.0f / (1 + exp(-a)))
#define normal_tanh(a)	(tanh(a))
#define scaled_tanh(a)	(1.7159f * tanh(0.66667f * a))
#define relu(a)		(a > 0 ? a : 0)
#define LeakyReLU(a)	(a > 0 ? a : a * 0.01)

#define LINEAR_FORWARD(act) \
__kernel void linear_forward_##act(__global float *y, __global float *a, uint8 pa)\
{\
	int gid = get_global_id(0);\
	if (gid < pa[1]) {\
		__global float *x = y + pa[2];\
		a += pa[3];\
		y += pa[4];\
		float sum = 0;\
		for (int k=0; k<=pa[0]; k++) {\
			sum += a[gid + pa[1]*k] * x[k];\
		}\
		y[gid] = act(sum);\
	} else if (gid == pa[1]) y[gid] = 1;\
}
LINEAR_FORWARD(identity);
LINEAR_FORWARD(softmax);	// FIXME
LINEAR_FORWARD(sigmoid);
LINEAR_FORWARD(normal_tanh);
LINEAR_FORWARD(scaled_tanh);
LINEAR_FORWARD(relu);
LINEAR_FORWARD(LeakyReLU);


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

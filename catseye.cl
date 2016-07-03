OCLSTRINGIFY(

#define identity(a)		(a)
#define d_identity(a)		(1.0)
#define softmax(a)		(a)	// FIXME
#define d_softmax(a)		(1.0)	// FIXME
#define sigmoid(a)		(1.0 / (1.0 + exp(-a)))
#define d_sigmoid(a)		((1.0 - a) * a)
#define normal_tanh(a)		(tanh(a))
#define d_normal_tanh(a)	(1.0 - a*a)
#define scaled_tanh(a)		(1.7159 * tanh(2.0/3.0 * a))
#define d_scaled_tanh(a)	((2.0/3.0)/1.7159 * (1.7159-a)*(1.7159+a))
#define relu(a)			(a > 0 ? a : 0)
#define d_relu(a)		(a > 0 ? 1.0 : 0)
#define LeakyReLU(a)		(a > 0 ? a : a * 0.01)
#define d_LeakyReLU(a)		(a > 0 ? 1.0 : 0.01)

#define ACTIVATION_FUNCTION(act) \
inline void activate_##act(__global float *o, int n)\
{\
	if (!get_group_id(0))\
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {\
		o[i] = act(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
ACTIVATION_FUNCTION(sigmoid)

#define DACTIVATION_FUNCTION(dact) \
inline void dactivate_##dact(__global float *o, int n)\
{\
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {\
		o[i] = d_##dact(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
DACTIVATION_FUNCTION(sigmoid)

#define LINEAR_FORWARD(act) \
void linear_forward_##act(__global const float *x, __global const float *w, __global float *o, uint is, uint os)\
{\
	if (!get_group_id(0))\
	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {\
		w += i;\
		float s = 0;\
		for (int k=0; k<is; k++) {\
			/*sum += w[os*k] * x[k];*/\
			s = fma(w[k*os], *x++, s);\
		}\
		s += w[is*os];\
		o[i] = act(s);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
LINEAR_FORWARD(identity)
LINEAR_FORWARD(softmax)			// FIXME
LINEAR_FORWARD(sigmoid)
LINEAR_FORWARD(normal_tanh)
LINEAR_FORWARD(scaled_tanh)
LINEAR_FORWARD(relu)
LINEAR_FORWARD(LeakyReLU)

#define LINEAR_BACKWARD(dact) \
void linear_backward_##dact(__global const float *o, __global const float *w, __global float *d, __global const float *delta, uint is, uint os)\
{\
	if (!get_group_id(0))\
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {\
		float s = 0;\
		for (int k=0; k<os; k++) {\
			/*s += (*w++) * delta[k];*/\
			s = fma(*w++, delta[k], s);\
		}\
		d[i] = s * dact(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
LINEAR_BACKWARD(identity)
LINEAR_BACKWARD(softmax)		// FIXME
LINEAR_BACKWARD(sigmoid)
LINEAR_BACKWARD(normal_tanh)
LINEAR_BACKWARD(scaled_tanh)
LINEAR_BACKWARD(relu)
LINEAR_BACKWARD(LeakyReLU)

#define ETA 1e-8
void linear_update(__global const float *o, __global float *w, __global const float *d, uint is, uint os)
{
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {
		float a = -ETA * o[i];
		for (int k=0; k<os; k++) {
			//*w++ -= a * d[k];
			*w = fma(a, d[k], *w);
			w++;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void forward(__global const float *x, __global float *w, __global float *o, __global float *d, __global float *t, uint8 args)
{
	linear_forward_sigmoid(x+args[5], w, o+784, 784, 200);
	/*if (!get_global_id(0)) {
		//for (int i=0; i<784; i++) printf("%f ", x[i+a[5]]);
		for (int i=0; i<200; i++) printf("%f ", o[784+i]);
		printf("\n");
	}*/

/*	linear_forward_identity(x+args[5], w, o, 784, 200);
	activate_sigmoid(o, 200);
//	barrier(CLK_GLOBAL_MEM_FENCE);
//	barrier(CLK_LOCAL_MEM_FENCE);
	if (!get_global_id(0)) {
		for (int i=0; i<200; i++) {
			if (o[i]!=o[784+i]) printf("%f-%f/%d ", o[i], o[784+i], i);
		}
		printf("\n");
	}*/

//	linear_forward_identity(x+args[5], w, o+784, 784, 200);
//	activate_sigmoid(o+784, 200);
	/*if (!get_global_id(0)) {
		for (int i=0; i<200; i++) printf("%f ", o[784+i]);
		printf("\n");
	}*/
	//barrier(CLK_GLOBAL_MEM_FENCE);
	linear_forward_identity(o+784, w+785*200, o+784+200, 200, 10);
}

void loss_0_1(__global const float *o, __global float *d, uint a, uint n)
{
	if (!get_group_id(0))
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
//		printf("%f/%d-%f ", d[i],i,o[i]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

void loss_mse(__global const float *o, __global float *d, __global const float *a, uint n)
{
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = o[i] - a[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void train(__global const float *x, __global float *w, __global float *o, __global float *d, __global float *t, uint8 args)
{
	union {
		__global uint		*ip;
		__global float	*fp;
	} ptr;
	ptr.fp = t;

	forward(x, w, o, d, t, args);
	barrier(CLK_GLOBAL_MEM_FENCE);
	loss_0_1(o+784+200, d+200+1, ptr.ip[args[0]], 10);
/*	if (!get_global_id(0)) {
		printf("cl:%d  ",t[args[0]]);
		for (int i=0; i<10; i++) {
			printf("%f ", d[200+1+i]);
		}
		printf("\n");
	}*/
	//dactivate_sigmoid(o+784, 200);
//	linear_backward_sigmoid(o+784+200, w+785*200, d, d+200+1, 200, 10);
//	linear_update(o, w, d, 200, 784);
//	linear_update(o+784, w+785*200, d+200+1, 10, 200);
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

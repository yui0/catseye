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

#if 0
#define ACTIVATION_FUNCTION(act) \
inline void activate_##act(global float *o, int n)\
{\
	if (!get_group_id(0))\
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {\
		o[i] = act(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
ACTIVATION_FUNCTION(sigmoid)

#define DACTIVATION_FUNCTION(dact) \
inline void dactivate_##dact(global float *o, int n)\
{\
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {\
		o[i] = d_##dact(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}
DACTIVATION_FUNCTION(sigmoid)
#endif

#define LINEAR_FORWARD(act) \
void linear_forward_##act(global const float *x, global const float *w, global float *o, uint is, uint os)\
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
		w -= i;\
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
void linear_backward_##dact(global const float *o, global const float *w, global float *d, global const float *delta, uint is, uint os)\
{\
	if (!get_group_id(0))\
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {\
		w += i*os;\
		float s = 0;\
		for (int k=0; k<os; k++) {\
			/*s += (*w++) * delta[k];*/\
			s = fma(*w++, delta[k], s);\
		}\
		d[i] = s * dact(o[i]);\
		w -= (i*os + os);\
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

void linear_update(float eta, global const float *o, global float *w, global const float *d, uint is, uint os)
{
	if (!get_group_id(0))
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {
		global float *p = w + i*os;
//		float a = eta * o[i];
		float a = -eta * o[i];
		for (int k=0; k<os; k++) {
//			*p++ -= a * d[k];
			*p = fma(a, d[k], *p);
			p++;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void forward(global const float *x, global float *w, global float *o, global float *d, global float *t, uint8 args)
{
	linear_forward_sigmoid(x+args[5], w, o+784+1, 784, 200);
	/*if (!get_global_id(0)) {
		//for (int i=0; i<784; i++) printf("%f ", x[i+a[5]]);
		for (int i=0; i<200; i++) printf("%f ", o[784+i]);
		printf("\n");
	}*/
	linear_forward_identity(o+784+1, w+785*200, o+784+1+200+1, 200, 10);
}

void loss_0_1(global const float *o, global float *d, uint a, uint n)
{
	if (!get_group_id(0))
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

void loss_mse(global const float *o, global float *d, global const float *a, uint n)
{
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = o[i] - a[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

/*int rand(int *seed) // 1 <= *seed < m
{
	const int a = 16807; // ie 7**5
	const int m = 2147483647; // ie 2**31-1
	*seed = (*seed * a) % m;
	return *seed;
}

kernel void random_number_kernel(global int *seed_memory)
{
	int gid = get_global_id(0);

	// Since the Park-Miller PRNG generates a SEQUENCE of random numbers
	// we have to keep track of the previous random number, because the next
	// random number will be generated using the previous one.
	int seed = seed_memory[gid];

	int random_number = rand(&seed); // Generate the next random number in the sequence.

	seed_memory[gid] = seed; // Save the seed for the next time this kernel gets enqueued.
}*/
uint xorshift_int(local uint4 *ctx)
{
	uint t = ctx->x ^ (ctx->x << 11);
	*ctx = ctx->yzww;
	ctx->w = ctx->w ^ (ctx->w >> 19) ^ (t ^ (t >> 8));

	return ctx->w;
}

float xorshift_float(local uint4 *ctx)
{
	return xorshift_int(ctx) * 2.3283064e-10;
}

kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, uint8 args)
{
	union {
		global uint *ip;
		global float *fp;
	} ptr;
	ptr.fp = t;

	int lid = get_local_id(0);
	local uint4 seed[256];
	local uint r[256];
	seed[lid] = lid;
	r[lid] = xorshift_int(seed+lid) % 60000;
	barrier(CLK_LOCAL_MEM_FENCE);
/*if (!get_global_id(0)) {
	for (int i=0; i<256; i++) {
		printf("%d ", r[i]);
	}
}*/

	for (int n=0; n</*args[1]*/256; n++) {
//		args[5] = n*784;
//		args[0] = n;
		args[5] = r[n]*784;
		args[0] = r[n];

		forward(x, w, o, d, t, args);

		loss_0_1(o+784+1+200+1, d+200+1, ptr.ip[args[0]], 10);
		o[784+1+200] = 1;
		linear_backward_identity(o+784+1, w+785*200, d, d+200+1, 200, 10);
		linear_update(/*args[1]*/0.01, x+args[5], w, d, 784, 200);
		linear_update(/*args[1]*/0.01, o+784+1, w+785*200, d+200+1, 200, 10);
	}
}


kernel void memset_uint4(global uint4 *mem, __private uint4 val)
{
	mem[get_global_id(0)] = val;
}
kernel void memset_float(global float *mem, __private float val)
{
	mem[get_global_id(0)] = val;
}
// pa[0]: in
// pa[1]: out
/*kernel void linear_forward(global float *x, global float *a, global float *y, uint8 pa)
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
kernel void gemv(global const float *a, global const float *x, global float *y,
	local float *work, int m, int n)
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
kernel void reduce_rows(global float *y, int m, int p)
{
	int row = get_global_id(0);
	float sum = (float)0;
	for (int col=0; col<p; col++) {
		sum += y[row + m*col];
	}
	y[row] = sum;
}

// http://stackoverflow.com/questions/15597299/matrix-vector-multiplications-using-opencl
/*kernel void matrixVectorMul(global float* resultVector,
    global float* matrixA,
    global float* vectorB, 
    int width_A)
{
    int tx = get_global_id(0);
    local float vectB[4096*2];

    event_t copy_event = async_work_group_copy(vectB, vectorB, 4096*2, 0);
    wait_group_events(1,copy_event);

    float value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectB[k];
    }

    resultVector[tx] = value;
}*/

);

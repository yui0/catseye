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
/*	if (!get_group_id(0))*/\
/*	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {*/\
	for (int i=get_global_id(0); i<os; i+=get_global_size(0)) {\
		w += i;\
		float s = 0;\
		for (int k=0; k<is; k++) {\
			/*sum += w[os*k] * x[k];*/\
			s = mad(w[k*os], *x++, s);\
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
/*	if (!get_group_id(0))*/\
/*	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {*/\
	for (int i=get_global_id(0); i<=is; i+=get_global_size(0)) {\
		w += i*os;\
		float s = 0;\
		for (int k=0; k<os; k++) {\
			/*s += (*w++) * delta[k];*/\
			s = mad(*w++, delta[k], s);\
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

inline void linear_update(float eta, global const float *o, global float *w, global const float *d, uint is, uint os)
{
	if (!get_group_id(0))
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {
//	for (int i=get_global_id(0); i<=is; i+=get_global_size(0)) {
//if (get_group_id(0)) printf("%d ",i);
		global float *p = w + i*os;
		float a = -eta * o[i];
		for (int k=0; k<os; k++) {
//			*p++ += a * d[k];
//printf("%f ",p[0]);
			*p = mad(a, d[k], *p);
			p++;
		}
//printf("%d ",p-w);
	}
//	barrier(CLK_LOCAL_MEM_FENCE);
//	barrier(CLK_GLOBAL_MEM_FENCE);
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

uint xorshift_int(local uint4 *ctx)
{
	uint t = ctx->x ^ (ctx->x << 11);
	*ctx = ctx->yzww;
	ctx->w = ctx->w ^ (ctx->w >> 19) ^ (t ^ (t >> 8));

	return ctx->w;
}
/*float xorshift_float(local uint4 *ctx)
{
	return xorshift_int(ctx) * 2.3283064e-10;
}*/

kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, uint8 args)
{
	/*if (!get_global_id(0)) {
		printf("OpenCL training start!!\n");
		printf("group size: %d\n", get_num_groups(0));
		printf("global size: %d\n", get_global_size(0));
		printf("local size: %d\n", get_local_size(0));
	}*/

	union {
		global uint *ip;
		global float *fp;
	} ptr;
	ptr.fp = t;

	o[784+1+200] = 1;

	local uint seed;
	local uint4 r;
	r.xyzw = args[2];
	for (int n=args[1]; n>0; n--) {
		if (!get_global_id(0)) seed = xorshift_int(&r) % 60000;
		barrier(CLK_LOCAL_MEM_FENCE);

//		args[5] = seed*784;
//		forward(x, w, o, d, t, args);
		global float *p = x + seed*784;
		linear_forward_sigmoid(p, w, o+784+1, 784, 200);
		linear_forward_identity(o+784+1, w+785*200, o+784+1+200+1, 200, 10);

		loss_0_1(o+784+1+200+1, d+200+1, ptr.ip[seed], 10);
//		o[784+1+200] = 1;
		linear_backward_identity(o+784+1, w+785*200, d, d+200+1, 200, 10);
		linear_update(/*eta*/0.01, p, w, d, 784, 200);
		linear_update(/*eta*/0.01, o+784+1, w+785*200, d+200+1, 200, 10);
	}
}


/*kernel void memset_uint4(global uint4 *mem, __private uint4 val)
{
	mem[get_global_id(0)] = val;
}
kernel void memset_float(global float *mem, __private float val)
{
	mem[get_global_id(0)] = val;
}*/
/*uint clock_time()
{
	uint clock_time;
	asm("mov.u32 %0, %%clock;" : "=r"(clock_time));
	return clock_time;
}*/
);

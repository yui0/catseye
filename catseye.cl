OCLSTRINGIFY(

// http://industrybestpractice.blogspot.jp/2012/07/global-synchronisation-in-opencl.html
void global_sync(volatile global int *flags)
{
	const size_t thread_id = get_local_id(0);
	const size_t workgroup_id = get_group_id(0);

	if (thread_id == 0) {
		flags[workgroup_id] = 1;
	}

	if (workgroup_id == 0) {
		if (thread_id < get_num_groups(0)) {
//			while (flags[thread_id] != 1) ;
			while (atomic_or(&flags[thread_id], 0) != 1) ;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

		if (thread_id < get_num_groups(0)) {
			flags[thread_id] = 0;
		}
	}

	if (thread_id == 0) {
//		while (flags[workgroup_id] != 0) ;
		while (atomic_or(&flags[workgroup_id], 0) != 0) ;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void vdot(global const float *x, global const float *y, global float *r, uint n)
{
	local float acc[1024];
	float priv_acc = 0;
	uint lid = get_local_id(0);

	acc[lid] = 0;
	for (uint i=lid; i<n; i+=get_local_size(0)) {
		priv_acc = mmad(x[i], y[i], priv_acc);
	}
	acc[lid] = priv_acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < 128) acc[lid] += acc[lid+128];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 64) acc[lid] += acc[lid+64];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 32) acc[lid] += acc[lid+32];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 16) acc[lid] += acc[lid+16];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 8) acc[lid] += acc[lid+8];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 4) acc[lid] += acc[lid+4];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 2) acc[lid] += acc[lid+2];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (!lid) r[0] = acc[0];
}
kernel void vdotT(global const float *x, global const float *y, global float *r, uint n, uint m)
{
	uint lid = get_local_id(0);
	local float acc[1024];
	float priv_acc = 0;

	for (uint i=lid; i<n; i+=get_local_size(0)) {
		priv_acc = mmad(x[i*m], y[i], priv_acc);
	}
	acc[lid] = priv_acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < 128) acc[lid] += acc[lid+128];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 64) acc[lid] += acc[lid+64];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 32) acc[lid] += acc[lid+32];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 16) acc[lid] += acc[lid+16];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 8) acc[lid] += acc[lid+8];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 4) acc[lid] += acc[lid+4];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid < 2) acc[lid] += acc[lid+2];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (!lid) r[0] = acc[0] + x[n*m];
//if (!lid) printf("%f ",r[0]);
}

//#define mmad(x,y,z)		(x+y*z)
#define mmad(x,y,z)		mad(x,y,z)
//#define mmad(x,y,z)		fma(x,y,z)

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
	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {\
/*	for (int i=get_global_id(0); i<os; i+=get_global_size(0)) {*/\
		w += i;\
		float s = 0;\
		for (int k=0; k<is; k++) {\
			s = mmad(w[k*os], x[k], s);\
		}\
		s += w[is*os];\
		o[i] = act(s);\
		w -= i;\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}\
kernel void __linear_forward_##act(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)\
{\
/*	linear_forward_##act(args[0] ? x+args[1] : o+args[1], w+args[2], o+args[3], args[4], args[5]);*/\
	x = args[0] ? x+args[1] : o+args[1];\
	w += args[2];\
	o += args[3];\
	int is = args[4];\
	int os = args[5];\
	for (int i=get_global_id(0); i<os; i+=get_global_size(0)) {\
		w += i;\
		float s = 0;\
		for (int k=0; k<is; k++) {\
			s = mmad(w[k*os], x[k], s);\
		}\
		s += w[is*os];\
		o[i] = act(s);\
		w -= i;\
	}\
}\
kernel void _linear_forward_##act(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)\
{\
	x = args[0] ? x+args[1] : o+args[1];\
	w += args[2];\
	o += args[3];\
	int is = args[4];\
	int os = args[5];\
	for (int i=0; i<os; i++) {\
		vdotT(w, x, o, is, os);\
		if (!get_global_id(0)) *o = act(*o);\
		w++;\
		o++;\
	}\
}\
void ___linear_forward_##act(global const float *x, global const float *w, global float *o, uint is, uint os)\
{\
	for (int i=0; i<os; i++) {\
		vdotT(w, x, o, is, os);\
		if (!get_local_id(0)) *o = act(*o);\
		w++;\
		o++;\
	}\
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
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {\
/*	for (int i=get_global_id(0); i<=is; i+=get_global_size(0)) {*/\
		w += i*os;\
		float s = 0;\
		for (int k=0; k<os; k++) {\
			s = mmad(*w++, delta[k], s);\
		}\
		d[i] = s * dact(o[i]);\
		w -= (i*os + os);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}\
kernel void _linear_backward_##dact(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)\
{\
	linear_backward_##dact(o+args[0], w+args[1], d+args[2], d+args[3], args[4], args[5]);\
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
//	if (!get_group_id(0))
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {
//	for (int i=get_global_id(0); i<=is; i+=get_global_size(0)) {
		global float *p = w + i*os;
		float a = -eta * o[i];
		for (int k=0; k<os; k++) {
			*p = mmad(a, d[k], *p);
			p++;
		}
	}
//	barrier(CLK_LOCAL_MEM_FENCE);
}
kernel void _linear_update(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)
{
	//linear_update(0.01, args[0] ? x+args[1] : o+args[1], w+args[2], d+args[3], args[4], args[5]);
	o = args[0] ? x+args[1] : o+args[1];
	w += args[2];
	d += args[3];
	uint is = args[4];
	uint os = args[5];
	float eta = 0.01;

	for (int i=get_global_id(0); i<=is; i+=get_global_size(0)) {
		global float *p = w + i*os;
		float a = -eta * o[i];
		for (int k=0; k<os; k++) {
			*p = mmad(a, d[k], *p);
			p++;
		}
	}
}
// for GPU, not for CPU
/*void linear_update(float eta, global const float *o, global float *w, global const float *d, uint is, uint os)
{
	if (!get_group_id(0))
	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {
		for (int k=0; k<is; k++) {
			w[i+k*os] = mmad(-eta * o[k], d[i], w[i+k*os]);
		}
		w[i+is*os] = mmad(-eta, d[i], w[i+is*os]);
	}
}*/

kernel void forward(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)
{
	linear_forward_sigmoid(x+args[0], w, o+784+1, 784, 200);
	linear_forward_identity(o+784+1, w+785*200, o+784+1+200+1, 200, 10);

//	___linear_forward_sigmoid(x+args[0], w, o+784+1, 784, 200);
//	global_sync(sync);
//	linear_forward_identity(o+784+1, w+785*200, o+784+1+200+1, 200, 10);
}

void loss_0_1(global const float *o, global float *d, uint a, uint n)
{
//	if (!get_group_id(0))
	for (int i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
kernel void _loss_0_1(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)
{
	loss_0_1(o+args[0], d+args[1], args[2], args[3]);
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

#if 1
kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)
{
/*	if (!get_global_id(0)) {
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

/*#ifdef __CPU__
	int time;
	if (get_global_id(0)==1) time = clock();
#endif*/
	o[784+1+200] = 1;

	local uint seed;
	local uint4 r;
	r.xyzw = args[2];
	if (!get_group_id(0))
	for (int n=args[1]; n>0; n--) {
		if (!get_global_id(0)) seed = xorshift_int(&r) % 60000;
//		if (!get_local_id(0)) seed = xorshift_int(&r) % 60000;
		barrier(CLK_LOCAL_MEM_FENCE);
		global float *p = x + seed*784;

//if (!get_group_id(0)) {
		linear_forward_sigmoid(p, w, o+784+1, 784, 200);
		//___linear_forward_sigmoid(p, w, o+784+1, 784, 200);
		linear_forward_identity(o+784+1, w+785*200, o+784+1+200+1, 200, 10);

		loss_0_1(o+784+1+200+1, d+200+1, ptr.ip[seed], 10);
		linear_backward_identity(o+784+1, w+785*200, d, d+200+1, 200, 10);
		linear_update(/*eta*/0.01, p, w, d, 784, 200);
		linear_update(/*eta*/0.01, o+784+1, w+785*200, d+200+1, 200, 10);
//}
//		global_sync(sync);
	}

/*#ifdef __CPU__
	if (get_global_id(0)==1) printf(" [%.2fs]\n", (clock()-time)/1000.0/1000.0);
#endif*/
}
#else
// http://www.chokkan.org/research/survey/Minibatch%20and%20Parallelization%20for%20Online%20Large%20Margin%20Structured%20Learning.pdf
kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, global int *sync, uint8 args)
{
	union {
		global uint *ip;
		global float *fp;
	} ptr;
	ptr.fp = t;

/*#ifdef __CPU__
	int time;
	if (get_global_id(0)==1) time = clock();
#endif*/
//	o[784+1+200] = 1;

	int MINIBATCH = get_num_groups(0);
	local uint seed[16];
//	global int *seed = sync+10;
	local uint4 r;
	r.xyzw = args[2];
	for (int n=args[1]/MINIBATCH; n>0; n--) {
//		if (!get_group_id(0))
//		for (int m=MINIBATCH-1; m>=0; m--) {
			{int m = get_group_id(0);
//			if (!get_global_id(0)) seed[m] = xorshift_int(&r) % 60000;
			if (!get_local_id(0)) seed[m] = xorshift_int(&r) % 60000;
			barrier(CLK_LOCAL_MEM_FENCE);
			global float *p = x + seed[m]*784;

			int dd = m*(200+1+10+1);
			int oo = m*(784+1+200+1+10+1);
//			if (!get_local_id(0)) sync[m] = seed[m]*784;
			barrier(CLK_GLOBAL_MEM_FENCE);

			linear_forward_sigmoid(p, w, o+oo+784+1, 784, 200);
//		___linear_forward_sigmoid(p, w, o+784+1, 784, 200);
//		global_sync(sync);
			linear_forward_identity(o+oo+784+1, w+785*200, o+oo+784+1+200+1, 200, 10);

			loss_0_1(o+oo+784+1+200+1, d+dd+200+1, ptr.ip[seed[m]], 10);
			linear_backward_identity(o+oo+784+1, w+785*200, d+dd, d+dd+200+1, 200, 10);

//			linear_update(0.01, p, w, d+dd, 784, 200);
//			linear_update(0.01, o+oo+784+1, w+785*200, d+dd+200+1, 200, 10);
		}
//		global_sync(sync);
//		if (!get_group_id(0))
//		for (int m=MINIBATCH-1; m>=0; m--) {
			{int m = get_group_id(0);
			global float *p = x + seed[m]*784;//sync[m];
			int dd = m*(200+1+10+1);
			int oo = m*(784+1+200+1+10+1);
			barrier(CLK_GLOBAL_MEM_FENCE);

			linear_update(0.01, p, w, d+dd, 784, 200);
			linear_update(0.01, o+oo+784+1, w+785*200, d+dd+200+1, 200, 10);
		}
		global_sync(sync);
	}

/*#ifdef __CPU__
	if (get_global_id(0)==1) printf(" [%.2fs]\n", (clock()-time)/1000.0/1000.0);
#endif*/
}
#endif

);

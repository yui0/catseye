OCLSTRINGIFY(
%PREPROCESSOR%

// http://industrybestpractice.blogspot.jp/2012/07/global-synchronisation-in-opencl.html
static void global_sync(volatile global uint *flags)
{
	const size_t thread_id = get_local_id(0);
	const size_t workgroup_id = get_group_id(0);

	if (thread_id == 0) {
		flags[workgroup_id] = 1;
	}

	if (workgroup_id == 0) {
		if (thread_id < get_num_groups(0)) {
//			while (flags[thread_id] != 1) ;
			while (atomic_or(&flags[thread_id], 0) != 1) ;	// for AMD
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

		if (thread_id < get_num_groups(0)) {
			flags[thread_id] = 0;
		}
	}

	if (thread_id == 0) {
//		while (flags[workgroup_id] != 0) ;
		while (atomic_or(&flags[workgroup_id], 0) != 0) ;	// for AMD
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
float atom_add_float(global float const *address, const float value)
{
	uint oldval;
	uint newval;
	uint readback;

	*(float*)&oldval = *address;
	*(float*)&newval = (*(float*)&oldval + value);
	while ((readback = atom_cmpxchg((global uint*)address, oldval, newval)) != oldval) {
		oldval = readback;
		*(float*)&newval = (*(float*)&oldval + value);
	}
	return *(float*)&oldval;
}

//#define mmad(x,y,z)		((x)*(y)+(z))
#define mmad(x,y,z)		mad((x),(y),(z))	// mad is intended to be used where speed is preferred over accuracy.
//#define mmad(x,y,z)		fma((x),(y),(z))

static void vdot(local float *acc, global const float *x, global const float *y, global float *r, uint n)
{
	uint lid = get_local_id(0);
	float priv_acc = 0;

	for (int i=lid; i<n; i+=get_local_size(0)) {
		priv_acc = mmad(x[i], y[i], priv_acc);
	}
	acc[lid] = priv_acc;
	barrier(CLK_LOCAL_MEM_FENCE);

//	if (lid < 256) acc[lid] += acc[lid+256];
//	barrier(CLK_LOCAL_MEM_FENCE);
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

	if (!lid) r[0] = acc[0] + acc[1];
}
static void vdotT(local float *acc, global const float *x, global const float *y, global float *r, uint n, uint m)
{
	uint lid = get_local_id(0);
	float priv_acc = 0;

	for (int i=lid; i<n; i+=get_local_size(0)) {
		priv_acc = mmad(x[i*m], y[i], priv_acc);
	}
	acc[lid] = priv_acc;
	barrier(CLK_LOCAL_MEM_FENCE);

//	if (lid < 256) acc[lid] += acc[lid+256];
//	barrier(CLK_LOCAL_MEM_FENCE);
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

	if (!lid) r[0] = acc[0] + acc[1];
}

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
#define relu(a)			(a>0 ? a : 0)
#define d_relu(a)		(a>0 ? 1.0 : 0)
#define LeakyReLU(a)		(a>0 ? a : a * 0.01)
#define d_LeakyReLU(a)		(a>0 ? 1.0 : 0.01)

#define LINEAR_FORWARD(act) \
static void linear_forward_##act(global const float *x, global const float *w, global float *o, uint is, uint os)\
{\
	for (int i=get_local_id(0); i<os; i+=get_local_size(0)) {\
		global const float *p = w + i;\
		float s = 0;\
		for (int k=0; k<is; k++) {\
			s = mmad(p[k*os], x[k], s);\
		}\
		s += p[is*os];\
		o[i] = act(s);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}\
static void __linear_forward_##act(local float *acc, global const float *x, global const float *w, global float *o, uint is, uint os)\
{\
	for (int i=0; i<os; i++) {\
		vdotT(acc, w+i, x, o+i, is, os);\
		if (!get_local_id(0)) o[i] = act(o[i] + w[is*os+i]);\
	}\
}\
static void g_linear_forward_##act(local float *acc, global const float *x, global const float *w, global float *o, uint is, uint os)\
{\
	for (int i=get_group_id(0); i<os; i+=get_num_groups(0)) {\
		vdotT(acc, w+i, x, o+i, is, os);\
		if (!get_local_id(0)) o[i] = act(o[i] + w[is*os+i]);\
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
static void linear_backward_##dact(global const float *o, global const float *w, global float *d, global const float *delta, uint is, uint os)\
{\
/*	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) { bias*/\
	for (int i=get_local_id(0); i<is; i+=get_local_size(0)) {\
		global const float *p = w + i*os;\
		float s = 0;\
		for (int k=0; k<os; k++) {\
			s = mmad(*p++, delta[k], s);\
		}\
		d[i] = s * d_##dact(o[i]);\
	}\
	barrier(CLK_LOCAL_MEM_FENCE);\
}\
kernel void _linear_backward_##dact(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)\
{\
	linear_backward_##dact(o+args[0], w+args[1], d+args[2], d+args[3], args[4], args[5]);\
}\
static void g_linear_backward_##dact(local float *acc, global const float *o, global const float *w, global float *d, global const float *delta, uint is, uint os)\
{\
	for (int i=get_group_id(0); i<=is; i+=get_num_groups(0)) {\
		vdot(acc, w+i*os, delta, d+i, os);\
		if (!get_local_id(0)) d[i] = d[i] * d_##dact(o[i]);\
	}\
}
LINEAR_BACKWARD(identity)
LINEAR_BACKWARD(softmax)		// FIXME
LINEAR_BACKWARD(sigmoid)
LINEAR_BACKWARD(normal_tanh)
LINEAR_BACKWARD(scaled_tanh)
LINEAR_BACKWARD(relu)
LINEAR_BACKWARD(LeakyReLU)

static void linear_update(float eta, global const float *o, global float *w, global const float *d, uint is, uint os)
{
	for (int i=get_local_id(0); i<=is; i+=get_local_size(0)) {
		global float *p = w + i*os;
//		float a = -eta * o[i];
		float a = -eta;
		if (i!=is) a *= o[i];	// !bias
		for (int k=0; k<os; k++) {
			//atom_add_float(p, a*d[k]);
			*p = mmad(a, d[k], *p);
			p++;
		}
	}
}
/*kernel void _linear_update(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)
{
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
static void g_linear_update(float eta, global const float *o, global float *w, global const float *d, uint is, uint os)
{
	for (int i=get_group_id(0); i<=is; i+=get_num_groups(0)) {
		global float *p = w + i*os;
		float a = -eta * o[i];

		uint lid = get_local_id(0);
		for (int i=lid; i<is; i+=get_local_size(0)) {
			p[lid] = mmad(a, d[lid], p[lid]);
		}
	}
}*/

// FIXME
kernel void forward(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)
{
	local float acc[256];//[512];
	g_linear_forward_sigmoid(acc, x+args[0], w, o+784+1, 784, 200);
	global_sync(sync);
	g_linear_forward_identity(acc, o+784+1, w+785*200, o+784+1+200+1, 200, 10);
}

// http://kourindrug.sakura.ne.jp/waifu2x.html
// http://int.main.jp/txt/waifu2x.html#sec7
// https://github.com/ueshita/waifu2x-converter-glsl/blob/master/shaders/convolve_fs.glsl
static void convolutional_layer_forward3x3(global const float *ss, global const float *ww, global float *oo, uint ix, uint iy, uint is, uint os, uint ich, uint och)
{
	global const float3 *weightMatrix = (global const float3 *)ww;
	uint sx = ix-2;	// out
	uint sy = iy-2;

	for (uint c=get_local_id(0); c<och; c+=get_local_size(0)) {	// out
		global const float3 *s = (global const float3 *)ss;
		global float3 *o = (global float3 *)&oo[sx*sy*c];
		for (uint cc=ich; cc>0; cc--) {	// in
			float3 w0 = *weightMatrix++;
			float3 w1 = *weightMatrix++;
			float3 w2 = *weightMatrix++;
			for (uint y=0; y<sy; y++) {
				for (uint x=0; x<sx; x++) {
					float3 s0 = *((global const float3 *)s);
					float3 s1 = *((global const float3 *)(s+sx));
					float3 s2 = *((global const float3 *)(s+sx*2));
					s++;
					*o++ = dot(s0, w0) + dot(s1, w1) + dot(s2, w2);
				}
				s += 2;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
#define CONVOLUTIONAL_ACT(act) \
static void convolutional_layer_##act(global const float *s, global const float *w, global float *o, uint ix, uint iy, uint is, uint os, uint ich, uint och)\
{\
	uint sx = ix-2;\
	uint sy = iy-2;\
	uint size = sx*sy*och;\
	for (uint n=get_local_id(0); n<size; n+=get_local_size(0)) {\
		o[n] = act(o[n]);\
	}\
}
CONVOLUTIONAL_ACT(relu)
CONVOLUTIONAL_ACT(LeakyReLU)
// http://blog.yusugomori.com/post/129688163130/%E6%95%B0%E5%BC%8F%E3%81%A7%E6%9B%B8%E3%81%8D%E4%B8%8B%E3%81%99-convolutional-neural-networks-cnn
static void convolutional_layer_backward3x3(global const float *prev_out, global const float *ww, global float *prev_delta, global const float *delta, uint ix, uint iy, uint is, uint os, uint ich, uint och)
{
	global const float3 *weightMatrix = (global const float3 *)ww;
	uint sx = ix-2;	// out
	uint sy = iy-2;

	for (uint c=get_local_id(0); c<och; c+=get_local_size(0)) {	// out
		global const float3 *d = (global const float3 *)delta;
		global float3 *p = (global float3 *)&prev_delta[sx*sy*c];
		for (uint cc=ich; cc>0; cc--) {	// in
			float3 w0 = *weightMatrix++; w0 = w0.s210;
			float3 w1 = *weightMatrix++; w1 = w1.s210;
			float3 w2 = *weightMatrix++; w2 = w2.s210;
			for (uint y=0; y<sy; y++) {
				for (uint x=0; x<sx; x++) {
					float3 s0 = *((global const float3 *)d);
					float3 s1 = *((global const float3 *)(d+sx));
					float3 s2 = *((global const float3 *)(d+sx*2));
					d++;
					//*p++ = dot(s2.s210, w0) + dot(s1.s210, w1) + dot(s0.s210, w2);
					*p++ = dot(s0, w2) + dot(s1, w1) + dot(s2, w0);
				}
				p += 2;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
static void convolutional_layer_update3x3(float eta, global const float *prev_out, global float *w, global const float *curr_delta, uint ix, uint iy, uint is, uint os, uint ich, uint och)
{
	global float3 *weightMatrix = (global float3 *)w;
	uint sx = ix-2;	// out
	uint sy = iy-2;

	for (uint c=get_local_id(0); c<och; c+=get_local_size(0)) {	// out
		global const float3 *d = (global const float3 *)curr_delta;
		global float3 *p = (global float3 *)&prev_out[sx*sy*c];
		for (uint cc=ich; cc>0; cc--) {	// in
			float3 w0 = 0;
			float3 w1 = 0;
			float3 w2 = 0;
			for (uint y=0; y<sy; y++) {
				for (uint x=0; x<sx; x++) {
					float3 s0 = (*d,*d,*d);
					float3 s1 = (*(d+sx),*(d+sx),*(d+sx));
					float3 s2 = (*(d+sx*2),*(d+sx*2),*(d+sx*2));
					d++;
					float3 x0 = *((global const float3 *)p);
					float3 x1 = *((global const float3 *)(p+ix));
					float3 x2 = *((global const float3 *)(p+ix*2));
					p++;
					w0 += dot(s0, x0);
					w1 += dot(s1, x1);
					w2 += dot(s2, x2);
				}
				p += 2;
			}
			*weightMatrix++ += eta * w0;
			*weightMatrix++ += eta * w1;
			*weightMatrix++ += eta * w2;
		}
	}
}

static void loss_0_1(global const float *o, global float *d, uint a, uint n)
{
	for (uint i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
kernel void _loss_0_1(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)
{
	loss_0_1(o+args[0], d+args[1], args[2], args[3]);
}
static void g_loss_0_1(global const float *o, global float *d, uint a, uint n)
{
	for (uint i=get_global_id(0); i<n; i+=get_global_size(0)) {
		d[i] = a==i ? o[i]-1 : o[i];	// 1-of-K
	}
}

static void loss_mse(global const float *o, global float *d, global const float *a, uint n)
{
	for (uint i=get_local_id(0); i<n; i+=get_local_size(0)) {
		d[i] = o[i] - a[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

static uint xorshift_int(local uint4 *ctx)
{
	uint t = ctx->x ^ (ctx->x << 11);
	*ctx = ctx->yzww;
	ctx->w = ctx->w ^ (ctx->w >> 19) ^ (t ^ (t >> 8));

	return ctx->w;
}
/*static float xorshift_float(local uint4 *ctx)
{
	return xorshift_int(ctx) * 2.3283064e-10;
}*/
/*#define frand()		( xor128() / ((double)XOR128_MAX + 1.0f) )
int binomial(numerus p)
{
	int c = 0;
	numerus r = frand();
	if (r < p) c++;
	return c;
}*/

#if 1
#ifdef CL_DEBUG
#define CL_SDEBUG(statment) statment
#define CL_NDEBUG(statment)
#else
#define CL_SDEBUG(statment)
#define CL_NDEBUG(statment) statment
#endif
kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)
{
/*#ifdef __CPU__
	uint time;
	if (get_global_id(0)==1) time = clock();
#endif*/
	//local float acc[256];
	union {
		global uint *ip;
		global float *fp;
	} ptr;
	ptr.fp = t;

	uint MINIBATCH = get_num_groups(0);
	local uint label;
	local uint seed;
	local uint4 r;
	local float eta;
	uint N;
	if (!get_local_id(0)) {
		N = args[0];
		eta = args[4]*1e-8;
		r.xyzw = args[2] + get_group_id(0);
	}
//	if (!get_group_id(0)) for (int n=args[1]; n>0; n--)	// 94% (mnist)
	CL_SDEBUG(if (get_group_id(0)==1))
	CL_NDEBUG(for (int n=args[1]/MINIBATCH; n>0; n--))
	{
		uint m = get_group_id(0);
		if (!get_local_id(0)) {
			CL_NDEBUG(seed = xorshift_int(&r) % N); CL_SDEBUG(seed = 1);
			label = ptr.ip[seed];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Asynchronous Update
%GEN_CODE%
		CL_NDEBUG(global_sync(sync));
	}

#if 0
	if (!get_group_id(0)) {
		local float acc[256];
		global float *mse = sync+11;
%MSE_CODE%
		barrier(CLK_LOCAL_MEM_FENCE);
/*		global float *err = sync+10;
		if (!get_local_id(0)) {
			*err = 0.5 * (*err + *mse/size);
			printf("epochs %d, mse %f", args[3], *err);
		}*/
		if (!get_local_id(0)) {
			printf("epochs %d, mse %f [%.2fs]\n", args[3], *mse/size, args[5]*0.001*0.001);
		}
	}
#endif

	// http://jubat.us/en/method.html
	// http://sinhrks.hatenablog.com/entry/2015/12/17/000538
/*	global_sync(sync);
	for (int i=get_global_id(0); i<785*200+201*10; i+=get_global_size(0)) {
		float a = w[i];
		for (int m=1; m<get_num_groups(0); m++) {
			a += w[i+m*(785*200+201*10)];
		}
		a /= get_num_groups(0);
//		if (w[i] != a) printf("%f %f!\n", w[i], a);
		for (int m=0; m<get_num_groups(0); m++) {
			w[i+m*(785*200+201*10)] = a;
		}
		w[i] = w[i+(785*200+201*10)*(get_num_groups(0)-1)];
	}*/

/*#ifdef __CPU__
	if (get_global_id(0)==1) printf(" [%.2fs]\n", (clock()-time)/1000.0/1000.0);
#endif*/
}
#else
// 95% (mnist)
kernel void train(global const float *x, global float *w, global float *o, global float *d, global float *t, global uint *sync, uint8 args)
{
	union {
		global uint *ip;
		global float *fp;
	} ptr;
	ptr.fp = t;

	global uint *seed = sync+10;
	local float acc[256];//[512];
	local uint4 r;
	r.xyzw = args[2];
	for (int n=args[1]; n>0; n--) {
		if (!get_global_id(0)) seed[0] = xorshift_int(&r) % 60000;
		global_sync(sync);
		global const float *p = x + seed[0]*784;

		g_linear_forward_sigmoid(acc, p, w, o+784+1, 784, 200);
		global_sync(sync);
		g_linear_forward_identity(acc, o+784+1, w+785*200, o+784+1+200+1, 200, 10);
		global_sync(sync);

if (!get_group_id(0)) {
		loss_0_1(o+784+1+200+1, d+200+1, ptr.ip[seed[0]], 10);
		linear_backward_identity(o+784+1, w+785*200, d, d+200+1, 200, 10);
//		linear_update(0.01, p, w, d, 784, 200);
//		linear_update(0.01, o+784+1, w+785*200, d+200+1, 200, 10);
}
//		g_loss_0_1(o+784+1+200+1, d+200+1, ptr.ip[seed[0]], 10);
//		global_sync(sync);
//		g_linear_backward_identity(acc, o+784+1, w+785*200, d, d+200+1, 200, 10);
		global_sync(sync);
		g_linear_update(0.01, p, w, d, 784, 200);
		g_linear_update(0.01, o+784+1, w+785*200, d+200+1, 200, 10);
	}
}
#endif

);

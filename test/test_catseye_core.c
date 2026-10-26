// Regression tests for catseye.h: shape setup, batch normalization, data augmentation,
// safetensors I/O, fp16 / Q4_0 conversion and the inference GEMM / softmax kernels.
#define CATS_QUIET
#include "../catseye.h"

static int g_fail;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static void test_activation_shapes(void)
{
	// ELU and log layers are element-wise: outputs must equal inputs (they used to take the next layer's size)
	CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_ELU}, {5, CATS_LINEAR}, {0, CATS_ACT_SIGMOID}, {0, CATS_ACT_LOG}, {5, CATS_LOSS_MSE} };
	CatsEye c = { .batch=2 };
	_CatsEye__construct(&c, u, 6);
	CHECK(c.layer[1].inputs==5 && c.layer[1].outputs==5);
	CHECK(c.layer[4].inputs==5 && c.layer[4].outputs==5);
	CatsEye__destruct(&c);
}

static void test_batchnorm(void)
{
	CatsEye_layer u[] = { {8, CATS_LINEAR}, {0, CATS_BATCHNORMAL}, {0, CATS_ACT_TANH}, {2, CATS_LINEAR}, {2, CATS_LOSS_MSE} };
	CatsEye c = { .batch=16 };
	xoroshiro128plus_init(7);
	_CatsEye__construct(&c, u, 5);
	CatsEye_layer *bn = &c.layer[1];
	CHECK(bn->r==1);
	real *x = malloc(sizeof(real)*8*16);
	for (int i=0; i<8*16; i++) x[i] = rand_normal(3, 2);
	// training passes feed the running statistics
	for (int k=0; k<200; k++) {
		memcpy(c.layer[0].x, x, sizeof(real)*8*16);
		for (int i=0; i<5; i++) c.layer[i].forward(&c.layer[i]);
	}
	// the batch is normalized: zero mean, unit variance (gamma=1, beta=0)
	real m = 0, v = 0;
	for (int i=0; i<bn->outputs*16; i++) m += bn->z[i];
	m /= bn->outputs*16;
	for (int i=0; i<bn->outputs*16; i++) v += (bn->z[i]-m)*(bn->z[i]-m);
	v /= bn->outputs*16;
	CHECK(fabs(m)<1e-6 && fabs(v-1)<1e-3);
	// single-sample inference uses the running statistics: close to the batch result for the same sample
	real train0[8];
	memcpy(train0, bn->z, sizeof(train0));
	memcpy(c.layer[0].x, x, sizeof(real)*8*16);
	for (int i=0; i<5; i++) c.layer[i].forward(&c.layer[i]);
	real *first_sample_train = malloc(sizeof(real)*8);
	memcpy(first_sample_train, bn->z, sizeof(real)*8);
	CatsEye_forward(&c, x);	// batch=1, p->infer
	real err = 0;
	for (int i=0; i<8; i++) err += fabs(c.layer[1].z[i]-first_sample_train[i]);
	CHECK(err/8 < 0.05);
	CHECK(c.infer==0 && c.batch==16);	// restored
	// gamma/beta and running statistics live in the weights, so they survive save/load
	bn->w[2*bn->r] = 1.5; bn->w[2*bn->r+1] = -0.25;
	CHECK(CatsEye_saveCats(&c, "/tmp/catseye_test_bn.cats")==0);
	bn->w[2*bn->r] = 1; bn->w[2*bn->r+1] = 0; bn->mu[0] = 99;
	CHECK(CatsEye_loadCats(&c, "/tmp/catseye_test_bn.cats")==0);
	CHECK(bn->w[2*bn->r]==1.5 && bn->w[2*bn->r+1]==-0.25 && bn->mu[0]!=99);
	remove("/tmp/catseye_test_bn.cats");
	free(x); free(first_sample_train);
	CatsEye__destruct(&c);
}

static void test_batchnorm_channels(void)
{
	// with batch>1 every channel is normalized over the whole batch (layout [batch][ch][hw])
	CatsEye_layer u[] = { {4*4, CATS_CONV, .ksize=1, .stride=1, .ch=2}, {0, CATS_BATCHNORMAL}, {2, CATS_LINEAR}, {2, CATS_LOSS_MSE} };
	CatsEye c = { .batch=3 };
	_CatsEye__construct(&c, u, 4);
	CatsEye_layer *bn = &c.layer[1];
	CHECK(bn->r==2);
	for (int i=0; i<c.layer[0].inputs*3; i++) c.layer[0].x[i] = rand_normal(0, 1) +(i%3);
	for (int i=0; i<4; i++) c.layer[i].forward(&c.layer[i]);
	for (int ch=0; ch<2; ch++) {
		real m = 0, v = 0;
		for (int b=0; b<3; b++) for (int i=0; i<16; i++) m += bn->z[b*bn->inputs +ch*16 +i];
		m /= 48;
		for (int b=0; b<3; b++) for (int i=0; i<16; i++) { real d = bn->z[b*bn->inputs +ch*16 +i]-m; v += d*d; }
		v /= 48;
		CHECK(fabs(m)<1e-9 && fabs(v-1)<1e-4);
	}
	CatsEye__destruct(&c);
}

static void test_augmentation(void)
{
	enum { W = 8, H = 8 };
	real src[W*H], dst[W*H];
	for (int i=0; i<W*H; i++) src[i] = i;
	// horizontal translation never moves rows (it used to flip the image vertically)
	int rows_ok = 1;
	for (int k=0; k<200; k++) {
		_CatsEye_DataAugmentation(dst, src, W, H, 1, CATS_TRANSLATION_X);
		for (int y=0; y<H; y++) for (int x=0; x<W; x++) if ((int)dst[y*W+x]/W != y) rows_ok = 0;
	}
	CHECK(rows_ok);
	int cols_ok = 1;
	for (int k=0; k<200; k++) {
		_CatsEye_DataAugmentation(dst, src, W, H, 1, CATS_TRANSLATION_Y);
		for (int y=0; y<H; y++) for (int x=0; x<W; x++) if ((int)dst[y*W+x]%W != x) cols_ok = 0;
	}
	CHECK(cols_ok);
	// noise is zero-mean
	real big[10000], out[10000], mean = 0;
	memset(big, 0, sizeof(big));
	_CatsEye_DataAugmentation(out, big, 100, 100, 1, CATS_NOISE);
	for (int i=0; i<10000; i++) mean += out[i];
	CHECK(fabs(mean/10000) < 0.002);
}

static void test_shuffle_buffer(void)
{
	CatsEye_layer u[] = { {4, CATS_LINEAR}, {2, CATS_LINEAR}, {2, CATS_LOSS_MSE} };
	CatsEye c = { .batch=1 };
	_CatsEye__construct(&c, u, 3);
	real x[100*4] = {0}, t[100*2] = {0};
	_CatsEye_data_transfer(&c, x, t, 10);
	_CatsEye_data_transfer(&c, x, t, 100);	// larger data set: the buffer must grow
	int seen[100] = {0};
	for (int i=0; i<100; i++) seen[c.shuffle_buffer[i]]++;
	int perm = 1;
	for (int i=0; i<100; i++) if (seen[i]!=1) perm = 0;
	CHECK(perm);
	CatsEye__destruct(&c);
}

static void test_half_and_q4(void)
{
	// every finite half survives half -> float -> half
	int bad = 0;
	for (unsigned h=0; h<0x10000; h++) {
		if (((h>>10)&0x1f)==0x1f) continue;	// inf / nan
		float f = cats_half_to_float(h);
		if (cats_float_to_half(f)!=h) bad++;
	}
	CHECK(bad==0);
	// Q4_0 round trip stays within a quantization step
	float x[64], q[64];
	uint8_t blk[2*CATS_Q4_0_BYTES];
	for (int i=0; i<64; i++) x[i] = sinf(i*0.7f)*3;
	cats_q4_0_quantize_row(x, 64, blk, 1);
	cats_q4_0_dequantize_row(blk, 64, q);
	float maxd = 0;
	for (int i=0; i<64; i++) maxd = fmaxf(maxd, fabsf(x[i]-q[i]));
	CHECK(maxd < 0.45f);
	// Q4_0 x Q8_0 dot against the float product of the dequantized values
	int8_t a[2*CATS_Q8_0_BYTES];
	float y[64], ref = 0;
	for (int i=0; i<64; i++) y[i] = cosf(i*0.3f);
	cats_q8_0_quantize(a, y, 64);
	for (int i=0; i<64; i++) ref += q[i]*y[i];
	CHECK(fabsf(cats_dot_q4_0_q8_0(blk, a, 64)-ref) < 0.1f);
	// Q8_0 weights (GGUF 34-byte blocks) x Q8_0 activations: dispatch and scalar agree with the float product
	enum { N8 = 96 };
	uint8_t w8[3*CATS_Q8_0W_BYTES];
	float xw[N8], dq[N8], y8[N8], ref8 = 0;
	for (int i=0; i<N8; i++) { xw[i] = sinf(i*0.37f)*(1+i%5); y8[i] = cosf(i*0.21f)*2; }
	for (int b=0; b<3; b++) {
		uint8_t *p = w8 + b*CATS_Q8_0W_BYTES;
		float amax = 0;
		for (int j=0; j<32; j++) amax = fmaxf(amax, fabsf(xw[b*32+j]));
		uint16_t dh = cats_float_to_half(amax/127);
		float d = cats_half_to_float(dh), id = d ? 1/d : 0;
		p[0] = dh&0xff; p[1] = dh>>8;
		for (int j=0; j<32; j++) p[2+j] = (uint8_t)(int8_t)lroundf(xw[b*32+j]*id);
	}
	cats_q8_0w_dequantize_row(w8, N8, dq);
	int8_t a8[3*CATS_Q8_0_BYTES];
	cats_q8_0_quantize(a8, y8, N8);
	for (int i=0; i<N8; i++) { ref8 += dq[i]*y8[i]; CHECK(fabsf(dq[i]-xw[i]) < 0.03f); }
	CHECK(fabsf(cats_dot_q8_0w_q8_0(w8, a8, N8)-ref8) < 0.2f);
	CHECK(fabsf(cats_dot_q8_0w_q8_0_scalar(w8, a8, N8)-cats_dot_q8_0w_q8_0(w8, a8, N8)) < 1e-3f);
}

static void test_gemm_softmax(void)
{
	enum { N = 37, IN = 70, OUT = 21 };
	static float x[N*IN], w[OUT*IN], b[OUT], c[N*OUT];
	for (int i=0; i<N*IN; i++) x[i] = sinf(i*0.13f);
	for (int i=0; i<OUT*IN; i++) w[i] = cosf(i*0.07f);
	for (int i=0; i<OUT; i++) b[i] = i*0.1f;
	for (int level=0; level<=3; level++) {
		const char *names[] = { "scalar", "avx", "avx2", "avx512" };
		setenv("CATS_SIMD", names[level], 1);
		(void)cats_cpu_simd_level();	// cached once per process: only the first level counts, run the check anyway
		cats_gemm_nt(c, x, w, N, IN, OUT, b);
		double worst = 0;
		for (int t=0; t<N; t++) for (int o=0; o<OUT; o++) {
			double s = b[o];
			for (int k=0; k<IN; k++) s += (double)x[t*IN+k]*w[o*IN+k];
			worst = fmax(worst, fabs(s-c[t*OUT+o]));
		}
		CHECK(worst < 1e-3);
	}
	float row[19];
	for (int i=0; i<19; i++) row[i] = i*0.5f -4;
	cats_softmax_scaled(row, 19, 3, 15, 1.0f);
	double sum = 0;
	for (int i=0; i<19; i++) sum += row[i];
	CHECK(fabs(sum-1) < 1e-5 && row[0]==0 && row[2]==0 && row[15]==0 && row[18]==0 && row[14]>row[3]);
}

static void test_safetensors(void)
{
	const char *path = "/tmp/catseye_test.safetensors";
	float f[6] = {1, 2, 3, 4, 5, 6};
	cats_st_entry e[2];
	char longname[190];
	memset(longname, 'n', sizeof(longname)-1);
	longname[sizeof(longname)-1] = 0;	// 189 characters, with 4 dims: used to overflow the header buffer
	e[0] = (cats_st_entry){ "w", "F32", 2, {2, 3}, f, sizeof(f) };
	e[1] = (cats_st_entry){ longname, "F32", 4, {1, 1, 2, 3}, f, sizeof(f) };
	CHECK(cats_st_write(path, e, 2, "pt")==0);
	cats_st s;
	CHECK(cats_st_open(&s, path)==0);
	const cats_st_tensor *t = cats_st_find(&s, "w");
	float back[6] = {0};
	CHECK(t && cats_st_numel(t)==6 && cats_st_read_f32(&s, t, back)==0 && back[5]==6);
	CHECK(cats_st_find(&s, longname)!=NULL);
	cats_st_close(&s);

	// a tensor that points beyond the file must be rejected, not read
	FILE *fp = fopen(path, "rb");
	uint8_t buf[4096];
	size_t n = fread(buf, 1, sizeof(buf), fp);
	fclose(fp);
	const char *needle = "\"data_offsets\":[0,24]";
	char *p = memmem(buf, n, needle, strlen(needle));
	CHECK(p!=NULL);
	if (p) {
		memcpy(p, "\"data_offsets\":[0,99]", strlen(needle));
		fp = fopen(path, "wb");
		fwrite(buf, 1, n, fp);
		fclose(fp);
		CHECK(cats_st_open(&s, path)!=0);
	}
	remove(path);
}

int main(void)
{
	xoroshiro128plus_init(1);
	test_activation_shapes();
	test_batchnorm();
	test_batchnorm_channels();
	test_augmentation();
	test_shuffle_buffer();
	test_half_and_q4();
	test_gemm_softmax();
	test_safetensors();
	printf(g_fail ? "FAILED: %d\n" : "all core tests passed\n", g_fail);
	return g_fail != 0;
}

// End-to-end: train small networks and classify with single-sample inference (CatsEye_forward / predict).
// With batch normalization this only works if inference uses the running statistics, not the sample's own.
#define CATS_QUIET
#define CATS_USE_ADAM
#include "../catseye.h"

enum { N = 512, D = 8 };
static real x[N*D];
static int16_t t[N];

static int run(const char *name, CatsEye_layer *u, int n)
{
	xoroshiro128plus_init(3);
	CatsEye c = { .batch=32 };
	_CatsEye__construct(&c, u, n);
	CatsEye_train(&c, x, t, N, 60, 0, 0);
	int ok = CatsEye_accuracy(&c, x, t, N);
	printf("%-12s accuracy %d/%d\n", name, ok, N);
	CatsEye__destruct(&c);
	return ok >= N*0.95;
}

int main(void)
{
	xoroshiro128plus_init(3);
	for (int i=0; i<N; i++) {	// two well separated clusters
		t[i] = i%2;
		for (int d=0; d<D; d++) x[i*D+d] = rand_normal(t[i] ? 2.5 : 27.5, 1.0);
	}
	int ok = 1;
	{ CatsEye_layer u[] = { {D, CATS_LINEAR, 0.001}, {16, CATS_ACT_TANH}, {16, CATS_LINEAR, 0.001}, {2, CATS_ACT_SOFTMAX}, {2, CATS_SOFTMAX_CE} };
	  ok &= run("mlp", u, 5); }
	{ CatsEye_layer u[] = { {D, CATS_LINEAR, 0.001}, {16, CATS_BATCHNORMAL, 0.001}, {0, CATS_ACT_TANH}, {16, CATS_LINEAR, 0.001}, {2, CATS_ACT_SOFTMAX}, {2, CATS_SOFTMAX_CE} };
	  ok &= run("mlp+bn", u, 6); }
	printf(ok ? "all training tests passed\n" : "FAILED\n");
	return !ok;
}

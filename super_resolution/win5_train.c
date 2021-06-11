//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc win5-rb_train.c -o win5-rb_train -lm -Ofast -fopenmp -lgomp
// clang win5-rb_train.c -o win5-rb_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL` -mf16c

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

//#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_ADAGRAD
//#define CATS_USE_RMSPROP
#define CATS_USE_ADAM
#define ADAM_BETA1	0.9
#define ADAM_BETA2	0.999
#define ETA		5e-5
//#define ETA		1e-4	// ADAM,AdaGrad (batch 1,3)
//#define ETA		3e-4	// ADAM (batch 1)
//#define ETA		1e-7	// SGD
#define BATCH		1
//#define BATCH		3
//#define BATCH		64

#define NAME		"win5"
#define CATS_CHECK
#define CATS_USE_FLOAT
#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"

int main()
{
	const int k = 96;
	const int size = k*k*3;
	const int sample = 946-1;

	// https://qiita.com/phyblas/items/2ad3d70841ca4a888ee4
	CatsEye_layer u[] = {
		{size/4, CATS_CONV, ETA, .ksize=7, .stride=1, .padding=3, .ch=64, .ich=3, .name="Input" },
//		{     0, CATS_BATCHNORMAL, ETA },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=7, .stride=1, .padding=3, .ch=64 },
//		{     0, CATS_BATCHNORMAL, ETA },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=7, .stride=1, .padding=3, .ch=64 },
//		{     0, CATS_BATCHNORMAL, ETA },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=7, .stride=1, .padding=3, .ch=64 },
//		{     0, CATS_BATCHNORMAL, ETA },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=7, .stride=1, .padding=3, .ch=4*3 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3, .name="Output" },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);
	int output = CatsEye_getLayer(&cat, "Output");
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	real *xh = calloc(sizeof(real), k/2*k/2*3*sample);
	int16_t t[sample];

	// 訓練データの読み込み
	printf("Training data: loading...");
	real *x = CatsEye_loadCifar("./animeface.bin", k*k*3, sizeof(int16_t), sample, (int16_t**)&t); // 0-1
	for (int i=0; i<sample*3/*ch*/; i++) {
		stbir_resize_float(x+k*k*i, k, k, 0, xh+k/2*k/2*i, k/2, k/2, 0, 1);
	}
	printf("OK\n");
	/*{
	uint8_t *pixels = calloc(3, size*100);
	for (int i=0; i<10*10; i++) {
		CatsEye_visualize(xh+size/4*i, k/2*k/2, k/2, &pixels[((i/10)*k/2*k*10+(i%10)*k/2)*3], k*10, 3);
		CatsEye_visualize(x+size*i, k*k, k, &pixels[((i/10)*k*k*10+(i%10)*k)*3], k*10, 3);
	}
	stbi_write_png(NAME"_.png", k*10, k*10, 3, pixels, 0);
	}*/

	for (int n=0; n<10; n++) {

	// 訓練
	printf("Starting training...\n");
//	CatsEye_train(&cat, xh, x, sample, 45/*repeat*/, sample/*random batch*/, 0);
//	CatsEye_train(&cat, xh, x, sample, 5/*repeat*/, sample/*random batch*/, 0);
	CatsEye_train(&cat, xh, x, sample, 1/*repeat*/, sample/*random batch*/, 0);
	printf("Training complete\n");
	cat.epoch = n;
	CatsEye_saveCats(&cat, NAME".cats");

	// 結果の表示
	uint8_t *pixels = calloc(3, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<10*10; i++) {
#if BATCH>=4
		cat.batch = 4;
		_CatsEye_forward(&cat);
		CatsEye_visualize(cat.layer[output].z+k*k*3, k*k, k, &pixels[((i/10)*k*k*10+(i%10)*k)*3], k*10, 3);
#else
		CatsEye_forward(&cat, xh+k/2*k/2*3*i);
		CatsEye_visualize(cat.layer[output].z, k*k, k, &pixels[((i/10)*k*k*10+(i%10)*k)*3], k*10, 3);
#endif
	}
	stbi_write_png(NAME"_.png", k*10, k*10, 3, pixels, 0);
	free(pixels);

	}

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

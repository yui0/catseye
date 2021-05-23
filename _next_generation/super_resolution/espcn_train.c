//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc espcn_train.c -o espcn_train -lm -Ofast -fopenmp -lgomp
// clang espcn_train.c -o espcn_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL` -mf16c

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

//#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_ADAGRAD
//#define CATS_USE_RMSPROP
#define CATS_USE_ADAM
#define ADAM_BETA1	0.5
//#define ADAM_BETA1	0.1
#define ADAM_BETA2	0.999
#define ETA		5e-4	// ADAM (batch 1,64)
//#define ETA		1e-4	// ADAM (batch 256)
//#define ETA		1e-3	// ADAM,AdaGrad (batch 1,3)
//#define ETA		5e-3	// ADAM (batch 1) with LeakyReLU
//#define ETA		8e-3	// ADAM (batch 1) with LeakyReLU
//#define ETA		1e-6	// SGD (batch 64), momentumSGD
//#define ETA		5e-6	// SGD (batch 64)
//#define ETA		1e-5	// SGD (batch 1)
#define BATCH		1
//#define BATCH		3
//#define BATCH		64
//#define BATCH		256

#define CATS_CHECK
#define CATS_USE_FLOAT
//#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"

#define NAME	"espcn"

int main()
{
	const int k = 96;
	const int size = k*k*3;
	const int sample = 946-1;

	// https://nykergoto.hatenablog.jp/entry/2019/05/28/%E7%94%BB%E5%83%8F%E3%81%AE%E8%B6%85%E8%A7%A3%E5%83%8F%E5%BA%A6%E5%8C%96%3A_ESPCN_%E3%81%AE_pytorch_%E5%AE%9F%E8%A3%85_/_%E5%AD%A6%E7%BF%92
	// https://qiita.com/nekono_nekomori/items/08ec250ceb09a0004768
	CatsEye_layer u[] = {
		{size/4, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=64, .ich=3 },
//		{size/4, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=64, .ich=3 },
//		{  size, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=64, .ich=3 },
//		{     0, CATS_ACT_TANH },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RELU },
//		{     0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

/*		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{     0, CATS_ACT_TANH },
		{     0, CATS_ACT_RELU },*/

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
//		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=32 },
//		{     0, CATS_ACT_TANH },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RELU },
//		{     0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

//		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=3, .name="Output" },
//		{  size, CATS_LOSS_IDENTITY_MSE },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=4*3 },
//		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4*3 }, // Good!!
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3, .name="Output" },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);
	int output = CatsEye_getLayer(&cat, "Output");

	real *xh = calloc(sizeof(real), k/2*k/2*3*sample);
	int16_t t[sample];
	//uint8_t *data = malloc(sample*size);

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

	// 訓練
	printf("Starting training...\n");
	CatsEye_train(&cat, xh, x, sample, 45/*repeat*/, sample/*random batch*/, 0);
//	CatsEye_train(&cat, xh, x, sample, 5/*repeat*/, sample/*random batch*/, 0);
//	CatsEye_train(&cat, x, x, sample, 5/*repeat*/, sample/*random batch*/, 0);
	printf("Training complete\n");
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

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

//---------------------------------------------------------
//	Cat's eye
//
//		©2018,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast -fopenmp -lgomp
// clang cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CATS_USE_ADAM
#define ADAM_BETA1	0.1
#define ADAM_BETA2	0.999

//#define ETA		1e-4
#define ETA		2e-4
#define BATCH	1
//#define BATCH	128

#define NAME		"cifar_autoencoder2"
#define CATS_CHECK
#define CATS_USE_FLOAT
#include "catseye.h"

int main()
{
	int k = 32;
	int size = 32*32*3;
	int label = 10;
	int sample = 10000;

/*	CatsEye_layer u[] = {
		{  size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .sx=32, .sy=32, .ich=3 },
		{     0, CATS_ACT_LEAKY_RELU, },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
		{     0, CATS_ACT_LEAKY_RELU },

		{     0, CATS_DECONV, ETA, .ksize=2, .stride=2, .padding=0, .ch=3 },
		{  size, CATS_LOSS_IDENTITY_MSE },
	};*/
	CatsEye_layer u[] = {
		{  size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .sx=32, .sy=32, .ich=3 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_DECONV, ETA, .ksize=2, .stride=2, .padding=0, .ch=3 },
		{  size, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = CatsEye_loadCifar("data_batch_1.bin", size, 1, sample, &t);
	free(t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	CatsEye_train(&cat, x, x, sample, 50/*repeat*/, 1000/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	uint8_t *pixels = calloc(1, size*100);
	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);
		CatsEye_visualize(cat.layer[cat.layers-1].x, 32*32, 32, &pixels[(i/10)*size*10+(i%10)*k*3], k*10, 3);
		CatsEye_visualize(x+size*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10, 3);
	}
	stbi_write_png(NAME".png", k*10, k*10, 3, pixels, 0);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

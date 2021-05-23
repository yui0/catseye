//---------------------------------------------------------
//	Cat's eye
//
//		©2018,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast -fopenmp -lgomp
// clang cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast

#define CATS_USE_ADAM
#define ADAM_BETA1	0.1
#define ADAM_BETA2	0.999

#define ETA		1e-3 // nan with epoch 30 over
//#define BATCH	1
#define BATCH	128

#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;

	CatsEye_layer u[] = {
		{  size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=0, .ch=64, .sx=34, .sy=34, .ich=3 },
		{     0, CATS_ACT_LEAKY_RELU, /*.alpha=0.01*/ },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=128 },
		{     0, CATS_ACT_LEAKY_RELU },

		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4*3 },
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3 },
		{  size, CATS_LOSS_MSE },
	};
	CatsEye cat = { .batch=1 };
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
	CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1000/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	uint8_t *pixels = calloc(1, size*100);
	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);
		CatsEye_visualize(cat.layer[cat.layers-1].x, 32*32, 32, &pixels[(i/10)*size*10+(i%10)*k*3], k*10, 3);
		CatsEye_visualize(x+size*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10, 3);
	}
	stbi_write_png("cifar_autoencoder.png", k*10, k*10, 3, pixels, 0);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

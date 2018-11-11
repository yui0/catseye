//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_lsgan.c -o mnist_lsgan -lm -Ofast -fopenmp -lgomp
// clang mnist_lsgan.c -o mnist_lsgan -lm -Ofast

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define BATCH	60000

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

	// https://qiita.com/taku-buntu/items/0093a68bfae0b0ff879d
	CatsEye_layer u[] = {
		// generator
		{     100, CATS_LINEAR, 0.01 },
//		{    1024, _CATS_ACT_RELU },
		{    1024, _CATS_ACT_LEAKY_RELU },
		{       0, CATS_LINEAR, 0.01 },
//		{ 128*7*7, _CATS_ACT_RELU },	// 128 7x7
		{ 128*7*7, _CATS_ACT_LEAKY_RELU },	// 128 7x7

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32, .sx=7, .sy=7 },	// 32 14x14
//		{       0, _CATS_ACT_RELU },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },	// 1 28x28
		{       0, _CATS_ACT_TANH },

		// discriminator
		{   28*28, CATS_CONV, 0.001, .ksize=5, .stride=1/*2*/, .ch=64 },
		{       0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, 0.001, .ksize=5, .stride=1/*2*/, .ch=128 },
		{       0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, 0.001 },
		{     256, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, 0.001 },
		{       1, _CATS_ACT_SIGMOID },
		{       1, CATS_LOSS_MSE },
//		{       2, _CATS_ACT_SIGMOID },
//		{       2, CATS_LOSS_0_1 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);
	real *noise = malloc(sizeof(real)*100*BATCH);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
	printf("OK\n");

//	int lreal[BATCH], lfake[BATCH];
	real lreal[BATCH], lfake[BATCH];
	for (int i=0; i<BATCH; i++) {
//		lreal[i] = 1;
//		lfake[i] = 0;
		lreal[i] = random(0.7, 1.2);
		lfake[i] = random(0.0, 0.3);
	}

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	for (int n=0; n<1000; n++) {
		// Training Discriminator
		cat.start = 9;
		cat.slide = size;
		printf("Training Discriminator [%d]: phase 1 [real]\n", n);
		_CatsEye_train(&cat, x, lreal, BATCH, 1/*repeat*/, 100/*random batch*/, 0);

		for (int i=0; i<100*BATCH; i++) {
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.stop = 9+1;
		cat.slide = 100;
		for (int i=0; i<9; i++) cat.layer[i].fix = 1;
		printf("Training Discriminator [%d]: phase 2 [fake]\n", n);
		_CatsEye_train(&cat, noise, lfake, BATCH, 1/*repeat*/, 100/*random batch*/, 0);
		for (int i=0; i<9; i++) cat.layer[i].fix = 0;
		cat.stop = 0;

		// Training Generater
		for (int i=0; i<100*BATCH; i++) {
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.slide = 100;
		for (int i=9; i<cat.layers; i++) cat.layer[i].fix = 1;
		printf("Training Generater [%d]\n", n);
		_CatsEye_train(&cat, noise, lreal, BATCH, 1/*repeat*/, 200/*random batch*/, 0);
		for (int i=9; i<cat.layers; i++) cat.layer[i].fix = 0;


		// 結果の表示
		unsigned char *pixels = calloc(1, size*100);
		for (int i=0; i</*50*/100; i++) {
//			double mse = 0;
			_CatsEye_forward(&cat, noise+100*i);
//			int p = _CatsEye_predict(&cat, noise+100*i);
//			printf("%d ", p);
			CatsEye_visualize(cat.layer[9].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

//			cat.start = 9;
//			_CatsEye_forward(&cat, x+size*i);
//			CatsEye_visualize(cat.layer[9].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

/*			unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*28];
			for (int j=0; j<size; j++) {
				CatsEye_layer *l = &cat.layer[8];
				mse += (x[size*i+j]-l->x[j])*(x[size*i+j]-l->x[j]);

				//p[5*size*10+(j/28)*28*10+(j%28)] = cat.layer[0].x[j] * 255.0;
				p[5*size*10+(j/28)*28*10+(j%28)] = cat.o[0][j] * 255.0;
			}
			printf("mse %lf\n", mse/size);*/
		}
		printf("\n");
		char buff[256];
		sprintf(buff, "/tmp/mnist_lsgan_%05d.png", n);
		stbi_write_png(buff, 28*10, 28*10, 1, pixels, 28*10);
		free(pixels);
	}
	printf("Training complete\n");

	free(noise);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

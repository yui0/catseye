//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast -fopenmp -lgomp
// clang cifar_autoencoder.c -o cifar_autoencoder -lm -Ofast

//#define CATS_DENOISING_AUTOENCODER
#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;

	// Convolutional AutoEncoder
	/*CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,14,14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,7,7


		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0.01, .ksize=3, .stride=1, .ch=1 },
		{     0, _CATS_ACT_SIGMOID },

		{  size, CATS_LOSS_MSE },
	};*/
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1, .ich=3 },
		{     0, CATS_CONV, 0.001, .ksize=3, .stride=1, .ch=16*3, .sx=34, .sy=34, .ich=3 },
		{     0, _CATS_ACT_LEAKY_RELU, /*.alpha=0.01*/ },
//		{     0, _CATS_ACT_RRELU, .min=0, .max=0.01 },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

//		{     0, CATS_PADDING, .padding=1 },
//		{     0, CATS_CONV, 0.001, .ksize=3, .stride=1, .ch=4*3 },
		{     0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=4*3 },
		{     0, _CATS_ACT_LEAKY_RELU },
//		{     0, _CATS_ACT_RRELU, .min=0, .max=0.01 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3 },
//		{     0, _CATS_ACT_RELU },
//		{     0, _CATS_ACT_SIGMOID },	// anime
//		{     0, _CATS_ACT_SOFTMAX },
		{  size, CATS_LOSS_MSE },
	};
/*	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16, .sx=34, .sy=34, .ich=1 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

//		{     0, CATS_PADDING, .padding=1 },
//		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
		{ 32*32, CATS_LOSS_MSE },
	};*/
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = _CatsEye_loadCifar("../example/data_batch_1.bin", size, 1, sample, &t);
	free(t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	_CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	_CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1000/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	for (int i=0; i<50; i++) {
		_CatsEye_forward(&cat, x+size*i);
//		CatsEye_visualize(cat.layer[cat.layers-1].x, size, k, &pixels[(i/10)*size*10+(i%10)*k], k*10);
//		CatsEye_visualize(cat.layer[cat.layers-1].x, size, k, &pixels[(i/10)*size/3*10+(i%10)*k], k*10);

		_CatsEye_visualize(cat.layer[cat.layers-1].x, 32*32, 32, &pixels[(i/10)*size*10+(i%10)*k*3], k*10, 3);
		_CatsEye_visualize(x+size*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10, 3);
/*		double mse = 0;
		unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*k];
		for (int j=0; j<size; j++) {
			CatsEye_layer *l = &cat.layer[cat.layers-1];
///			p[(j/28)*28*10+(j%28)] = l->x[j] * 255.0;
			mse += (x[size*i+j]-l->x[j])*(x[size*i+j]-l->x[j]);
//			p[(j/28)*28*10+(j%28)] = cat.o[2][j] * 255.0;
//			mse += (x[size*i+j]-cat.o[2][j])*(x[size*i+j]-cat.o[2][j]);

//			p[5*size*10+(j/28)*28*10+(j%28)] = x[size*i+j] * 255.0;
//			p[5*size*10+((j/k)*k*10+(j%k))] = cat.layer[0].x[j] * 255.0;
			p[5*size/3*10+((j/k)*k*10+(j%k))] = cat.layer[0].x[j] * 255.0;
		}
		printf("mse %lf\n", mse/size);*/
	}
	stbi_write_png("cifar_autoencoder.png", k*10, k*10, 3, pixels, 0);
//	stbi_write_png("cifar_autoencoder.png", k*10, k*10, 1, pixels, k*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

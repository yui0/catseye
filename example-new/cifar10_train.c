//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_train.c -o cifar10_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang cifar10_train.c -o cifar10_train -lm -Ofast -march=native -funroll-loops
//#define CATS_SSE
//#define CATS_AVX
#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int k = 32;
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;

/*	int u[] = {
		0, 0, 3, size, 0, 0, 0, 100,				// input 32x32x3, mini batch size is 100 by random

		CATS_CONV, CATS_ACT_LEAKY_RELU, 10, 0, 0, 0, 3, 1,	// CONV1
		CATS_CONV, CATS_ACT_LEAKY_RELU, 10, 0, 0, 0, 3, 1,	// CONV2
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL3 97.7%(1000), 98.5%(3000)
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 256, 0, 0, 0, 0,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;
	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);*/

	CatsEye_layer u[] = {	// 44.7%(100), 73%(1000), 98.6%(2000)
//		{   size, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, .ch=10, .ich=3 },
//		{      0, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, .ch=10 },
		{   size, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, .ch=10, .padding=1, .ich=3 },
		{      0, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, .ch=10, .padding=1 },
		{      0, CATS_MAXPOOL,                  0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_LINEAR, CATS_ACT_LEAKY_RELU,  0.01 },
		{    256, CATS_LINEAR,   CATS_ACT_IDENTITY,  0.01 },
		{  label, CATS_LOSS,         CATS_LOSS_0_1,  0.01 },
	};
	/*CatsEye_layer u[] = {	// 43.9%(100)
		{   size, CATS_CONV,         CATS_ACT_RELU,  0.01, .ksize=3, .stride=1, .ch=10, .ich=3 },
		{      0, CATS_CONV,         CATS_ACT_RELU,  0.01, .ksize=3, .stride=1, .ch=10 },
		{      0, CATS_MAXPOOL,                  0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_LINEAR,       CATS_ACT_RELU,  0.01 },
		{    256, CATS_LINEAR,   CATS_ACT_IDENTITY,  0.01 },
		{  label, CATS_LOSS,         CATS_LOSS_0_1,  0.01 },
	};*/
	/*CatsEye_layer u[] = {	// !!!
		{   size, CATS_CONV,         CATS_ACT_TANH,  0.01, .ksize=3, .stride=1, .ch=10, .ich=3 },
		{      0, CATS_CONV,         CATS_ACT_TANH,  0.01, .ksize=3, .stride=1, .ch=10 },
		{      0, CATS_MAXPOOL,                  0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_LINEAR,       CATS_ACT_TANH,  0.01 },
		{    256, CATS_LINEAR,   CATS_ACT_IDENTITY,  0.01 },
		{  label, CATS_LOSS,         CATS_LOSS_0_1,  0.01 },
	};*/
	/*CNN(
	  (conv1): Conv2d (3, 6, kernel_size=(5, 5), stride=(1, 1))
	  (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	  (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
	  (fc1): Linear(in_features=400, out_features=120)
	  (fc2): Linear(in_features=120, out_features=84)
	  (fc3): Linear(in_features=84, out_features=10)
	)*/
	/*CatsEye_layer u[] = {	// 32.4%(100)
		{   size, CATS_CONV,       CATS_ACT_RELU,  0.01, .ksize=5, .stride=1, .ch=6, .ich=3 },
		{      0, CATS_MAXPOOL,                0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_CONV,       CATS_ACT_RELU,  0.01, .ksize=5, .stride=1, .ch=16 },
		{      0, CATS_LINEAR,     CATS_ACT_RELU,  0.01 },
		{    120, CATS_LINEAR,     CATS_ACT_RELU,  0.01 },
		{     84, CATS_LINEAR, CATS_ACT_IDENTITY,  0.01 },
		{  label, CATS_LOSS,       CATS_LOSS_0_1,  0.01 },
	};*/
	// https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
	/*CatsEye_layer u[] = {	// 33.2%(100)
		{   size, CATS_CONV,       CATS_ACT_RELU,  0.01, .ksize=5, .stride=1, .ch=16, .ich=3 },
		{      0, CATS_MAXPOOL,                0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_CONV,       CATS_ACT_RELU,  0.01, .ksize=5, .stride=1, .ch=20 },
		{      0, CATS_MAXPOOL,                0,  0.01, .ksize=2, .stride=2 },
		//{      0, CATS_CONV,       CATS_ACT_RELU,  0.01, .ksize=5, .stride=1, .ch=20 },
		//{      0, CATS_MAXPOOL,                0,  0.01, .ksize=2, .stride=2 },
//		{      0, CATS_LINEAR,  CATS_ACT_SIGMOID,  0.01 },
		{      0, CATS_LINEAR, CATS_ACT_IDENTITY,  0.01 },
		{  label, CATS_LOSS,       CATS_LOSS_0_1,  0.01 },
	};*/
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = _CatsEye_loadCifar("../example/data_batch_1.bin", 32, 32, 1, sample, &t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, t, sample, 1000/*repeat*/, 0.01);
	_CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 100/*random batch*/);
//	_CatsEye_train(&cat, x, t, sample, 1000/*repeat*/, 100/*random batch*/);
//	_CatsEye_train(&cat, x, t, sample, 2000/*repeat*/, 100/*random batch*/);
	printf("Training complete\n");
//	CatsEye_save(&cat, "cifar10.weights");
//	CatsEye_saveJson(&cat, "cifar10.json");
//	CatsEye_saveBin(&cat, "cifar10.bin");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
//		int p = CatsEye_predict(&cat, x+size*i);
		int p = _CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
//				CatsEye_visualize(x+size*i, size, k*3, &pixels[(c/10)*size*10+(c%10)*k*3], k*3*10);
				real *xx = &x[size*i];
				unsigned char *p = &pixels[(c/10)*size*10+(c%10)*k*3];
				for (int y=0; y<k; y++) {
					for (int x=0; x<k; x++) {
						p[(y*k*10+x)*3  ] = xx[y*k+x] * 255.0;
						p[(y*k*10+x)*3+1] = xx[k*k+y*k+x] * 255.0;
						p[(y*k*10+x)*3+2] = xx[2*k*k+y*k+x] * 255.0;
					}
				}

				//CatsEye_visualize(cat.o[0], k*k, k, &pixels[(c/10)*k*k*10+(c%10)*k], k*10);
//				CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[(c/10)*k*k*10+(c%10)*k], k*10);
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
//	stbi_write_png("cifar10_train_wrong.png", k*10, k*10, 1, pixels, 0);
	stbi_write_png("cifar10_train_wrong.png", k*10, k*10, 3, pixels, 0);

/*	int n[10];
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
//		int p = CatsEye_predict(&cat, x+size*i);
		int p = _CatsEye_predict(&cat, x+size*i);

//		CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[p*k*k*10+(n[p]%10)*k], k*10);
		n[p]++;
	}
	stbi_write_png("cifar10_classify.png", k*10, k*10, 1, pixels, 0);*/

	memset(pixels, 0, size*100);
	/*for (int i=0; i<10*10; i++)*/ {
		int p = _CatsEye_predict(&cat, x/*+size*i*/);

		for (int n=0; n<cat.layers; n++) {
			CatsEye_layer *l = &cat.layer[n];
			if (l->type == CATS_LINEAR) {
				continue;
			}

/*			int mch = l->ich > 10 ? 10 : l->ich;
			for (int ch=0; ch<mch; ch++) {
				unsigned char *p = &pixels[n*(k+2) +ch*(k+2)*k*10];
				for (int y=0; y<l->sy; y++) {
					for (int x=0; x<l->sx; x++) {
//						p[(y*k*10+x)] = l->x[y*l->sx+x +ch*l->sx*l->sy] * 255.0;
						p[(y*k*10+x)] = l->x[y*l->sx+x +ch*l->sx*l->sy] * 200.0;
					}
				}
			}*/

			int mch = l->ch > 10 ? 10 : l->ch;
			for (int ch=0; ch<mch; ch++) {
				unsigned char *p = &pixels[n*(k+2) +ch*(k+2)*k*10];
				for (int y=0; y<l->oy; y++) {
					for (int x=0; x<l->ox; x++) {
//						p[(y*k*10+x)] = l->z[y*l->ox+x +ch*l->ox*l->oy] * 255.0;
						p[(y*k*10+x)] = l->z[y*l->ox+x +ch*l->ox*l->oy] * 200.0;
					}
				}
			}
		}
	}
	stbi_write_png("cifar10_classify.png", k*10, k*10, 1, pixels, 0);

/*	memset(pixels, 0, size*100);
	for (int i=0; i<10; i++) {
		CatsEye_forward(&cat, x+size*i);

		// 初段フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 1, 0, &pixels[i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 1, &pixels[k*k*10+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 2, &pixels[k*k*10*2+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 3, &pixels[k*k*10*3+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 4, &pixels[k*k*10*4+i*k], k*10);

		// 2段目フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 2, 0, &pixels[k*k*10*5+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 1, &pixels[k*k*10*6+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 2, &pixels[k*k*10*7+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 3, &pixels[k*k*10*8+i*k], k*10);
	}
	// フィルタ
	for (int i=0; i<u[CHANNEL+LPLEN]; i++) {
		int n = (u[KSIZE+LPLEN]+2);
		CatsEye_visualizeUnits(&cat, 1, 0, i, &pixels[k*k*10*(9+(i*n)/(k*10))+(i*n)%(k*10)], k*10);
	}
	stbi_write_png("cifar10_train.png", k*10, k*10, 1, pixels, 0);*/

/*	memset(pixels, 0, size*100);
	for (int i=0; i<10; i++) {
		memset(cat.o[layers-1], 0, label);
		cat.o[layers-1][i] = 1;
		CatsEye_backpropagate(&cat, layers-2);

		CatsEye_visualizeUnits(&cat, 0, 1, 0, &pixels[i*k], k*10);
	}
	stbi_write_png("cifar10_gen.png", k*10, k*10, 1, pixels, k*10);*/
	free(pixels);

	free(t);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

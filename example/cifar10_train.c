//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
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

#if 0
	int u[] = {	// 67.7% (http://aidiary.hatenablog.com/entry/20151108/1446952402)
//		0, 0, 3, size, 0, 0, 0, 100,			// input 32x32x3, mini batch size is 100 by random
		0, 0, 1, 32*32, 0, 0, 0, 100,			// input 32x32x3, mini batch size is 100 by random

		CATS_CONV, CATS_ACT_RELU, 32, 0, 0, 0, 5/*3*/, 1,	// CONV1 32ch k3
		CATS_MAXPOOL, 0, 32, 0, 0, 0, 2, 2,

//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 7200, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#endif
#if 0
	// Network in Network
/*	int u[] = {
		0, 0, 3, size, 0, 0, 0, 200,				// input 32x32x3, mini batch size is 200 by random

		CATS_CONV, CATS_ACT_LEAKY_RELU, 48, 0, 0, 0, 3, 1,	// CONV1
		CATS_CONV, CATS_ACT_RELU, 10, 0, 0, 0, 1, 1,		// CCCP1
		CATS_CONV, CATS_ACT_LEAKY_RELU, 64, 0, 0, 0, 3, 1,	// CONV2
		CATS_CONV, CATS_ACT_RELU, 8, 0, 0, 0, 1, 1,		// CCCP2 92.3%(1000)

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};*/
/*	int u[] = {
		0, 0, 3, size, 0, 0, 0, 200,				// input 32x32x3, mini batch size is 200 by random

		CATS_CONV, CATS_ACT_LEAKY_RELU, 64, 0, 0, 0, 3, 1,	// CONV1 93.8%(1000)
		CATS_CONV, CATS_ACT_RELU, 8, 0, 0, 0, 1, 1,		// CCCP2
		CATS_CONV, CATS_ACT_LEAKY_RELU, 96, 0, 0, 0, 3, 1,	// CONV3
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL4 96.5%(1000)

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};*/
	int u[] = {
		0, 0, 3, size, 0, 0, 0, 100,				// input 32x32x3

		// N4 https://papers.nips.cc/paper/5636-recursive-training-of-2d-3d-convolutional-networks-for-neuronal-boundary-prediction.pdf
		CATS_CONV, CATS_ACT_LEAKY_RELU, 48, 0, 0, 0, 4, 1,	// CONV1
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL1
		CATS_CONV, CATS_ACT_LEAKY_RELU, 48, 0, 0, 0, 5, 1,	// CONV2
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL2
		CATS_CONV, CATS_ACT_LEAKY_RELU, 48, 0, 0, 0, 4, 1,	// CONV3 66.0%(1000)
//		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL3
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 48, 0, 0, 0, 4, 1,	// CONV4
//		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL4
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 20, 0, 0, 0, 3, 1,	// CONV5
//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#else
	int u[] = {
		0, 0, 3, size, 0, 0, 0, 100,				// input 32x32x3, mini batch size is 100 by random

		// http://ai-programming.hatenablog.jp/entry/2016/03/04/160243
/*		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV1
		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV2
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL3
		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV4
		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV5
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL6
		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV7 50.8%(1000)
		//CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,	// CONV8
		//CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL9
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 512, 0, 0, 0, 0,*/

		// http://qiita.com/nzw0301/items/c7a11baba0f2e029695e
		CATS_CONV, CATS_ACT_LEAKY_RELU, 10, 0, 0, 0, 3, 1,	// CONV1
		CATS_CONV, CATS_ACT_LEAKY_RELU, 10, 0, 0, 0, 3, 1,	// CONV2
		CATS_MAXPOOL, 0, 0, 0, 0, 0, 2, 2,			// POOL3 97.7%(1000), 98.5%(3000)
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 256/*512*/, 0, 0, 0, 0,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#endif
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	// 訓練データの読み込み
	printf("Training data:\n");
	int *t;
	numerus *x = CatsEye_loadCifar("data_batch_1.bin", sample, &t);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 1000/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "cifar10.weights");
	CatsEye_saveJson(&cat, "cifar10.json");
//	CatsEye_saveBin(&cat, "cifar10.bin");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
//				CatsEye_visualize(x+size*i, size, k*3, &pixels[(c/10)*size*10+(c%10)*k*3], k*3*10);
				numerus *xx = &x[size*i];
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

	int n[10];
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
//		if (p==t[i]) {
//			p--;
			CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[p*k*k*10+(n[p]%10)*k], k*10);
			//CatsEye_visualize(x+size*i, 28*28, 28, &pixels[(c/10)*k*k*10+(c%10)*28], k*10);
			n[p]++;
//		}
	}
	stbi_write_png("cifar10_classify.png", k*10, k*10, 1, pixels, 0);

	memset(pixels, 0, size*100);
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
	stbi_write_png("cifar10_train.png", k*10, k*10, 1, pixels, 0);

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

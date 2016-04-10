//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_train.c -o cifar10_train -lm -Ofast -fopenmp -lgomp
// clang cifar10_train.c -o cifar10_train -lm -Ofast
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int k = 32;
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 1000;//10000;

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
#if 1
	int u[] = {	//
		0, 0, 3, size, 0, 0, 0, 100,			// input 32x32x3, mini batch size is 100 by random
//		0, 0, 1, 32*32, 0, 0, 0, 100,			// input 32x32x3, mini batch size is 100 by random

		CATS_CONV, CATS_ACT_ELU, 32, 0, 0, 0, 5, 1,		// CONV1 32ch k5, only 96.6%
		CATS_CONV, CATS_ACT_ELU, 64, 0, 0, 0, 3, 1,

//		CATS_CONV, CATS_ACT_LEAKY_RELU, 16, 0, 0, 0, 3, 1,	// CONV1 32ch k3 97.9%
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 16, 0, 0, 0, 3, 1,	// CONV2 32ch k3

//		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 5, 1,	// CONV1 32ch k5, only 97.3%
		//CATS_MAXPOOL, 0, 32, 0, 0, 0, 2, 2,
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 64, 0, 0, 0, 3, 1,	// CONV2 32ch k3, only 99.1%
		//CATS_MAXPOOL, 0, 64, 0, 0, 0, 2, 2,			// 96.1%
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 128, 0, 0, 0, 3, 1,	// CONV1 32ch k3
//		CATS_MAXPOOL, 0, 128, 0, 0, 0, 2, 2,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 256, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#else
	int u[] = {
		0, 0, 1, 32*32, 0, 0, 0, 100,			// input 32x32x3, mini batch size is 100 by random

//		CATS_CONV, CATS_ACT_RELU, 32, 0, 0, 0, 5, 1,	// CONV1 32ch k5
		CATS_CONV, CATS_ACT_TANH, 32, 0, 0, 0, 5, 1,	// CONV1 32ch k5
		CATS_MAXPOOL, 0, 32, 0, 0, 0, 2, 2,
//		CATS_CONV, CATS_ACT_RELU, 64, 0, 0, 0, 3, 1,	// CONV2 64ch k3
//		CATS_CONV, CATS_ACT_TANH, 64, 0, 0, 0, 3, 1,	// CONV2 64ch k3
//		CATS_MAXPOOL, 0, 64, 0, 0, 0, 2, 2,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 512, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#endif
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	int t[sample+1];					// ラベルデータ
	double *x = malloc(sizeof(double)*size*(sample+1));	// 訓練データ
	unsigned char *data = malloc((sample+1)*size);

//	double table[256];
//	for (int i=0; i<256; i++) table[i] = i/255.0;

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("data_batch_1.bin", "rb");
	if (fp==NULL) return -1;
	fread(data, (size+1)*sample, 1, fp);
//	unsigned char *p = data;
//	double *xx = x;
	for (int n=0; n<sample; n++) {
		t[n] = data[n*(size+1)];
		for (int i=0; i<size; i++) x[n*size+i] = data[n*(size+1)+i] * (1.0/255.0);
//		t[n] = *p++;
//		for (int i=0; i<size; i++) *xx++ = table[*p++];

//		if (n%100) { printf("."); fflush(stdout); }
	}
	fclose(fp);
	free(data);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist.weights");
//	CatsEye_saveJson(&cat, "cifar10_cnn_train.json");
//	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
				//CatsEye_visualize(cat.o[0], 28*28, 28, &pixels[(c/10)*k*k*10+(c%10)*28], k*10);
				CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[(c/10)*k*k*10+(c%10)*k], k*10);
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png("cifar10_train_wrong.png", k*10, k*10, 1, pixels, k*10);

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
	stbi_write_png("cifar10_classify.png", k*10, k*10, 1, pixels, k*10);

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
	stbi_write_png("cifar10_train.png", k*10, k*10, 1, pixels, k*10);

	memset(pixels, 0, size*100);
	for (int i=0; i<10; i++) {
		memset(cat.o[layers-1], 0, label);
		cat.o[layers-1][i] = 1;
		CatsEye_backpropagate(&cat, layers-2);

		CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[i*k], k*10);
	}
	stbi_write_png("cifar10_gen.png", k*10, k*10, 1, pixels, k*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

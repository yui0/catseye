//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_train.c -o mnist_cnn_train -lm -fopenmp -lgomp
// clang mnist_cnn_train.c -o mnist_cnn_train -lm
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 784;	// 入力層ユニット(28x28)
	int hidden = 64;	// 隠れ層ユニット
	int label = 10;	// 出力層ユニット(0-9)
	int sample = 60000;

	int k = 4;		// カーネルサイズ
	int s = 28-k;		// 出力サイズ

	int u[] = {
		0, 0, 1, size,    0, 0, 0, 0,

//		1, 2, 1, 25*25, 28, 28, 3, 1,
//		1, 2, 5, 5*25*25, 28, 28, 3, 1,				// sigmoid, 5ch
//		1, 3, 5, 5*25*25, 28, 28, 3, 1,				// tanh, 5ch, 3x3
		CATS_CONV, CATS_ACT_TANH, 5, 5*s*s, 28, 28, k, 1,	// tanh, 5ch
//		1, 5, 5, 5*25*25, 28, 28, 3, 1,				// ReLU, 5ch

		CATS_MAXPOOL, 0, 5, 5*12*12, s, s, 2, 1,		// maxpooling

//		0, 2, 1, hidden,  0, 0, 0, 0,

		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
//	CatsEye__construct(&cat, size, hidden, label, 0);
	CatsEye__construct(&cat, 0, 0, layers, u);

	double *x = malloc(sizeof(double)*size*sample);	// 訓練データ
	int t[sample];			// ラベルデータ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist.weights");
//	CatsEye_saveJson(&cat, "mnist.json");
//	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	unsigned char *pixels = calloc(1, size*100);
	for (int i=0; i<10; i++) {
		CatsEye_forward(&cat, x+size*i);
		CatsEye_visualize(cat.o[1], s*s, s, &pixels[i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s], s*s, s, &pixels[28*28*10+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*2], s*s, s, &pixels[28*28*10*2+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*3], s*s, s, &pixels[28*28*10*3+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*4], s*s, s, &pixels[28*28*10*4+i*28], 28*10);

		CatsEye_visualize(cat.o[2], 12*12, 12, &pixels[28*28*10*5+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][12*12], 12*12, 12, &pixels[28*28*10*6+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][12*12*2], 12*12, 12, &pixels[28*28*10*7+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][12*12*3], 12*12, 12, &pixels[28*28*10*8+i*28], 28*10);
	}
/*	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);
//		CatsEye_visualize(cat.o[1], cat.u[LPLEN*1+SIZE], 28, &pixels[(i/10)*28*28*10 + (i%10)*28], 28*10);
		CatsEye_visualize(cat.o[1], 25*25, 25, &pixels[(i/10)*28*28*10 + (i%10)*28], 28*10);
	}*/
	CatsEye_visualize(cat.w[0], 5*k*k, k, &pixels[28*10*28*9], 28*10);
	stbi_write_png("mnist_cnn_train.png", 28*10, 28*10, 1, pixels, 28*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc dae.c -o dae -lm -Ofast -fopenmp -lgomp
// clang dae.c -o dae -lm -Ofast
// http://joisino.hatenablog.com/entry/2015/09/09/224157
#define CATS_AUTOENCODER
#define CATS_DENOISING_AUTOENCODER
#define CATS_SIGMOID_CROSSENTROPY
//#define CATS_LOSS_MSE
#include "../catseye.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int hidden = 128;
	int sample = 60000;

	int u[] = {
		0, 0, 1, size, 0, 0, 0, 1500,	// mini batch size is 1500 by random
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, hidden, 0, 0, 0, 0,
//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, size, 0, 0, 0, 1,	// use mse
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, size, 0, 0, 0, 2,	// sparse auto encoder
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	double *x = malloc(sizeof(double)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	free(data);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 0.01);
	printf("Training complete\n");
	CatsEye_save(&cat, "dae.weights");
//	CatsEye_saveJson(&cat, "dae.json");

	// 結果の表示
	unsigned char *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		CatsEye_forward(&cat, x+size*i);

		CatsEye_visualize(cat.o[2], size, 28, &pixels[(i/10)*size*10 + (i%10)*28], 28*10);
		CatsEye_visualize(&x[size*i], size, 28, &pixels[5*size*10+(i/10)*size*10 + (i%10)*28], 28*10);
//		CatsEye_visualizeUnits(&cat, 0, 2, 0, p, 28*10);
//		CatsEye_visualizeUnits(&cat, 0, 0, 0, &p[5*size*10], 28*10);
		for (int j=0; j<size; j++) {
			mse += (x[size*i+j]-cat.o[2][j])*(x[size*i+j]-cat.o[2][j]);
		}
		printf("mse %lf\n", mse/size);
	}
	stbi_write_png("dae.png", 28*10, 28*10, 1, pixels, 28*10);

	memset(pixels, 0, 28*28*100);
	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 28, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
//		CatsEye_visualizeUnits(&cat, 1, 0, 0, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
	}
	stbi_write_png("dae_weights.png", 28*10, 28*10, 1, pixels, 28*10);

	srand((unsigned)(time(0)));
	memset(pixels, 0, 28*28*100);
	for (int i=0; i<100; i++) {
		for (int n=0; n<hidden; n++) {
			cat.o[1][n] = (rand()/(RAND_MAX+1.0)) * 1.0;
//			cat.o[1][n] = rand();
		}
		//CatsEye_layer_forward[u[LPLEN+ACT]](cat.o[1], cat.w[1], cat.z[1], cat.o[2], &u[LPLEN*(1+1)]);
		CatsEye_propagate(&cat, 1);

		CatsEye_visualize(cat.o[2], size, 28, &pixels[(i/10)*size*10 + (i%10)*28], 28*10);
	}
	stbi_write_png("dae_gen.png", 28*10, 28*10, 1, pixels, 28*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

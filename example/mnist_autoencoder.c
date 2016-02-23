//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast -fopenmp -lgomp
// clang mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast
//#define CATS_AUTOENCODER
#define CATS_LOSS_MSE
//#define CATS_OPT_ADAGRAD
//#define CATS_OPT_ADAM
//#define CATS_OPT_RMSPROP
//#define CATS_OPT_RMSPROPGRAVES
//#define CATS_OPT_MOMENTUM
#include "../catseye.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int hidden = 64;	// 隠れ層
	//int hidden = 500;	// 隠れ層
	int sample = 60000;

	CatsEye cat;
	CatsEye__construct(&cat, size, hidden, size, 0);

	double *x = malloc(sizeof(double)*size*sample);	// 訓練データ
//	int t[sample];			// ラベルデータ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	/*fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);*/
	free(data);

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, x, sample-1, 100, 1e-5);		// SGD[h64/20.0]
//	CatsEye_train(&cat, x, x, sample-1, 500, 1e-5);		// SGD[h64/15.0]
	CatsEye_train(&cat, x, x, sample-1, 1500, 1e-5);	// SGD[h64/15.0]
//	CatsEye_train(&cat, x, x, sample-1, 5500, 1e-5);	// SGD[h64/3.9]
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist_autoencoder.weights");
//	CatsEye_saveJson(&cat, "mnist_autoencoder.json");

	// 結果の表示
	unsigned char *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		CatsEye_forward(&cat, x+size*i);
		unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*28];
		for (int j=0; j<size; j++) {
			p[(j/28)*28*10+(j%28)] = cat.o3[j] * 255.0;
			mse += (x[size*i+j]-cat.o3[j])*(x[size*i+j]-cat.o3[j]);

			p[5*size*10+(j/28)*28*10+(j%28)] = x[size*i+j] * 255.0;
		}
		printf("mse %lf\n", mse);
	}
	stbi_write_png("mnist_autoencoder.png", 28*10, 28*10, 1, pixels, 28*10);

	// 重みをスケーリング
/*	double max = cat.w1[0];
	double min = cat.w1[0];
	for (int i=0; i<(size+1)*hidden; i++) {
		if (max < cat.w1[i]) max = cat.w1[i];
		if (min > cat.w1[i]) min = cat.w1[i];
	}
	for (int i=0; i<(size+1)*hidden; i++) {
		cat.w1[i] = (cat.w1[i] - min) / (max - min);
		//cat.w1[i] *= 255.0;
		pixels[i] = cat.w1[i] * 255.0;
	}
	stbi_write_png("autoencoder_weights.png", size+1, hidden, 1, pixels, size+1);*/
	free(pixels);

	return 0;
}

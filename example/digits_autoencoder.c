//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc digits_autoencoder.c -o digits_autoencoder -lm -Ofast -fopenmp -lgomp
// clang digits_autoencoder.c -o digits_autoencoder -lm -Ofast
//#define CATS_AUTOENCODER
//#define CATS_SIGMOID_CROSSENTROPY
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
	int size = 64;		// 入出力層(8x8)
	int hidden = 16;	// 隠れ層
//	int hidden = 32;	// 隠れ層
//	int hidden = 48;	// 隠れ層
//	int hidden = 56;	// 隠れ層
//	int hidden = 64;	// 隠れ層
	int sample = 1797;

	CatsEye cat;
	CatsEye__construct(&cat, size, hidden, size, 0);

	double x[size*sample];	// 訓練データ
	int t[sample];		// ラベルデータ

	// CSVの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("digits.csv", "r");
	if (fp==NULL) return -1;
	int n=0;
	while (feof(fp)==0) {
		// データの読み込み
		if (n<3) printf("[%4d]  ", n);
		for (int j=0; j<size; j++) {
			if (!fscanf(fp, "%lf,", x+size*n+j)) {
				// 0-1に正規化
				x[size*n+j] /= 16.0;
			}
			if (n<3) printf("%6.2f  ", x[size*n+j]);
		}
		fscanf(fp, "%d", t+n);
		if (n<3) printf("<%d>\n", t[n]);
		n++;
	}
	fclose(fp);
	sample = n;

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	//CatsEye_train(&cat, x, x, sample-1, 1000/*repeat*/, 0.004);
//	CatsEye_train(&cat, x, x, 1, 50/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 2, 1000/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 3, 1000/*repeat*/, 0.001);
//	CatsEye_train(&cat, x, x, 10, 1000, 0.001);		// Success by SGD[0.0]
	//CatsEye_train(&cat, x, x, 50, 1000000, 1e-5);		// SGD[3.0], Momentum[9.0], Adam[96], SGD[h48/0.06]
//	CatsEye_train(&cat, x, x, 100, 100000, 1e-5);		// SGD[h48/4.5]
//	CatsEye_train(&cat, x, x, sample-1, 10000, 0.01);	// AdaGrad[127]
//	CatsEye_train(&cat, x, x, sample-1, 10000, 1e-4);	// Adam, RMSpropGraves, SGD[89.0]
	//CatsEye_train(&cat, x, x, 50, 30000, 1e-3);		// AdaGrad[h48/7.2]
//	CatsEye_train(&cat, x, x, 50, 100000, 1e-4);		// SGD[h48/0.25], SGD[h56/0.002], SGD[h64/0.000013]
	//CatsEye_train(&cat, x, x, 50, 1900, 1e-2);		// SGD[h48+sigmoid/0.000099]
	//CatsEye_train(&cat, x, x, sample-1, 1900, 1e-3);	// SGD[h48+sigmoid/0.000063]
	//CatsEye_train(&cat, x, x, sample-1, 5500, 1e-4);	// SGD[h48+sigmoid/0.0029]
	CatsEye_train(&cat, x, x, sample-1, 10000, 1e-5);	// SGD[27.0], SGD[h56/4.6], SGD[h64/3.5], SGD[h16/128]
	//CatsEye_train(&cat, x, x, sample-1, 500, 1e-4);	// SGD[52.3]
	//CatsEye_train(&cat, x, x, sample-1, 30000, 1e-5);	// SGD[h48/8.8], SGD[h48+sigmoid/0.0047]
	//CatsEye_train(&cat, x, x, sample-1, 10000, 1e-6);	// SGD[78.0], SGD/tanh[51.4]
	printf("Training complete\n");
//	CatsEye_save(&cat, "digits_autoencoder.weights");
//	CatsEye_saveJson(&cat, "digits_autoencoder.json");

	// 結果の表示
	unsigned char *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		CatsEye_forward(&cat, x+size*i);
		unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*8];
		for (int j=0; j<size; j++) {
			p[(j/8)*8*10+(j%8)] = cat.o3[j] * 255.0;
			mse += (x[size*i+j]-cat.o3[j])*(x[size*i+j]-cat.o3[j]);

			p[5*size*10+(j/8)*8*10+(j%8)] = x[size*i+j] * 255.0;
		}
		printf("mse %lf\n", mse);
	}
	stbi_write_png("digits_autoencoder.png", 8*10, 8*10, 1, pixels, 8*10);

	for (int n=0; n<10/*hidden*/; n++) {
		// 重みをスケーリング
//		double *w = &cat.w1[n*(size+1)];
		double *w = &cat.w1[n];
		double max = w[0];
		double min = w[0];
		for (int i=1; i<size; i++) {
			if (max < w[i *hidden]) max = w[i *hidden];
			if (min > w[i *hidden]) min = w[i *hidden];
		}
		//printf("%lf %lf\n", max, min);
		for (int i=0; i<size; i++) {
			pixels[n*size + (i/8)*8 + i%8] = ((w[i *hidden] - min) / (max - min)) * 255.0;
		}
	}
	stbi_write_png("digits_autoencoder_weights.png", 8, 8*10/*hidden*/, 1, pixels, /*size+1*/8);
	free(pixels);

	return 0;
}

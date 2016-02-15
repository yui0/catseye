//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc autoencoder.c -o autoencoder -lm -fopenmp -lgomp
// clang autoencoder.c -o autoencoder -lm
#define CATS_LOSS_MSE
#include "../catseye.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 64;		// 入出力層(8x8)
	int hidden = 32;	// 隠れ層
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
//	CatsEye_train(&cat, x, x, sample-1, 100/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 1, 50/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 2, 1000/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 3, 1000/*repeat*/, 0.001);
	CatsEye_train(&cat, x, x, 10, 10000/*repeat*/, 0.001);
	printf("Training complete\n");
//	CatsEye_save(&cat, "digits.weights");
//	CatsEye_saveJson(&cat, "digits.json");

	// 結果の表示
/*	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);*/

	for (int i=0; i<10/*sample*/; i++) {
		double mse = 0;
		unsigned char pixels[size*2];
		CatsEye_forward(&cat, x+size*i);
		for (int j=0; j<size; j++) {
			pixels[j] = cat.o3[j] * 255.0;
			mse += (x[size*i+j]-cat.o3[j])*(x[size*i+j]-cat.o3[j]);

			pixels[size+j] = x[size*i+j] * 255.0;
		}
		char name[256];
		snprintf(name, 256, "autoencoder_%02d.png", i);
		stbi_write_png(name, 8, 8*2, 1, pixels, 8);
		printf("mse %lf\n", mse);
	}

	// 重みをスケーリング
	unsigned char *pixels = malloc((size+1)*hidden);
	double max = cat.w1[0];
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

//	unsigned char *pixels;
//	int width, height, bpp = 1;
//	pixels = malloc(width*bpp);
//	stbi_write_png("autoencoder.png", width, height, bpp, pixels, width*bpp);
	stbi_write_png("autoencoder.png", size+1, hidden, 1, pixels, size+1);
	free(pixels);

	return 0;
}

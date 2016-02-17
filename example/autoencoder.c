//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc autoencoder.c -o autoencoder -lm -Ofast -fopenmp -lgomp
// clang autoencoder.c -o autoencoder -lm -Ofast
#define CATS_LOSS_MSE
#define CATS_ADAGRAD
//#define CATS_ADAM
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
	//CatsEye_train(&cat, x, x, sample-1, 1000/*repeat*/, 0.004);
//	CatsEye_train(&cat, x, x, 1, 50/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 2, 1000/*repeat*/, 0.01);
//	CatsEye_train(&cat, x, x, 3, 1000/*repeat*/, 0.001);
//	CatsEye_train(&cat, x, x, 10, 10000/*repeat*/, 0.001);
	//CatsEye_train(&cat, x, x, 10, 100000/*repeat*/, 0.01);
	//CatsEye_train(&cat, x, x, 50, 1000000/*repeat*/, 0.01);
	CatsEye_train(&cat, x, x, sample-1, 10000/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "autoencoder.weights");
//	CatsEye_saveJson(&cat, "autoencoder.json");

	// 結果の表示
	unsigned char *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		CatsEye_forward(&cat, x+size*i);
		for (int j=0; j<size; j++) {
			pixels[(i/10)*8*8*10+(i%10)*8+(j/8)*8*10+(j%8)] = cat.o3[j] * 255.0;
			mse += (x[size*i+j]-cat.o3[j])*(x[size*i+j]-cat.o3[j]);

			pixels[5*8*8*10+(i/10)*8*8*10+(i%10)*8+(j/8)*8*10+(j%8)] = x[size*i+j] * 255.0;
		}
		printf("mse %lf\n", mse);
	}
	stbi_write_png("autoencoder.png", 8*10, 8*10, 1, pixels, 8*10);

	// 重みをスケーリング
//	unsigned char *pixels = malloc((size+1)*hidden);
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
	stbi_write_png("autoencoder_weights.png", size+1, hidden, 1, pixels, size+1);
	free(pixels);

	return 0;
}

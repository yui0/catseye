//---------------------------------------------------------
//	Cat's eye
//
//		©2016,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast -fopenmp -lgomp
// clang mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast

#define CATS_USE_FLOAT
#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define ETA 1e-3
//#define ETA 1e-5 // ぼやける

int main()
{
	const int wh = 28;
	const int size = 28*28;	// 入出力層(28x28)
	const int sample = 60000;

/*	CatsEye_layer u[] = {	// eta 1e-5 strange!!
		{ size, CATS_LINEAR, ETA },
		{   64, CATS_ACT_TANH },
		{   64, CATS_LINEAR, ETA },
		{ size, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};*/
	// https://www.sejuku.net/blog/63331
	CatsEye_layer u[] = {	// epoch 20/ eta 1e-3
		{ size, CATS_LINEAR, ETA },
		{   64, CATS_ACT_RELU },
		{   64, CATS_LINEAR, ETA },
		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};
//	CatsEye cat = { .batch=1 };	// 0.0%
	CatsEye cat = { .batch=256 };	// 0.0%
	CatsEye__construct(&cat, u);

//	int16_t t[sample];	// ラベルデータ
	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	uint8_t *data = malloc(sample*size);

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
	CatsEye_train(&cat, x, x, sample, 20/*epoch*/, sample, 0);
//	CatsEye_train(&cat, x, x, sample-1, 100, 1e-2);		// SGD[h64/3.3], SGD[h64+s/7.4/OK]
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist_autoencoder.weights");
//	CatsEye_saveJson(&cat, "mnist_autoencoder.json");

	// 結果の表示
	uint8_t *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);

		CatsEye_layer *l = &cat.layer[cat.end-1];
		double mse = 0;
		uint8_t *p = &pixels[(i/10)*size*10 + (i%10)*wh];
		for (int j=0; j<size; j++) {
			p[(j/wh)*wh*10+(j%wh)] = l->z[j] * 255.0;
			mse += (x[size*i+j]-l->z[j])*(x[size*i+j]-l->z[j]);

			p[5*size*10+(j/wh)*wh*10+(j%wh)] = x[size*i+j] * 255.0;
		}
//		printf("%d mse %lf\n", t[i], mse);
	}
	stbi_write_png("mnist_autoencoder.png", wh*10, wh*10, 1, pixels, wh*10);

	memset(pixels, 0, size*100);
/*	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 28, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
	}
	stbi_write_png("mnist_autoencoder_weights.png", 28*10, 28*10, 1, pixels, 28*10);*/
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

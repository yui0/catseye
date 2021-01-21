//---------------------------------------------------------
//	Cat's eye
//
//		©2016,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc digits_autoencoder.c -o digits_autoencoder -lm -Ofast -fopenmp -lgomp
// clang digits_autoencoder.c -o digits_autoencoder -lm -Ofast

#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#define ETA 0.01	// batch 1
//#define ETA 0.001
//#define ETA 0.0001
//#define ETA 0.00003
#define ETA 1e-3
//#define ETA 3e-5
//#define ETA 1e-5
//#define ETA 1e-8

int main()
{
	int size = 64;		// 入出力層(8x8)
	int sample = 1797;

/*	CatsEye_layer u[] = {
		{ size, CATS_LINEAR, ETA },
//		{   40, CATS_ACT_TANH },
		{   48, CATS_ACT_SIGMOID },
		{   48, CATS_LINEAR, ETA },
//		{ size, CATS_ACT_TANH },
//		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};*/
/*	CatsEye_layer u[] = {
		{ size, CATS_LINEAR, ETA },
		{ 400, CATS_ACT_TANH },
//		{ 400, CATS_ACT_SIGMOID },
		{ 400, CATS_LINEAR, ETA },
		{ 400, CATS_ACT_TANH },
//		{ 400, CATS_ACT_SIGMOID },
		{ 400, CATS_LINEAR, ETA },
		{ size, CATS_ACT_TANH },
//		{ 400, CATS_ACT_LEAKY_RELU },
//		{ 400, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};*/
	// https://cntk.ai/pythondocs/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.html
#if 0
	CatsEye_layer u[] = {
		{ size, CATS_LINEAR, ETA },
//		{  400, CATS_ACT_TANH }, // 0.01
//		{  400, CATS_LINEAR, ETA },
/*		{  100, CATS_ACT_TANH }, // 0.001
		{  100, CATS_LINEAR, ETA },
		{ size, CATS_ACT_TANH },*/

/*		{   48, CATS_ACT_RELU },
		{   48, CATS_LINEAR, ETA },
		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_CROSS_ENTROPY },*/

//		{   64, CATS_ACT_RELU }, // 1e-5 (10000)
/*		{   64, CATS_ACT_TANH },
		{   64, CATS_LINEAR, ETA },*/
//		{ size, CATS_ACT_TANH },

		{   32, CATS_ACT_TANH },
		{   32, CATS_LINEAR, ETA },
		{ size, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE

//		{   32, CATS_ACT_RELU },
//		{   32, CATS_LINEAR, ETA },
/*		{   16, CATS_ACT_RELU },
		{   16, CATS_LINEAR, ETA },
//		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },*/
	};
#endif
	CatsEye_layer u[] = {	// epoch 20/ eta 1e-5
		{ size, CATS_LINEAR, ETA },
		{    3, CATS_ACT_RELU },
		{    3, CATS_LINEAR, ETA },
		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};
	CatsEye cat = { .batch=1 };	// 0.0%
	CatsEye__construct(&cat, u);

	real x[size*sample];	// 訓練データ
	int16_t t[sample];	// ラベルデータ

	// CSVの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("digits.csv", "r");
	if (fp==NULL) return -1;
	int n=0;
	while (feof(fp)==0) {
		// データの読み込み
		if (n<3) printf("[%4d]  ", n);
		for (int j=0; j<size; j++) {
#ifdef CATS_USE_FLOAT
			if (!fscanf(fp, "%f,", x+size*n+j)) {
#else
			if (!fscanf(fp, "%lf,", x+size*n+j)) {
#endif
				// 0-1に正規化
				x[size*n+j] /= 16.0;
			}
			if (n<3) printf("%6.2f  ", x[size*n+j]);
		}
		fscanf(fp, "%hd", t+n);
		if (n<3) printf("<%d>\n", t[n]);
		n++;
	}
	fclose(fp);
	printf("\n");
	sample = n;

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, x, sample, 100/*epoch*/, sample, /*sample/10*/0);
//	CatsEye_train(&cat, x, x, sample, 1000/*epoch*/, sample, 0);
//	CatsEye_train(&cat, x, x, sample, 10000/*epoch*/, sample, 0);
	printf("Training complete\n");

	// 結果の表示
	uint8_t *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);

		CatsEye_layer *l = &cat.layer[cat.end-1];
		double mse = 0;
		uint8_t *p = &pixels[(i/10)*size*10 + (i%10)*8];
		for (int j=0; j<size; j++) {
			p[(j/8)*8*10+(j%8)] = (uint8_t)(l->z[j] * 255.0);
			mse += (x[size*i+j]-l->z[j])*(x[size*i+j]-l->z[j]);

			p[5*size*10+(j/8)*8*10+(j%8)] = (uint8_t)(x[size*i+j] * 255.0);
		}
		printf("%d mse %lf\n", t[i], mse);
	}
	stbi_write_png("digits_autoencoder.png", 8*10, 8*10, 1, pixels, 8*10);

	memset(pixels, 0, 8*8*100);
/*	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 8, &pixels[(n/10)*8*8*10 + (n%10)*8], 8*10);
	}
	for (int n=0; n<l->ch; n++) {
		int s = l->ksize*l->ksize;
		int n = l->ksize+2;
		CatsEye_visualize(l->W+s*i, s, l->ksize, &pixels[28*28*10*(9+(i*n)/(28*10))+(i*n)%(28*10)], 28*10, 1);
	}
	stbi_write_png("digits_autoencoder_weights.png", 8*10, 8*10, 1, pixels, 8*10);*/
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}

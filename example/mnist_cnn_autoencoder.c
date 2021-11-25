//---------------------------------------------------------
//	Cat's eye
//
//		©2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast -fopenmp -lgomp
// clang mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast

#define CATS_USE_FLOAT
//#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include"svg.h"

#define ETA 1e-2
//#define ETA 1e-3

int main()
{
	const int wh = 28;
	const int size = 28*28;	// 入出力層(28x28)
	const int sample = 60000;

	// https://rightcode.co.jp/blog/information-technology/autoencoder-dimensionality-reduction-implications-of-machine-learning
/*	CatsEye_layer u[] = {	// epoch 20/ eta 1e-5
		{  size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=16 }, // b, 16, 10x10
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // b, 16, 5x5
		{     0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=32 }, // b, 32, 3x3
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=1 }, // b, 32, 2x2
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=0, .ch=64 }, // b, 64, 1x1
		{     0, CATS_ACT_LEAKY_RELU },

		{ size, CATS_LINEAR, ETA },
		{   32, CATS_ACT_RELU },
		{   32, CATS_LINEAR, ETA },
		{    2, CATS_ACT_RELU },
		{    2, CATS_LINEAR, ETA },
		{   32, CATS_ACT_RELU },
		{   32, CATS_LINEAR, ETA },
		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};*/
/*	CatsEye_layer u[] = { // 1e-2
		{  size, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=4 },
//		{  size, CATS_PADDING, .padding=1, .ich=1 },
//		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=4 },
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=16 },
		{     0, CATS_ACT_LEAKY_RELU, .name="encoder" },

//		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
//		{     0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=4 },
//		{  size, CATS_ACT_SIGMOID },
		{  size, CATS_LOSS_MSE },
	};*/
	CatsEye_layer u[] = { // https://kitakantech.com/cifar10-autoencoder/
		// 28 x 28 x 1
		{  size, CATS_CONV, ETA, .ksize=5, .stride=2, .padding=2, .ch=8 },
		{     0, CATS_ACT_LEAKY_RELU },
		// 14 x 14 x 4
		{     0, CATS_CONV, ETA, .ksize=5, .stride=2, .padding=2, .ch=8 },
		{     0, CATS_ACT_LEAKY_RELU, .name="encoder" },

		{     0, CATS_LINEAR, ETA },
		{  size, CATS_PIXELSHUFFLER, .r=2, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },

/*		{     0, CATS_DECONV, ETA, .ksize=6, .stride=2, .padding=2, .ch=8 },
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_DECONV, ETA, .ksize=6, .stride=2, .padding=2, .ch=1 },*/
		{  size, CATS_ACT_SIGMOID },
		{  size, CATS_LOSS_MSE },
	};
#if 0
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1, .ich=3 },
		{     0, CATS_CONV, 0.001, .ksize=3, .stride=1, .ch=16*3, .sx=34, .sy=34, .ich=3 },
		{     0, CATS_ACT_LEAKY_RELU, /*.alpha=0.01*/ },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

		{     0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=4*3 },
		{     0, CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3 },
		{  size, CATS_LOSS_MSE },
	};
#endif
	CatsEye cat = { .batch=1 };	// 0.0%
//	CatsEye cat = { .batch=256 };	// 0.0%
	CatsEye__construct(&cat, u);

	int16_t t[sample];	// ラベルデータ
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
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
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
			p[(j/wh)*wh*10+(j%wh)] = (uint8_t)(l->z[j] * 255.0);
			mse += (x[size*i+j]-l->z[j])*(x[size*i+j]-l->z[j]);

			p[5*size*10+(j/wh)*wh*10+(j%wh)] = (uint8_t)(x[size*i+j] * 255.0);
		}
//		printf("%d mse %lf\n", t[i], mse);
	}
	stbi_write_png("mnist_cnn_autoencoder.png", wh*10, wh*10, 1, pixels, wh*10);

	memset(pixels, 0, size*100);
/*	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 28, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
	}
	stbi_write_png("mnist_autoencoder_weights.png", 28*10, 28*10, 1, pixels, 28*10);*/
	free(pixels);

	// 潜在変数
	double xs[sample], ys[sample];
	for (int i=0; i<sample/50; i++) {
		CatsEye_forward(&cat, x+size*i);

		int e = CatsEye_getLayer(&cat, "encoder");
		CatsEye_layer *l = &cat.layer[e];
		xs[i] = l->z[0];
		ys[i] = l->z[1];
	}
	svg *psvg = svg_create(512, 512);
	//if (!psvg) return;
	svg_scatter(psvg, xs, ys, sample/50, t, 10, SVG_FRAME);
	svg_finalize(psvg);
	svg_save(psvg, "mnist_cnn_autoencoder.svg");
	svg_free(psvg);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

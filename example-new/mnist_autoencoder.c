//---------------------------------------------------------
//	Cat's eye
//
//		©2016,2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast -fopenmp -lgomp
// clang mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast

//#define CATS_DENOISING_AUTOENCODER
#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

	// Convolutional AutoEncoder
#if 0
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 16,14,14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=8 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 8,7,7

/*		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=8 },
		{     0, _CATS_ACT_LEAKY_RELU },
//		{     0, CATS_PADDING, .padding=1 },		// avoid 3,3
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 8,4,4


		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=8 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=2 },*/

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=8 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=2 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=4 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=1 },
		{     0, _CATS_ACT_SIGMOID },

		{  size, CATS_LOSS_MSE },
	};
#endif
	/*CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,14,14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,7,7


		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=1 },
		{     0, _CATS_ACT_SIGMOID },

		{  size, CATS_LOSS_MSE },
	};*/
#if 0
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,14,14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 16,7,7


//		{     0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
		{  size, CATS_LOSS_MSE },
	};
#endif
	/*CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=4 },
//		{     0, _CATS_ACT_RELU }, // minus??
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
//		{     0, _CATS_ACT_SIGMOID },
		{  size, CATS_LOSS_MSE },
	};*/
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16 },
//		{     0, _CATS_ACT_RELU }, // minus??
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
//		{     0, _CATS_ACT_SIGMOID },
		{  size, CATS_LOSS_MSE },
	};
#if 0
	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=16 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 16,14,14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=32 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },	// 32,7,7

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV, 0, 0.01, .ksize=3, .stride=1, .ch=4 },
		{     0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },
		{  size, CATS_LOSS_MSE },
	};
#endif
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	free(data);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	_CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1500/*random batch*/, 0);
//	_CatsEye_train(&cat, x, x, sample, 300/*repeat*/, 1500/*random batch*/, 0);
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist_autoencoder.weights");
//	CatsEye_saveJson(&cat, "mnist_autoencoder.json");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		_CatsEye_forward(&cat, x+size*i);
//		CatsEye_visualize(cat.layer[cat.layers-4].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);
//		CatsEye_visualize(cat.layer[cat.layers-3].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);
//		CatsEye_visualize(cat.layer[cat.layers-2].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);
		CatsEye_visualize(cat.layer[cat.layers-1].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

		unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*28];
		for (int j=0; j<size; j++) {
//			CatsEye_layer *l = &cat.layer[cat.layers-5];	// input
//			CatsEye_layer *l = &cat.layer[cat.layers-4];	// Conv
//			CatsEye_layer *l = &cat.layer[cat.layers-3];	// ReLU
//			CatsEye_layer *l = &cat.layer[cat.layers-2];	// Pool
			CatsEye_layer *l = &cat.layer[cat.layers-1];
///			p[(j/28)*28*10+(j%28)] = l->x[j] * 255.0;
			mse += (x[size*i+j]-l->x[j])*(x[size*i+j]-l->x[j]);
//			p[(j/28)*28*10+(j%28)] = cat.o[2][j] * 255.0;
//			mse += (x[size*i+j]-cat.o[2][j])*(x[size*i+j]-cat.o[2][j]);

//			p[5*size*10+(j/28)*28*10+(j%28)] = x[size*i+j] * 255.0;
			//p[5*size*10+(j/28)*28*10+(j%28)] = cat.layer[0].x[j] * 255.0;
			p[5*size*10+(j/28)*28*10+(j%28)] = cat.o[0][j] * 255.0;
		}
		printf("mse %lf\n", mse/size);
	}
	stbi_write_png("mnist_autoencoder.png", 28*10, 28*10, 1, pixels, 28*10);

/*	memset(pixels, 0, 28*28*100);
	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 28, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
	}
	stbi_write_png("mnist_autoencoder_weights.png", 28*10, 28*10, 1, pixels, 28*10);*/
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

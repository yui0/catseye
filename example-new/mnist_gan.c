//---------------------------------------------------------
//	Cat's eye
//
//		©2018-2019 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_gan.c -o mnist_gan -lm -Ofast -fopenmp -lgomp
// clang mnist_gan.c -o mnist_gan -lm -Ofast

#define CATS_USE_FLOAT
#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_ADAGRAD
//#define CATS_USE_RMSPROP
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define NAME	"mnist_gan"
//#define ZDIM	100
#define ZDIM	62
//#define NAME	"_mnist_gan"
//#define ZDIM	10

#define SAMPLE	60000
#define BATCH	10240
#define BATCH_G	20480
#define ETA	0.00005
//#define ETA	0.00003
//#define ETA	0.00001
//#define BATCH	640
//#define BATCH_G	1280

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

#if 0
	// https://qiita.com/triwave33/items/1890ccc71fab6cbca87e
	// https://github.com/gwaygenomics/keras_gan/blob/master/mnist_mlp_gan.ipynb
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, 0.01, .outputs=256 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_BATCHNORMAL },

		{       0, CATS_LINEAR, 0.01, .outputs=512 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_BATCHNORMAL },

		{       0, CATS_LINEAR, 0.01, .outputs=1024 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		//{       0, CATS_BATCHNORMAL }, // BAD[output]

		{       0, CATS_LINEAR, 0.01 },
		{    size, _CATS_ACT_TANH },	// [-1,1]
//		{    size, _CATS_ACT_SIGMOID },	// [0,1]

/*		{    ZDIM, CATS_LINEAR, 0.01, .outputs=1600 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.3 },

		{       0, CATS_LINEAR, 0.01, .outputs=1200 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.3 },

		{       0, CATS_LINEAR, 0.01, .outputs=1000 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.3 },

		{       0, CATS_LINEAR, 0.01 },
		{    size, _CATS_ACT_SIGMOID },	// [0,1]*/

		// discriminator
		{    size, CATS_LINEAR, 0.01 },
		{     512, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, 0.01 },
		{     256, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, 0.01 },
		{       1, _CATS_ACT_SIGMOID },

		{       1, CATS_LOSS_MSE },
	};
#endif
	// https://cntk.ai/pythondocs/CNTK_206A_Basic_GAN.html
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=128 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA },
		{    size, _CATS_ACT_TANH },	// [-1,1]

		// discriminator
		{    size, CATS_LINEAR, ETA, .outputs=128, .name="Discriminator" },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA },
		{       1, _CATS_ACT_SIGMOID },

		{       1, CATS_LOSS_MSE },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);
	cat.epoch = 0;
	int discriminator = CatsEye_getLayer(&cat, "Discriminator");
	printf("Discriminator: #%d\n", discriminator);
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);
	real *noise = malloc(sizeof(real)*ZDIM*SAMPLE);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
	printf("OK\n");

//	int16_t lreal[SAMPLE], lfake[SAMPLE];
	real lreal[SAMPLE], lfake[SAMPLE];
	for (int i=0; i<SAMPLE; i++) {
//		lreal[i] = 1;
//		lfake[i] = 0;
		lreal[i] = random(0.7, 1.2);
		lfake[i] = random(0.0, 0.3);
	}

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	for (int n=cat.epoch; n<10000; n++) {
		// Training Discriminator
		cat.start = discriminator;
		cat.slide = size;
		printf("Training Discriminator #%d: phase 1 [real]\n", n);
		_CatsEye_train(&cat, x, lreal, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);

		// Training Discriminator [ D(G(z)) = 0 ]
		for (int i=0; i<ZDIM*SAMPLE; i++) {
//			noise[i] = random(0, 1);
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.stop = discriminator/*+1*/;
		cat.slide = ZDIM;
		for (int i=0; i<discriminator; i++) cat.layer[i].fix = 1;
		printf("Training Discriminator #%d: phase 2 [fake]\n", n);
		_CatsEye_train(&cat, noise, lfake, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);
		for (int i=0; i<discriminator; i++) cat.layer[i].fix = 0;
		cat.stop = 0;

		// Training Generater [ D(G(z)) = 1 ]
		for (int i=0; i<ZDIM*SAMPLE; i++) {
//			noise[i] = random(0, 1);
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.slide = ZDIM;
		for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 1;
		printf("Training Generater #%d\n", n);
		_CatsEye_train(&cat, noise, lreal, SAMPLE, 1/*repeat*/, BATCH_G/*random batch*/, 0);
		for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 0;


		// 結果の表示
		unsigned char *pixels = calloc(1, size*100);
		for (int i=0; i</*50*/100; i++) {
//			double mse = 0;
			_CatsEye_forward(&cat, noise+ZDIM*i);
//			int p = _CatsEye_predict(&cat, noise+ZDIM*i);
//			printf("%d ", p);
			CatsEye_visualize(cat.layer[discriminator].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

//			cat.start = 9;
//			_CatsEye_forward(&cat, x+size*i);
//			CatsEye_visualize(cat.layer[9].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

/*			unsigned char *p = &pixels[(i/10)*size*10 + (i%10)*28];
			for (int j=0; j<size; j++) {
				CatsEye_layer *l = &cat.layer[8];
				mse += (x[size*i+j]-l->x[j])*(x[size*i+j]-l->x[j]);

				//p[5*size*10+(j/28)*28*10+(j%28)] = cat.layer[0].x[j] * 255.0;
				p[5*size*10+(j/28)*28*10+(j%28)] = cat.o[0][j] * 255.0;
			}
			printf("mse %lf\n", mse/size);*/
		}
		printf("\n");
		char buff[256];
		sprintf(buff, "/tmp/"NAME"_%05d.png", n);
		stbi_write_png(buff, 28*10, 28*10, 1, pixels, 28*10);
		free(pixels);

		cat.epoch = n;
		CatsEye_saveCats(&cat, NAME".cats");
	}
	printf("Training complete\n");

	free(noise);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_infoGAN.c -o mnist_infoGAN -lm -Ofast -fopenmp -lgomp
// clang mnist_infoGAN.c -o mnist_infoGAN -lm -Ofast

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define NAME	"mnist_infoGAN"
#define ZDIM	100
//#define NAME	"_mnist_infoGAN"
//#define ZDIM	10
#define CLASS	1
#define DIM	(CLASS+ZDIM)

#define SAMPLE	60000
#define BATCH	100
#define BATCH_G	200
#define OUTPUT	9
//#define OUTPUT	7
//#define ETA	0.0001
//#define ETA	0.01

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

#if 0
	// https://qiita.com/taku-buntu/items/0093a68bfae0b0ff879d
	// http://yusuke-ujitoko.hatenablog.com/entry/2017/08/30/205204
	CatsEye_layer u[] = {
		// generator
//		{     DIM, CATS_LINEAR, ETA, .outputs=1024 },
		{     DIM, CATS_LINEAR, ETA, .outputs=256 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },	// 128 7x7

		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, 0.001, .ksize=3, .stride=1/*2*/, .ch=64 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, 0.001, .ksize=3, .stride=1/*2*/, .ch=128 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA },
		{       2, _CATS_ACT_SIGMOID },
		{       2, CATS_LOSS_MSE },
	};
#endif
#if 0
	// https://cntk.ai/pythondocs/CNTK_206B_DCGAN.html
	CatsEye_layer u[] = {
		// generator
		{     DIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL },
//		{       0, _CATS_ACT_RELU },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL },
//		{       0, _CATS_ACT_RELU },	// 128 7x7
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28
//		{    size, _CATS_ACT_SIGMOID },	// [ 0,1]
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, 0.001, .ksize=3, .stride=1/*2*/, .ch=16 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, 0.001, .ksize=3, .stride=1/*2*/, .ch=64 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA },
		{       2, _CATS_ACT_SIGMOID },
		{       2, CATS_LOSS_MSE },
	};
#endif
#define ETA	3e-6
	CatsEye_layer u[] = {
		{     DIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU },	// 128 7x7

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, ETA, .ksize=5, .stride=1/*2*/, .ch=64 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=5, .stride=1/*2*/, .ch=128 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA },
		{     256, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA },
		{       2, _CATS_ACT_SIGMOID },
		{       2, CATS_LOSS_MSE },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);
	cat.epoch = 0;
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);
	real *noise = malloc(sizeof(real)*DIM*SAMPLE);

	real lreal[SAMPLE*2], lfake[SAMPLE*2];
	for (int i=0; i<SAMPLE; i++) {
		lreal[i*2] = random(0.7, 1.2);
		lfake[i*2] = random(0.0, 0.3);
		lfake[i*2+1] = (int)(frand()*9);
	}

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	fp = fopen("../example/train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) lreal[i*2+1] = data[i];
	fclose(fp);
	free(data);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	for (int n=cat.epoch; n<10000; n++) {
		// Training Discriminator
		cat.start = OUTPUT;
		cat.slide = size;
		printf("Training Discriminator #%d: phase 1 [real]\n", n);
		_CatsEye_train(&cat, x, lreal, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);

		// Training Discriminator [ D(G(z)) = 0 ]
		for (int i=0; i<DIM*SAMPLE; i++) {
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		for (int i=0; i<SAMPLE; i++) {
//			noise[i*DIM+ZDIM] = (int)(frand()*9);
			noise[i*DIM+ZDIM] = lfake[i*2+1];
		}
		cat.start = 0;
		cat.stop = OUTPUT+1;
		cat.slide = DIM;
		for (int i=0; i<OUTPUT; i++) cat.layer[i].fix = 1;
		printf("Training Discriminator #%d: phase 2 [fake]\n", n);
		_CatsEye_train(&cat, noise, lfake, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);
		for (int i=0; i<OUTPUT; i++) cat.layer[i].fix = 0;
		cat.stop = 0;

		// Training Generater [ D(G(z)) = 1 ]
		for (int i=0; i<DIM*SAMPLE; i++) {
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		for (int i=0; i<SAMPLE; i++) {
			noise[i*DIM+ZDIM] = lreal[i*2+1];
		}
		cat.start = 0;
		cat.slide = DIM;
		for (int i=OUTPUT; i<cat.layers; i++) cat.layer[i].fix = 1;
		printf("Training Generater #%d\n", n);
		_CatsEye_train(&cat, noise, lreal, SAMPLE, 1/*repeat*/, BATCH_G/*random batch*/, 0);
		for (int i=OUTPUT; i<cat.layers; i++) cat.layer[i].fix = 0;


		// 結果の表示
		unsigned char *pixels = calloc(1, size*100);
		for (int i=0; i<100; i++) {
			noise[i*DIM] = i%10;
			_CatsEye_forward(&cat, noise+DIM*i);
			CatsEye_visualize(cat.layer[OUTPUT].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);
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

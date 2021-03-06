//---------------------------------------------------------
//	Cat's eye
//
//		©2018-2019 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_lsgan.c -o mnist_lsgan -lm -Ofast -fopenmp -lgomp
// clang mnist_lsgan.c -o mnist_lsgan -lm -Ofast

#define CATS_USE_FLOAT
#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_ADAGRAD
//#define CATS_USE_RMSPROP
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define NAME	"mnist_lsgan"
//#define ZDIM	100
//#define ZDIM	62
//#define NAME	"_mnist_lsgan"
#define ZDIM	20
//#define ZDIM	10

#define SIZE	28
//#define SIZE	64
#if SIZE == 28
#define BPP	1
#define SAMPLE	60000
#else
#define BPP	3
#define SAMPLE	4721
#endif
#define BATCH	100
#define BATCH_G	400

//#define ETA	0.0001
#define ETA	0.001
//#define ETA	0.01

int main()
{
	int size = SIZE*SIZE*BPP;
	int sample = SAMPLE;

#if 0
	CatsEye_layer u[] = {
		// generator
#if 0
		{    ZDIM, CATS_LINEAR, 0.01, .outputs=1024 },
		{       0, CATS_BATCHNORMAL },
//		{    1024, _CATS_ACT_RELU },
		{    1024, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, 0.01, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL },
//		{ 128*7*7, _CATS_ACT_RELU },	// 128 7x7
		{ 128*7*7, _CATS_ACT_LEAKY_RELU },	// 128 7x7

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32, .sx=7, .sy=7 },	// 32 14x14
		//{       0, CATS_BATCHNORMAL },
//		{       0, _CATS_ACT_RELU },
		//{       0, _CATS_ACT_LEAKY_RELU },

/*		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=64 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=16, .sx=7, .sy=7 },	// 16 14x14
		{       0, _CATS_ACT_LEAKY_RELU },*/

		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },	// 1 28x28
		//{       0, CATS_BATCHNORMAL }, // BAD[output]
		{    size, _CATS_ACT_TANH },	// [-1,1]
//		{    size, _CATS_ACT_SIGMOID },	// [0,1]

//		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=8 },	// 8 28x28
//		{       0, CATS_CONV, 0.001, .ksize=1, .stride=1, .ch=1 },
//		{    size, _CATS_ACT_SIGMOID },	// [0,1]
#endif
		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL },
		{       0, _CATS_ACT_LEAKY_RELU },	// 128 7x7

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		//{    size, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=64 },
		{    size, CATS_CONV, ETA, .ksize=5, .stride=1/*2*/, .ch=64, .name="Discriminator" },
		//{    size, CATS_CONV, ETA, .ksize=5, .stride=2, .ch=64 },
//		{       0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		//{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=128 },
		{       0, CATS_CONV, ETA, .ksize=5, .stride=1/*2*/, .ch=128 },
		//{       0, CATS_CONV, ETA, .ksize=5, .stride=2, .ch=128 },
//		{       0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA },
		{     256, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
//		{    1024, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA },
		{       1, _CATS_ACT_SIGMOID },
		{       1, CATS_LOSS_MSE },
//		{       2, _CATS_ACT_SIGMOID },
//		{       2, CATS_LOSS_0_1 },
	};
#endif
#if 0
	// https://sonaeru-blog.com/dcgan/
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU },	// 128 7x7

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28

		/*{    ZDIM, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8, .ich=128, .sx=7, .sy=7 },
		{       0, _CATS_ACT_RELU },	// 128 7x7

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },	// 32 14x14
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_RELU },	// 128 14x14

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },	// 32 28x28
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_RELU },	// 64 28x28

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=1 },*/

		/*{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=128*7*7 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU }, // 128 7x7

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32, .ich=128, .sx=7, .sy=7 }, // 32 14x14
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },	// 1 28x28
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },*/
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32, .name="Discriminator" },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
		//{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256 },
		//{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, _CATS_ACT_SIGMOID },
		{       1, CATS_LOSS_MSE },
	};
#endif
	// https://cntk.ai/pythondocs/CNTK_206B_DCGAN.html
#if SIZE == 28
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL, /*.gamma=0.8*/ },
		{       0, _CATS_ACT_LEAKY_RELU, /*.alpha=0.2*/ },
//		{       0, _CATS_ACT_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=64*2*7*7 },
		{       0, CATS_BATCHNORMAL, /*.gamma=0.8*/ },
		{       0, _CATS_ACT_LEAKY_RELU, /*.alpha=0.2*/ },
//		{       0, _CATS_ACT_RELU },	// 128 7x7

/*		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32, .ich=128, .sx=7, .sy=7 },	// 32 14x14
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_RELU },	// 32 14x14

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },	// 1 28x28*/

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=16, .ich=128, .sx=7, .sy=7 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=1 },	// 1 28x28
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=0, .ch=64/*32*/, .name="Discriminator" },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=0, .ch=128/*64*/ },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256/*1024*/ },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, _CATS_ACT_SIGMOID },
		{       1, CATS_LOSS_MSE },
	};
#else
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
//		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, _CATS_ACT_RELU },

		{       0, CATS_LINEAR, ETA, .outputs=64*2*16*16 },
		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
//		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, _CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=64, .ich=128, .sx=16, .sy=16 },
		{       0, CATS_PIXELSHUFFLER, .r=4, .ch=3 },	// 3 64x64
		{    size, _CATS_ACT_TANH },	// [-1,1]


		// discriminator
		{    size, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=0, .ch=32, .name="Discriminator" },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=0, .ch=64 },
		{       0, CATS_AVGPOOL, .ksize=2, .stride=2 },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1024 },
//		{       0, CATS_BATCHNORMAL, .gamma=0.8 },
		{       0, _CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, _CATS_ACT_SIGMOID },
		{       1, CATS_LOSS_MSE },
	};
#endif
	CatsEye cat;
	_CatsEye__construct(&cat, u);
	cat.epoch = 0;
	int discriminator = CatsEye_getLayer(&cat, "Discriminator");
	printf("Discriminator: #%d\n", discriminator);
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}


	// 訓練データの読み込み
	printf("Training data: loading...");
	real *noise = malloc(sizeof(real)*ZDIM*SAMPLE);
#if SIZE == 28
	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	unsigned char *data = malloc(sample*size);
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
#else
	int16_t *t;
	real *x = _CatsEye_loadCifar("imas-face.bin", size, sizeof(int16_t), sample, &t);
	free(t);
#endif
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	for (int n=cat.epoch; n<10000; n++) {
//		int16_t lreal[SAMPLE], lfake[SAMPLE];
		real lreal[SAMPLE], lfake[SAMPLE];
		for (int i=0; i<SAMPLE; i++) {
//			lreal[i] = 1;
//			lfake[i] = 0;
			lreal[i] = random(0.7, 1.2);
			lfake[i] = random(0.0, 0.3);
		}

		// Training Discriminator
		cat.start = discriminator;
		cat.slide = size;
		printf("Training Discriminator #%d: phase 1 [real]\n", n);
		_CatsEye_train(&cat, x, lreal, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);

		// Training Discriminator [ D(G(z)) = 0 ]
		for (int i=0; i<ZDIM*SAMPLE; i++) {
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.stop = discriminator+1;
		cat.slide = ZDIM;
		for (int i=0; i<discriminator; i++) cat.layer[i].fix = 1;
		printf("Training Discriminator #%d: phase 2 [fake]\n", n);
		_CatsEye_train(&cat, noise, lfake, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);
		for (int i=0; i<discriminator; i++) cat.layer[i].fix = 0;
		cat.stop = 0;

		// Training Generater [ D(G(z)) = 1 ]
		for (int i=0; i<ZDIM*SAMPLE; i++) {
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
			//CatsEye_visualize(cat.layer[discriminator].x, size, SIZE, &pixels[(i/10)*size*10+(i%10)*SIZE], SIZE*10);
//			CatsEye_visualize(cat.layer[discriminator].x, size, SIZE*BPP, &pixels[(i/10)*size*10+(i%10)*SIZE*BPP], SIZE*BPP*10);
			_CatsEye_visualize(cat.layer[discriminator].x, SIZE*SIZE, SIZE, &pixels[(i/10)*size*10+(i%10)*SIZE*BPP], SIZE*10, BPP);

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
		stbi_write_png(buff, SIZE*10, SIZE*10, BPP, pixels, 0);
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

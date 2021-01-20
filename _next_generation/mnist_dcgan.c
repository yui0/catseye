//---------------------------------------------------------
//	Cat's eye
//
//		©2018-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_gan.c -o mnist_gan -lm -Ofast -fopenmp -lgomp
// clang mnist_gan.c -o mnist_gan -lm -Ofast

#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NAME	"mnist_dcgan"
//#define ZDIM	100
//#define ZDIM	62
#define ZDIM	10 // https://qiita.com/triwave33/items/a5b3007d31d28bc445c2

#define SAMPLE	60000
#define BATCH	20000
#define BATCH_G	40000
#define ETA	0.0002
//#define ETA	0.00005
//#define ETA	0.00003
//#define ETA	0.00001
#define ETA_AE	0.0005

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

	CatsEye_layer u_ae[] = {
#if 0
		// decoder / generator
		{    ZDIM, CATS_LINEAR, ETA_AE, .outputs=128 },
		{       0, CATS_ACT_RELU },
		{       0, CATS_LINEAR, ETA_AE, .outputs=4*14*14/*128*16*16*/ },
		{       0, CATS_ACT_RELU },

		{       0, CATS_PIXELSHUFFLER, .r=2/*4*/, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]
		{    size, CATS_LOSS_MSE },
#endif
		{    ZDIM, CATS_LINEAR, ETA_AE },
		{     128, CATS_LINEAR, ETA_AE },
		{    size, CATS_PIXELSHUFFLER, .r=2, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]
		{    size, CATS_LOSS_MSE },
	};
	CatsEye cat_ae = { .batch=256 };
	CatsEye__construct(&cat_ae, u_ae);

	// https://aidiary.hatenablog.com/entry/20180311/1520762446
	CatsEye_layer u[] = {
		// generator
#if 0
		{    ZDIM, CATS_LINEAR, ETA, .outputs=128/*1024*/ },
		{       0, CATS_ACT_RELU },
		{       0, CATS_LINEAR, ETA, .outputs=4*14*14/*128*16*16*/ },
		{       0, CATS_ACT_RELU },

		{       0, CATS_PIXELSHUFFLER, .r=2/*4*/, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]
#endif
		{    ZDIM, CATS_LINEAR, ETA_AE },
		{     128, CATS_LINEAR, ETA_AE },
		{    size, CATS_PIXELSHUFFLER, .r=2, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]

		// discriminator
		{    size, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=8/*64*/, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=16/*128*/ },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256/*1024*/ },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },

		{       1, CATS_SIGMOID_BCE },
	};
	CatsEye cat = { .batch=1 };
	CatsEye__construct(&cat, u);
	cat.epoch = 0;
	int discriminator = CatsEye_getLayer(&cat, "Discriminator");
//	printf("Discriminator: #%d\n", discriminator);
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	uint8_t *data = malloc(sample*size);
	real *noise = malloc(sizeof(real)*ZDIM*SAMPLE);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) {
		printf(" Can't open!\n");
		return -1;
	}
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
//	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
	printf("OK\n");

	// 訓練
	printf("Starting training...\n");
	for (int e=0; e<20; e++) {
		for (int i=0; i<ZDIM*SAMPLE; i++) {
			noise[i] = rand_normal(0, 1);
		}
		CatsEye_train(&cat_ae, noise, x, sample, 1/*epoch*/, sample, 0);
	}
//	CatsEye_train(&cat_ae, x, x, sample, 20/*epoch*/, sample, 0);
	printf("Training complete\n");

	int decoder = CatsEye_getLayer(&cat_ae, "decoder");
//	CatsEye_layer *l_ae = cat_ae.layer[decoder];
	CatsEye_layer *l = &cat.layer[0];
	for (int n=decoder; n<cat_ae.layers-1; n++) {
		memcpy(l->W, cat_ae.w[n], cat_ae.ws[n]);
		l++;
	}

	// ラベル作成
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
		CatsEye_train(&cat, x, lreal, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);

		// Training Discriminator [ D(G(z)) = 0 ]
		for (int i=0; i<ZDIM*SAMPLE; i++) {
//			noise[i] = random(0, 1);
//			noise[i] = random(-1, 1);
			noise[i] = rand_normal(0, 1);
		}
		cat.start = 0;
		cat.stop = discriminator+1;
		cat.slide = ZDIM;
		for (int i=0; i<discriminator; i++) cat.layer[i].fix = 1;
		printf("Training Discriminator #%d: phase 2 [fake]\n", n);
		CatsEye_train(&cat, noise, lfake, SAMPLE, 1/*repeat*/, BATCH/*random batch*/, 0);
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
		CatsEye_train(&cat, noise, lreal, SAMPLE, 1/*repeat*/, BATCH_G/*random batch*/, 0);
		for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 0;


		// 結果の表示
		uint8_t *pixels = calloc(1, size*100);
		for (int i=0; i</*50*/100; i++) {
//			double mse = 0;
			CatsEye_forward(&cat, noise+ZDIM*i);
//			int p = _CatsEye_predict(&cat, noise+ZDIM*i);
//			printf("%d ", p);
			CatsEye_visualize(cat.layer[discriminator].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10, 1);

//			cat.start = 9;
//			_CatsEye_forward(&cat, x+size*i);
//			CatsEye_visualize(cat.layer[9].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10);

/*			uint8_t *p = &pixels[(i/10)*size*10 + (i%10)*28];
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

//---------------------------------------------------------
//	Cat's eye
//
//		©2018-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_gan.c -o mnist_gan -lm -Ofast -fopenmp -lgomp
// clang mnist_gan.c -o mnist_gan -lm -Ofast

#define CATS_USE_ADAM
#define ADAM_BETA1	0.5
#define ADAM_BETA2	0.999

//#define ETA		2e-4 // OK
#define ETA		1e-4 // nan with epoch 30 over

#define BATCH	1
//#define BATCH	64
//#define BATCH	128

#define NOISE_FIXED
#define EPHOCS	100

//#define CATS_OPENCL
//#define CATS_OPENGL
#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#define KSIZE	28
#define KSIZE	48

#if KSIZE == 28

#define NAME	"mnist_dcgan_28"
#define SAMPLE	60000
#define CH	1
#define ZDIM	10 // https://qiita.com/triwave33/items/a5b3007d31d28bc445c2

#elif KSIZE == 48

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define NAME	"mnist_dcgan_48"
#define SAMPLE	(946-1)
#define CH	3
//#define ZDIM	100
//#define ZDIM	50
#define ZDIM	(3*3*6)

#else

#define KSIZE	96
#define NAME	"mnist_dcgan_96"
#define SAMPLE	(946-1)
#define CH	3
#define ZDIM	62

#endif

#define K	(KSIZE)
#define K2	(KSIZE/2)
#define K4	(KSIZE/4)
#define SIZE	(KSIZE*KSIZE*CH)

int main()
{
#if 0
	// https://lionbridge.ai/ja/articles/pytorch-gan-anime-character/
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=3*3*1024 },
/*		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_LINEAR, ETA, .outputs=3*3*1024 },*/
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=256, .sx=3, .sy=3, .ich=1024 },

		// 6 -> 12
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=512 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=512 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=128 },

		// 12 -> 24
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=256 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=64 },

		// 24 -> 48
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=128 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=3 },
		{    SIZE, CATS_ACT_TANH },	// [-1,1]

		// discriminator
		// 48 -> 24
		{    SIZE, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=64, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 24 -> 12
		{       0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=128 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 12 -> 6
		{       0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=256 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 6 -> 3
		{       0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=512 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
//		{       0, CATS_GAP }, // 7x7x128 -> 128

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=0, .ch=1 },
//		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },
		{       1, CATS_SIGMOID_BCE },
	};
#else
	// https://github.com/musyoku/LSGAN/blob/master/train_animeface/model.py
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=3*3*512, .wrange=0.02 },
//		{    ZDIM, CATS_CONV, ETA, .ksize=1, .stride=1, .sx=3, .sy=3, .ich=11, .ch=512, .wrange=0.02 },
		{       0, CATS_ACT_RELU, .alpha=0.2 },
//		{       0, CATS_BATCHNORMAL },

		// 3 -> 6
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=128, .sx=3, .sy=3, .ich=512 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=256, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RELU, .alpha=0.2 },

		// 6 -> 12
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=64 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=128, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RELU, .alpha=0.2 },

		// 12 -> 24
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=64, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RELU, .alpha=0.2 },

		// 24 -> 48
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=16 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=3 },
		{    SIZE, CATS_ACT_TANH, .name="Generator" },	// [-1,1]

		// discriminator
		// 48 -> 24
		{    SIZE, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=32, .wrange=0.02, .name="Discriminator" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 24 -> 12
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=64, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 12 -> 6
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=128, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 6 -> 3
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=256, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },
		{       1, CATS_SIGMOID_BCE },
	};
#endif
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);
	cat.epoch = 0;
	int discriminator = CatsEye_getLayer(&cat, "Discriminator");
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	uint8_t *data = malloc(SAMPLE*SIZE);
	real *noise = malloc(sizeof(real)*ZDIM*SAMPLE);

	// 訓練データの読み込み
	printf("Training data: loading...");
#if KSIZE == 28
	real *x = malloc(sizeof(real)*SIZE*SAMPLE);	// 訓練データ
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) {
		printf(" Can't open!\n");
		return -1;
	}
	fread(data, 16, 1, fp);		// header
	fread(data, SIZE, SAMPLE, fp);	// data
	for (int i=0; i<SAMPLE*SIZE; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<SAMPLE*SIZE; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
#elif KSIZE == 48
	real *x = malloc(sizeof(real)*SIZE*SAMPLE);
	int16_t t[SAMPLE];
	real *x2 = CatsEye_loadCifar("./animeface.bin", 96*96*3, sizeof(int16_t), SAMPLE, (int16_t**)&t); // 0-1
	for (int i=0; i<SAMPLE*CH; i++) {
		stbir_resize_float(x2+96*96*i, 96, 96, 0, x+48*48*i, 48, 48, 0, 1);
	}
	free(x2);
	for (int i=0; i<SAMPLE*SIZE; i++) { // -1,1
		x[i] = x[i]*2-1;
	}
#else
	int16_t t[SAMPLE];
	real *x = CatsEye_loadCifar("./animeface.bin", SIZE, sizeof(int16_t), SAMPLE, (int16_t**)&t); // 0-1
#endif
	printf("OK\n");

	// ラベル作成
//	int16_t lreal[SAMPLE], lfake[SAMPLE];
	real lreal[SAMPLE], lfake[SAMPLE];
	for (int i=0; i<SAMPLE; i++) {
//		lreal[i] = 1;
//		lfake[i] = 0;
		lreal[i] = random(0.7, 1.2);
		lfake[i] = random(0.0, 0.3);
	}
#ifdef NOISE_FIXED
	for (int i=0; i<ZDIM*SAMPLE; i++) { // fixed case
		noise[i] = rand_normal(0, 0.1);
	}
#endif

	// 訓練
	int step = 0;
	int repeat = SAMPLE/cat.batch;
//	real grad[BATCH];
	printf("Starting training...\n");
	for (int n=cat.epoch; n<EPHOCS; n++) {
		_CatsEye_data_transfer(&cat, x, lreal, SAMPLE);
		int base = cat.shuffle_base;
		for (int r=0; r<repeat; r++) {
			// Training Discriminator [ D(x) ]
			cat.start = discriminator;
			cat.slide = SIZE;
//			printf("Training Discriminator #%d: phase 1 [real]\n", n);
			cat.learning_data = x;
			cat.label_data = lreal;
			cat.shuffle_base = base;
			_CatsEye_forward(&cat);
			base = cat.shuffle_base;
			real loss = cat.loss;
//			memcpy(grad, cat.layer[output].dIn, sizeof(real)*BATCH);
			cat.start = discriminator;
			_CatsEye_backward(&cat);

			// Training Discriminator [ D(G(z)) = 0 ]
#ifndef NOISE_FIXED
			for (int i=0; i<ZDIM*SAMPLE; i++) {
//				noise[i] = random(0, 1);
//				noise[i] = random(-1, 1);
				noise[i] = rand_normal(0, 0.1);
			}
#endif
			cat.start = 0;
//			cat.stop = discriminator+1;
			cat.slide = ZDIM;
//			for (int i=0; i<discriminator; i++) cat.layer[i].fix = 1;
//			printf("Training Discriminator #%d: phase 2 [fake]\n", n);
			cat.learning_data = noise;
			cat.label_data = lfake;
			_CatsEye_forward(&cat);
			loss += cat.loss;
//			for (int i=0; i<BATCH; i++) grad[i] = (grad[i] + cat.layer[output].dIn[i])/2;
//			memcpy(cat.layer[output].dIn, grad, sizeof(real)*BATCH);

			cat.start = discriminator;
			_CatsEye_backward(&cat);

			if ((step % 1)==0) {
				// Training Generater [ D(G(z)) = 1 ]
#ifndef NOISE_FIXED
				for (int i=0; i<ZDIM*SAMPLE; i++) {
					noise[i] = rand_normal(0, 0.1);
				}
#endif
				cat.start = 0;
				cat.slide = ZDIM;
				for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 1;
//				printf("Training Generater #%d\n", n);
				cat.learning_data = noise;
				cat.label_data = lreal;
				_CatsEye_forward(&cat);
				_CatsEye_backward(&cat);
				for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 0;
			}
//			if ((step % 100)==0) {
				printf("Epoch: %d/100, Step: %d, D Loss: %f, G Loss: %f\n", n+1, step, loss, cat.loss);
//			}
//			if ((step % 1000)==0) {
			if ((step % 100)==0) {
				uint8_t *pixels = calloc(1, SIZE*100);
				for (int i=0; i<100; i++) {
					CatsEye_forward(&cat, noise+ZDIM*i);
					CatsEye_visualize(cat.layer[discriminator].x, K*K, K, &pixels[((i/10)*K*K*10+(i%10)*K)*CH], K*10, CH);
				}
//				printf("\n");
				char buff[256];
				sprintf(buff, "/tmp/"NAME"_%05d.png", n);
				stbi_write_png(buff, K*10, K*10, CH, pixels, 0);
				free(pixels);

				cat.epoch = n;
				CatsEye_saveCats(&cat, NAME".cats");
			}

			step++;
		}
	}
	printf("Training complete\n");

	free(noise);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

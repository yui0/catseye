//---------------------------------------------------------
//	Cat's eye
//
//		©2018-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc dcgan.c -o dcgan -lm -Ofast -fopenmp -lgomp
// clang dcgan.c -o dcgan -lm -Ofast

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CATS_USE_ADAM
#define ADAM_BETA1	0.5
#define ADAM_BETA2	0.999

#define ETA		1e-4
#define BATCH		1
//#define BATCH		64
//#define BATCH		128
#define EPHOCS		100
#define NOISE_FIXED

#define CATS_CHECK
//#define CATS_OPENCL
//#define CATS_OPENGL
#define CATS_USE_FLOAT
#include "catseye.h"

//#define KSIZE	28
#define KSIZE	48

#if KSIZE == 28

#define NAME	"dcgan_28"
#define SAMPLE	60000
#define CH	1
#define ZDIM	10 // https://qiita.com/triwave33/items/a5b3007d31d28bc445c2

#elif KSIZE == 48

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define NAME	"dcgan2_48"
#define SAMPLE	(946-1)
#define CH	3
//#define ZDIM	100
//#define ZDIM	50
#define ZDIM	(3*3*6)

#else

#define KSIZE	96
#define NAME	"dcgan_96"
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
#if 1
	// https://github.com/musyoku/LSGAN/blob/master/train_animeface/model.py
	CatsEye_layer u[] = {
		// generator
//		{    ZDIM, CATS_LINEAR, ETA, .outputs=3*3*512, .wrange=0.02 },
		{    ZDIM, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .sx=3, .sy=3, .ich=6, .ch=512, .wrange=0.02 },
		{       0, CATS_ACT_RRELU },
//		{       0, CATS_BATCHNORMAL },

		// 3 -> 6
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=128, .sx=3, .sy=3, .ich=512 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=256, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },

		// 6 -> 12
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=64 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=128, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },

		// 12 -> 24
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },
//		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=12, .wrange=0.02 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .wrange=0.02 },
//		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=64, .wrange=0.02 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },

		// 24 -> 48
//		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=3 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=16 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=3 },
		{    SIZE, CATS_ACT_TANH, .name="Generator" },	// [-1,1]

		// discriminator
		// 48 -> 24
		{    SIZE, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=32, .wrange=0.02, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 24 -> 12
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=64, .wrange=0.02 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 12 -> 6
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=128, .wrange=0.02 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 6 -> 3
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=256, .wrange=0.02 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=0, .ch=1/*, .wrange=0.02*/ }, // 3x3 -> 1x1x1
//		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },
		{       1, CATS_SIGMOID_BCE },
	};
#else
	// https://medium.com/@crosssceneofwindff/%E7%B6%9A-gan%E3%81%AB%E3%82%88%E3%82%8B%E4%BA%8C%E6%AC%A1%E5%85%83%E7%BE%8E%E5%B0%91%E5%A5%B3%E7%94%BB%E5%83%8F%E7%94%9F%E6%88%90-a32fbe808eb0
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=3*3*512 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },

		// 3 -> 6
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128, .name="conv1_1", .sx=3, .sy=3, .ich=512 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=512 },
		{       0, CATS_SHORTCUT, .layer="conv1_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=128, .sx=3, .sy=3, .ich=512 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=256/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 6 -> 12
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=64, .name="conv2_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=256 },
		{       0, CATS_SHORTCUT, .layer="conv2_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=64 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 12 -> 24
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=32, .name="conv3_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128 },
		{       0, CATS_SHORTCUT, .layer="conv3_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=64/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 24 -> 48
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=16, .name="conv4_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=16 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=64 },
		{       0, CATS_SHORTCUT, .layer="conv4_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=16 },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=3 },
		{    SIZE, CATS_ACT_TANH, .name="Generator" },	// [-1,1]

		// discriminator
		// 48 -> 24
		{    SIZE, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, /*.wrange=0.02,*/ .name="Discriminator" },
//		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_ACT_RRELU },

		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=16, .name="conv5_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=16 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=64 },
		{       0, CATS_SHORTCUT, .layer="conv5_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=2, .padding=0, .ch=128/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 24 -> 12
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=32, .name="conv6_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128 },
		{       0, CATS_SHORTCUT, .layer="conv6_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=2, .padding=0, .ch=256/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 12 -> 6
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=64, .name="conv7_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=256 },
		{       0, CATS_SHORTCUT, .layer="conv7_1" },
		{       0, CATS_ACT_RRELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=2, .padding=0, .ch=512/*, .wrange=0.02*/ },
		{       0, CATS_ACT_RRELU },

		// 6 -> 3
		// ResidualLayer
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128, .name="conv8_1" },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_RRELU },
		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=512 },
		{       0, CATS_SHORTCUT, .layer="conv8_1" },
		{       0, CATS_ACT_RRELU },

//		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=0, .ch=1/*, .wrange=0.02*/ }, // 3x3 -> 1x1x1
		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },
		{       1, CATS_SIGMOID_BCE },
	};
#endif
	CatsEye cat = { .batch=BATCH, /*.da=CATS_ORIGINAL|CATS_FLIP*/ };
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
	real *pix = CatsEye_loadCifar("./animeface.bin", 96*96*3, sizeof(int16_t), SAMPLE, (int16_t**)&t); // 0-1
	for (int i=0; i<SAMPLE*CH; i++) {
		stbir_resize_float(pix+96*96*i, 96, 96, 0, x+48*48*i, 48, 48, 0, 1);
	}
	free(pix);
/*	for (int i=0; i<SAMPLE; i++) {
		real *y = x +i*SIZE;
		real *u = x +i*SIZE +SIZE/CH;
		real *v = x +i*SIZE +SIZE/CH*2;
		for (int n=0; n<SIZE/CH; n++) {
			real r = *y *255;
			real g = *u *255;
			real b = *v *255;
			*y++ = (0.298912*r +0.586611*g +0.114478*b)/255.0;	// CCIR Rec.601
//			*u++ = -0.1687*r -0.3313*g +0.500 *b;
//			*v++ =  0.500 *r -0.4187*g -0.0813*b;
			*u++ = (0.298912*r +0.586611*g +0.114478*b)/255.0;	// CCIR Rec.601
			*v++ = (0.298912*r +0.586611*g +0.114478*b)/255.0;	// CCIR Rec.601
		}
	}*/
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
			// Training Discriminator with real [ D(x) = 1 ]
//			printf("Training Discriminator #%d: phase 1 [real]\n", n);
			cat.start = discriminator;
			cat.slide = SIZE;
			cat.learning_data = x;
			cat.label_data = lreal;
			cat.shuffle_base = base;
			_CatsEye_forward(&cat);
			base = cat.shuffle_base;
			real loss = cat.loss;
			cat.start = discriminator;
			_CatsEye_zero_grad(&cat);
			_CatsEye_backward(&cat);
//			_CatsEye_update(&cat);

			// Training Discriminator with fake [ D(G(z)) = 0 ]
//			printf("Training Discriminator #%d: phase 2 [fake]\n", n);
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
			cat.learning_data = noise;
			cat.label_data = lfake;
			_CatsEye_forward(&cat);
			loss += cat.loss;
			cat.start = discriminator;
//			for (int i=0; i<discriminator; i++) cat.layer[i].fix = 1;
//			_CatsEye_zero_grad(&cat);
			_CatsEye_backward(&cat);
			_CatsEye_update(&cat);
//			for (int i=0; i<discriminator; i++) cat.layer[i].fix = 0;

			if ((step % 1)==0) {
				// Training Generater [ D(G(z)) = 1 ]
//				printf("Training Generater #%d\n", n);
#ifndef NOISE_FIXED
				for (int i=0; i<ZDIM*SAMPLE; i++) {
					noise[i] = rand_normal(0, 0.1);
				}
#endif
				cat.start = 0;
				cat.slide = ZDIM;
				for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 1;
				cat.learning_data = noise;
				cat.label_data = lreal;
				_CatsEye_forward(&cat);
				_CatsEye_zero_grad(&cat);
				_CatsEye_backward(&cat);
				_CatsEye_update(&cat);
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
//					CatsEye_visualizeYUV(cat.layer[discriminator].x, K*K, K, &pixels[((i/10)*K*K*10+(i%10)*K)*CH], K*10);
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

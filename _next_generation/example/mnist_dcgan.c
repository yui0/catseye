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
#define ETA		0.001
//#define ETA		0.0002
//#define ETA		0.00005
//#define ETA		0.00002 //
//#define ETA_AE	0.0005

//#define CATS_OPENCL
//#define CATS_OPENGL
#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NAME	"mnist_dcgan"
//#define ZDIM	100
//#define ZDIM	62
#define ZDIM	10 // https://qiita.com/triwave33/items/a5b3007d31d28bc445c2

#define SAMPLE	60000
//#define BATCH	1
#define BATCH	64
//#define BATCH	128

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

#if 1
	// https://aidiary.hatenablog.com/entry/20180311/1520762446
	CatsEye_layer u[] = {
		// generator
#if 0
		{    ZDIM, CATS_LINEAR, ETA, .outputs=14*14*256 },
		{       0, CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256, .ich=16, .sx=14, .sy=14 },
		{       0, CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
		{       0, CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
		{       0, CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4 },
		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },

		{    size, CATS_ACT_TANH },	// [-1,1]
#else
		{    ZDIM, CATS_LINEAR, ETA, .outputs=128*14*14/*512*/ },
		{       0, CATS_ACT_RELU },
//		{       0, CATS_LINEAR, ETA, .outputs=128*14*14 },
//		{       0, CATS_ACT_RELU },

		{       0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ich=128, .sx=14, .sy=14, .ch=4 },

		{       0, CATS_PIXELSHUFFLER, .r=2/*4*/, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]
#endif

		// discriminator
		{    size, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=64, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		{       0, CATS_CONV, ETA, .ksize=4, .stride=2, .padding=1, .ch=128 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
//		{       0, CATS_GAP }, // 7x7x128 -> 128
		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },

		{       1, CATS_SIGMOID_BCE },
	};
#else
	// https://github.com/Yangyangii/GAN-Tutorial/blob/master/MNIST/DCGAN.ipynb
	CatsEye_layer u[] = {
		// generator
		{    ZDIM, CATS_LINEAR, ETA, .outputs=1024 },
		{       0, CATS_ACT_RELU },
		{       0, CATS_LINEAR, ETA, .outputs=4*14*14 },
		{       0, CATS_ACT_RELU },

		{       0, CATS_PIXELSHUFFLER, .r=2, .ch=1 },
		{    size, CATS_ACT_TANH },	// [-1,1]

		// discriminator
		// 28 -> 14
		{    size, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=512, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 14 -> 7
		{       0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=256 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
		// 7 -> 4
		{       0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=128 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },
//		{       0, CATS_AVGPOOL, .ksize=4, .stride=1 }, // 4x4x128 -> 128
		{       0, CATS_GAP }, // 4x4x128 -> 128

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },

		{       1, CATS_SIGMOID_BCE },
	};
#endif
	CatsEye cat = { .batch=BATCH };
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
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1;
	fclose(fp);
	free(data);
	printf("OK\n");

#if 0
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
		memcpy(l->w, cat_ae.w[n], cat_ae.ws[n]);
		l++;
	}
#endif

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
	int step = 0;
	int repeat = SAMPLE/cat.batch;
//	real grad[BATCH];
	printf("Starting training...\n");
	for (int n=cat.epoch; n<50; n++) {
		_CatsEye_data_transfer(&cat, x, lreal, SAMPLE);
		int base = cat.shuffle_base;
		for (int r=0; r<repeat; r++) {
			// Training Discriminator [ D(x) ]
			cat.start = discriminator;
			cat.slide = size;
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
			for (int i=0; i<ZDIM*SAMPLE; i++) {
//				noise[i] = random(0, 1);
//				noise[i] = random(-1, 1);
				noise[i] = rand_normal(0, 1);
			}
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
				for (int i=0; i<ZDIM*SAMPLE; i++) {
					noise[i] = rand_normal(0, 1);
				}
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
				printf("Epoch: %d/50, Step: %d, D Loss: %f, G Loss: %f\n", n, step, loss, cat.loss);
//			}
//			if ((step % 1000)==0) {
			if ((step % 100)==0) {
				uint8_t *pixels = calloc(1, size*100);
				for (int i=0; i<100; i++) {
					CatsEye_forward(&cat, noise+ZDIM*i);
					CatsEye_visualize(cat.layer[discriminator].x, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10, 1);
				}
//				printf("\n");
				char buff[256];
				sprintf(buff, "/tmp/"NAME"_%05d.png", n);
				stbi_write_png(buff, 28*10, 28*10, 1, pixels, 28*10);
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

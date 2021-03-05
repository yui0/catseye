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
#define ETA		0.0002

//#define CATS_OPENCL
//#define CATS_OPENGL
#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NAME		"mnist_cgan"
#define CLASS_NUM	10
//#define ZDIM		100
#define ZDIM		62
#define ZSIZE		(ZDIM + CLASS_NUM)
#define DSIZE		(28*28 + CLASS_NUM)

#define SAMPLE		60000
#define BATCH		64

int main()
{
	int size = 28*28;	// 入出力層(28x28)
	int sample = 60000;

	// https://github.com/Yangyangii/GAN-Tutorial/blob/master/MNIST/Conditional-GAN.ipynb
	CatsEye_layer u[] = {
		// generator
		{   ZSIZE, CATS_LINEAR, ETA, .outputs=128, .name="Generater" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=512 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1024 },
//		{       0, CATS_BATCHNORMAL },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=size },
		{    size, CATS_ACT_TANH, .name="GAN" },	// [-1,1]

		{    size, CATS_CONCAT, .layer="Generater", .offset=ZDIM, /*.size=CLASS_NUM*/ },

		// discriminator
		{   DSIZE, CATS_LINEAR, ETA, .outputs=512, .name="Discriminator" },
//		{    size, CATS_LINEAR, ETA, .outputs=512, .name="Discriminator" },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=256 },
		{       0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

		{       0, CATS_LINEAR, ETA, .outputs=1 },
		{       1, CATS_ACT_SIGMOID },
		{       1, CATS_SIGMOID_BCE, .name="Output" },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);
	cat.epoch = 0;
	int gan = CatsEye_getLayer(&cat, "GAN");
	int discriminator = CatsEye_getLayer(&cat, "Discriminator");
	if (!CatsEye_loadCats(&cat, NAME".cats")) {
		printf("Loading success!!\n");
	}

	real *x = calloc(DSIZE*sample, sizeof(real));	// 訓練データ
	uint8_t *data = malloc(sample*size);
	real *noise = calloc(ZSIZE*SAMPLE, sizeof(real));

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
//	for (int i=0; i<sample*size; i++) x[i] = data[i] /255.0 *2 -1; // [0,1] => [-1,1]
	for (int i=0; i<sample; i++) {
		for (int n=0; n<size; n++) x[i*DSIZE+n] = data[i*size+n] /255.0;
//		for (int n=0; n<size; n++) x[i*DSIZE+n] = data[i*size+n] /255.0 *2 -1; // [0,1] => [-1,1]
	}
	fclose(fp);
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
//	for (int i=0; i<sample; i++) t[i] = data[i];
	for (int i=0; i<sample; i++) x[i*DSIZE+size+data[i]] = 1;
	for (int i=0; i<SAMPLE; i++) noise[i*ZSIZE+ZDIM+data[i]] = 1;
	fclose(fp);
	free(data);
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

	// 訓練
	int step = 0;
	int repeat = SAMPLE/cat.batch;
//	real grad[BATCH];
	printf("Starting training...\n");
	for (int n=cat.epoch; n<40; n++) {
		_CatsEye_data_transfer(&cat, x, lreal, SAMPLE);
		int base = cat.shuffle_base;
		for (int r=0; r<repeat; r++) {
			// Training Discriminator [ D(x) ]
			cat.start = discriminator;
			cat.slide = DSIZE;
//			printf("Training Discriminator #%d: phase 1 [real]\n", n);
			cat.learning_data = x;
			cat.label_data = lreal;
			cat.shuffle_base = base;
			_CatsEye_forward(&cat);
			base = cat.shuffle_base;
			real loss = cat.loss;
//			memcpy(grad, cat.layer[output].dIn, sizeof(real)*BATCH);
//			cat.start = discriminator;
			_CatsEye_backward(&cat);

			// Training Discriminator [ D(G(z)) = 0 ]
			for (int i=0; i<ZSIZE*SAMPLE; i++) {
//				noise[i] = random(0, 1);
//				noise[i] = random(-1, 1);
				noise[i] = rand_normal(0, 1);
			}
//			cat.shuffle_base = base;
			for (int i=0; i<SAMPLE; i++) {
				for (int n=0; n<CLASS_NUM; n++) noise[i*ZSIZE+ZDIM+n] = 0;
//				int n = random(0, CLASS_NUM-1);
				int n = i%10;
				noise[i*ZSIZE+ZDIM+n] = 1;
			}
			cat.start = 0;
//			cat.stop = discriminator+1;
			cat.slide = ZSIZE;
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
//				for (int i=0; i<ZDIM*SAMPLE; i++) noise[i] = rand_normal(0, 1);
//				cat.shuffle_base = base;
				for (int i=0; i<ZSIZE*SAMPLE; i++) noise[i] = rand_normal(0, 1);
				for (int i=0; i<SAMPLE; i++) {
					for (int n=0; n<CLASS_NUM; n++) noise[i*ZSIZE+ZDIM+n] = 0;
					int n = i%10;
					noise[i*ZSIZE+ZDIM+n] = 1;
				}
				cat.start = 0;
				cat.slide = ZSIZE;
				for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 1;
//				printf("Training Generater #%d\n", n);
				cat.learning_data = noise;
				cat.label_data = lreal;
				_CatsEye_forward(&cat);
				_CatsEye_backward(&cat);
				for (int i=discriminator; i<cat.layers; i++) cat.layer[i].fix = 0;
			}
//			if ((step % 100)==0) {
				printf("Epoch: %d/40, Step: %d, D Loss: %f, G Loss: %f\n", n, step, loss, cat.loss);
//			}
//			if ((step % 1000)==0) {
			if ((step % 100)==0) {
				uint8_t *pixels = calloc(1, size*100);
				for (int i=0; i<100; i++) {
					CatsEye_forward(&cat, noise+ZSIZE*i);
					CatsEye_visualize(cat.layer[gan].z, size, 28, &pixels[(i/10)*size*10+(i%10)*28], 28*10, 1);
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

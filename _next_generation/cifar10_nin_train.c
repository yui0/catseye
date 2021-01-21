//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops

//#define CATS_USE_RMSPROP
//#define ETA	0.001 // RMSProp

#define ETA	0.01 // SGD

#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NAME	"cifar10_nin_train"
#define SIZE	32	// 69.7%(10)
//#define SIZE	96	// 92.0%(10)

int main()
{
#if SIZE == 32
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;		// 出力層
	int sample = 10000;
#else
	int k = 96;		// image size
	int size = 96*96*3;	// 入力層
	int label = 10;		// 出力層
	int sample = 946-1;
#endif

	// Network in Network
	CatsEye_layer u[] = { 	// 49.7%(10)
		{  size, CATS_CONV, ETA, .ksize=7, .stride=1, .ch=96, .ich=3, .sx=k, .sy=k, .name="conv1-1" },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96, .name="conv1-2" },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96, .name="conv1-3" },
		{     0, CATS_MAXPOOL, .ksize=6, .stride=4 },
//		{     0, CATS_AVGPOOL, .ksize=6, .stride=4 }, // average pooling 34.1%(10)
//		{     0, CATS_GAP }, // global average pooling 21.7%(10)
		{     0, CATS_LINEAR, ETA, .outputs=label },
		{ label, CATS_ACT_SOFTMAX },
		{ label, CATS_SOFTMAX_CE },
	};
#if 0
	CatsEye_layer u[] = { // http://taka74k4.hatenablog.com/entry/2017/09/19/203748
		{  size, CATS_CONV, ETA, .ksize=7, .stride=1, .ch=96, .ich=3, .sx=k, .sy=k, .name="conv1-1" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96, .name="conv1-2" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96, .name="conv1-3" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1
//		{     0, CATS_MAXPOOL, .ksize=3, .stride=1 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=4, .stride=1, .ch=192, .name="conv2-1" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192, .name="conv2-2" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192, .name="conv2-3" },
//		{     0, CATS_ACT_RELU },
//		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1
//		{     0, CATS_MAXPOOL, .ksize=3, .stride=1 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=192, .name="conv3-1" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192, .name="conv3-2" },
//		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=10, .name="conv3-3" },
//		{     0, CATS_ACT_RELU },

		{     0, CATS_MAXPOOL, .ksize=6, .stride=1 }, // k3,s2,p1

//		{     0, CATS_AVGPOOL, .ksize=3, .stride=1 }, // global average pooling
		{     0, CATS_LINEAR, ETA, .outputs=label },
		{ label, CATS_ACT_SOFTMAX },
		{ label, CATS_SOFTMAX_CE },
	};
#endif
#if 0
	CatsEye_layer u[] = {
//		{  size, CATS_CONV, ETA, .ksize=11, .stride=4, .ch=192, .ich=3, .sx=k, .sy=k },
		{  size, CATS_CONV, ETA, .ksize=11, .stride=2, .ch=192, .ich=3, .sx=k, .sy=k },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=160 },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96 },
		{     0, CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=5, .stride=1, .ch=192/*, .padding=2*/ },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, CATS_ACT_RELU },
//		{     0, CATS_AVGPOOL, .ksize=2, .stride=2 }, // k3,s2,p1
/*		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=192 },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=label },
		{     0, CATS_ACT_RELU },*/

//		{     0, CATS_AVGPOOL, .ksize=3, .stride=1 }, // global average pooling
		{     0, CATS_LINEAR, ETA, .outputs=label },
		{ label, CATS_ACT_SOFTMAX },
		{ label, CATS_SOFTMAX_CE },
	};
#endif
	CatsEye cat = { .batch=1 };
	CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
#if SIZE == 32
	real *x = CatsEye_loadCifar("data_batch_1.bin", 32*32*3, 1, sample, &t);
#else
	real *x = CatsEye_loadCifar("animeface.bin", k*k*3, sizeof(int16_t), sample, &t);
#endif
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	printf("Training complete\n");

	// 結果の表示
	static int result[10][10];
	uint8_t *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		result[t[i]][p]++;
		if (p==t[i]) r++;
		else {
			if (c<100) {
				CatsEye_visualize(x+size*i, k*k, k, &pixels[(c/10)*size*10+(c%10)*k*3], k*10, 3);
			}
			c++;
		}
	}
	for (int i=0; i<10; i++) {
		for (int j=0; j<10; j++) {
			printf("%3d ", result[i][j]);
		}
		printf("\n");
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png(NAME"_wrong.png", k*10, k*10, 3/*bpp*/, pixels, 0);

	int n[10]; // 10 classes
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
		int p = CatsEye_predict(&cat, x+size*i);

		CatsEye_visualize(x+size*i, k*k, k, &pixels[(p*k*k*10+(n[p]%10)*k)*3], k*10, 3);
		n[p]++;
	}
	stbi_write_png(NAME"_classify.png", k*10, k*10, 3, pixels, 0);

	memset(pixels, 0, size*100);
	/*for (int i=0; i<10*10; i++)*/ {
		int p = CatsEye_predict(&cat, x/*+size*i*/);

		int x = 0;
		for (int n=0; n<cat.layers; n++) {
			CatsEye_layer *l = &cat.layer[n];
			if (l->type == CATS_LINEAR) {
				continue;
			}

			int mch = l->ch > 10 ? 10 : l->ch;
			for (int ch=0; ch<mch; ch++) {
				CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pixels[x +ch*(l->oy+2)*k*10], k*10, 1);
			}
			x += l->ox+2;
		}
	}
	stbi_write_png(NAME"_predict.png", k*10, k*10, 1, pixels, 0);
	free(pixels);

	free(t);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_RMSPROP
//#define ETA	0.0001	// 00.0
#define ETA	0.001	// 88.0%(10)
//#define ETA	1e-6
#define BATCH	1
//#define BATCH	64	// 44.5

#define NAME	"resnet18"
//#define SIZE	32	// 00.0%(10)
#define SIZE	96	// 00.0%(10)
//#define SIZE	227

#define CATS_CHECK
#define CATS_USE_FLOAT
#define CATS_OPENCL
//#define CATS_OPENGL
#include "./catseye.h"

int main()
{
#if SIZE == 32
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;
#else
	int k = 96;		// image size
	int size = 96*96*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 946-1;
#endif

	// https://buildersbox.corp-sansan.com/entry/2020/10/13/110000
	CatsEye_layer u[] = {
		{  size, CATS_CONV, ETA, .ksize=7, .stride=2, .padding=3, .ch=64, .ich=3, .sx=k, .sy=k },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
//		{     0, CATS_MAXPOOL, .ksize=3, .stride=2, .padding=1 },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

		// ResidualLayer
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .name="conv2_1" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv2_1" },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64, .name="conv2_2" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv2_2" },
		{     0, CATS_ACT_RRELU },

		// trans
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=128 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU }, // 74.6%(10)

		// ResidualLayer
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128, .name="conv3_1" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv3_1" },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128, .name="conv3_2" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=128 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv3_2" },
		{     0, CATS_ACT_RRELU },

		// trans
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .padding=0, .ch=256 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },

		// ResidualLayer
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256, .name="conv4_1" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv4_1" },
		{     0, CATS_ACT_RRELU },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256, .name="conv4_2" },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=256 },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_SHORTCUT, .layer="conv4_2" },
		{     0, CATS_ACT_RRELU }, // 88.0%(10)

		{     0, CATS_GAP }, // -> 512
		{     0, CATS_LINEAR, ETA, .outputs=label },

		{ label, CATS_ACT_SOFTMAX },
		{ label, CATS_SOFTMAX_CE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
#if SIZE == 32
	real *x = CatsEye_loadCifar("./data_batch_1.bin", 32*32*3, 1, sample, &t);
#else
	real *x = CatsEye_loadCifar("./animeface.bin", k*k*3, sizeof(int16_t), sample, &t);
#endif
	printf("OK\n");

	// 訓練
	printf("Starting training...\n");
	CatsEye_train(&cat, x, t, sample, 10/*repeat*/, sample/*random batch*/, sample/10/*verify*/);
	printf("Training complete\n");

	// 結果の表示
	static int result[10][10];
	unsigned char *pixels = calloc(1, size*100);
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

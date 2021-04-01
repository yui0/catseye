//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc espcn_train.c -o espcn_train -lm -Ofast -fopenmp -lgomp
// clang espcn_train.c -o espcn_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL` -mf16c

#define CATS_USE_ADAM
#define ADAM_BETA1	0.5
#define ADAM_BETA2	0.999
#define ETA		1e-4
//#define CATS_USE_RMSPROP
//#define ETA		0.01	// batch 1
//#define BATCH		1
#define BATCH		64

#define CATS_USE_FLOAT
//#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define NAME	"espcn"

int main()
{
#if SIZE == 128
	const int size = 128*128*3;	// 入力層ユニット
#else
	const int k = 96;		// image size
	const int size = k*k*3;	// 入力層
	const int label = 10;	// 出力層
	const int sample = 946-1;
#endif

	// https://nykergoto.hatenablog.jp/entry/2019/05/28/%E7%94%BB%E5%83%8F%E3%81%AE%E8%B6%85%E8%A7%A3%E5%83%8F%E5%BA%A6%E5%8C%96%3A_ESPCN_%E3%81%AE_pytorch_%E5%AE%9F%E8%A3%85_/_%E5%AD%A6%E7%BF%92
	CatsEye_layer u[] = {
		{size/4, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=64, .ich=3 },
		{     0, CATS_ACT_TANH },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
		{     0, CATS_ACT_TANH },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
		{     0, CATS_ACT_TANH },

		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4*3 },
//		{     0, CATS_ACT_TANH }, // avoid nan
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3 },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	real *xh = calloc(sizeof(real), k/2*k/2*sample);	// 訓練データ
	int16_t t[sample];				// ラベルデータ
	//uint8_t *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data: loading...");
	real *x = CatsEye_loadCifar("./animeface.bin", k*k*3, sizeof(int16_t), sample, &t); // 0-1
	for (int i=0; i<sample*3; i++) {
		stbir_resize_float(x+k*k*i, k, k, 0, xh+k/2*k/2*i, k/2, k/2, 0, 1);
	}
	printf("OK\n");
	/*{
	unsigned char *pixels = calloc(3, size*100);
	for (int i=0; i<10*10; i++) {
		CatsEye_visualize(xh+size/4*i, k/2*k/2, k/2, &pixels[((i/10)*k/2*k*10+(i%10)*k/2)*3], k*10, 3);
		CatsEye_visualize(x+size*i, k*k, k, &pixels[((i/10)*k*k*10+(i%10)*k)*3], k*10, 3);
	}
	stbi_write_png(NAME"_.png", k*10, k*10, 3, pixels, 0);
	}*/

	// 訓練
	printf("Starting training...\n");
//	CatsEye_train(&cat, xh, x, sample, 10/*repeat*/, sample/*random batch*/, sample/10);
	CatsEye_train(&cat, xh, x, sample, 45/*repeat*/, sample/*random batch*/, 0);
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

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

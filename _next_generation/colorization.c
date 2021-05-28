//---------------------------------------------------------
//	Cat's eye
//
//		©2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc colorized.c -o colorized -lm -Ofast -fopenmp -lgomp
// clang colorized.c -o colorized -lm -Ofast

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CATS_USE_ADAM
#define ADAM_BETA1	0.1
#define ADAM_BETA2	0.999

#define ETA		5e-4
//#define ETA		1e-3
#define BATCH		1
//#define BATCH	128

#define NAME		"colorization"
#define CATS_CHECK
#define CATS_USE_FLOAT
#include "catseye.h"

#define K		32
#define SIZE		(K*K*3)
#define SAMPLE		10000
#define LATENT_DIM	4096

int main()
{
	int k = K;
	int size = SIZE;
	int label = 10;
	int sample = SAMPLE;

	// http://www.net.c.dendai.ac.jp/~tamura/
	// https://qiita.com/MuAuan/items/e5f3e67ee24a776380aa
#if 0
	CatsEye_layer u[] = {
		{   K*K, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=32, .sx=K, .sy=K, .ich=1 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=64 },
		{     0, CATS_ACT_RRELU, .name="encoder" },

		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=16 },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=12 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3, .name="decoder" },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
#else
	CatsEye_layer u[] = {
		{   K*K, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=32, .sx=K, .sy=K, .ich=1, .name="input" },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=64 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=2, .padding=1, .ch=128 },
		{     0, CATS_ACT_RRELU, .name="encoder" },

		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=32 },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=16 },

//		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=12 },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=8 },
		{     0, CATS_ACT_RRELU },
//		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3, .name="decoder" },
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=2, .name="decoder" },

		{     0, CATS_CONCAT, .layer="input", .order=1 },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
#endif
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = CatsEye_loadCifar("data_batch_1.bin", size, 1, sample, &t);
	free(t);
	printf("OK\n");
	real *y = malloc(K*K*SAMPLE *sizeof(real));
	real *p = y;
	for (int i=0; i<SAMPLE; i++) {
		real *r = x +i*SIZE;
		real *g = x +i*SIZE +K*K;
		real *b = x +i*SIZE +K*K*2;
		for (int n=0; n<K*K; n++) {
			real _r = *r;
			real _g = *g;
			real _b = *b;
			*r++ = *p++ = 0.298912* _r +0.586611* _g +0.114478* _b;	// CCIR Rec.601
			*g++ = -0.1687* _r -0.3313* _g +0.500 * _b;
			*b++ =  0.500 * _r -0.4187* _g -0.0813* _b;
		}
	}
	/*{
		uint8_t *pixels = calloc(1, size*100);
		for (int i=0; i<50; i++) {
			CatsEye_visualize(y+K*K*i, 32*32, 32, &pixels[50*32*32*1 +(i/10)*K*K*10+(i%10)*K*1], K*10, 1);
		}
		stbi_write_png("colorized2.png", k*10, k*10, 1, pixels, 0);
		free(pixels);
	}*/

	// 訓練
	printf("Starting training...\n");
//	CatsEye_train(&cat, x, x, sample, 100/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
//	CatsEye_train(&cat, y, x, sample, 10/*repeat*/, 1000/*random batch*/, 0);
//	CatsEye_train(&cat, y, x, sample, 50/*repeat*/, 1000/*random batch*/, 0);
	CatsEye_train(&cat, y, x, sample, 100/*repeat*/, 1000/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	uint8_t *pixels = calloc(1, size*100);
	for (int i=0; i<50; i++) {
//		CatsEye_forward(&cat, x+size*i);
		CatsEye_forward(&cat, y+K*K*i);
		CatsEye_visualizeYUV(cat.layer[cat.layers-1].x, 32*32, 32, &pixels[(i/10)*size*10+(i%10)*k*3], k*10);
		CatsEye_visualizeYUV(x+size*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10);

//		CatsEye_visualize(cat.layer[cat.layers-1].x, 32*32, 32, &pixels[(i/10)*size*10+(i%10)*k*3], k*10, 3);
//		CatsEye_visualize(x+size*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10, 3);
		//CatsEye_visualize(y+K*K*i, 32*32, 32, &pixels[50*32*32*3 +(i/10)*size*10+(i%10)*k*3], k*10, 1);
	}
	stbi_write_png("colorized.png", k*10, k*10, 3, pixels, 0);
	free(pixels);

	free(y);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

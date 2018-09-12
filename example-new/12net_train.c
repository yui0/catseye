//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc 12net_train.c -o 12net_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang 12net_train.c -o 12net_train -lm -Ofast -march=native -funroll-loops
// ./make_dataset ./datasets/

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

int main()
{
	int k = 12;
	int size = 12*12*3;	// 入力層
//	int sample = 18925;
	int sample = 789431;	// 93%(10), 94%(100)

	// https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
	CatsEye_layer u[] = {	// 12-net 99.9%(1000)
		{   size, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, /*.padding=1,*/ .ch=16, .ich=3 },
		{      0, CATS_MAXPOOL,                  0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_LINEAR, CATS_ACT_LEAKY_RELU,  0.01 },	// 16 outputs (Fully-connected layer)
		{     16, CATS_LINEAR,   CATS_ACT_IDENTITY,  0.01 },	// face / non-face
		{      2, CATS_LOSS,         CATS_LOSS_0_1,  0.01 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = _CatsEye_loadCifar("face-train.bin", 12*12*3, sizeof(int16_t), sample, &t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	_CatsEye_train(&cat, x, t, sample, 1000/*repeat*/, 100/*random batch*/);
//	_CatsEye_train(&cat, x, t, sample, 1000/*repeat*/, sample/10/*random batch*/);
//	_CatsEye_train(&cat, x, t, sample, 280/*repeat*/, sample/10/*random batch*/);
	_CatsEye_train(&cat, x, t, sample, 10/*repeat*/, sample/10/*random batch*/);
	printf("Training complete\n");
	CatsEye_saveCats(&cat, "12net.cats");
	_CatsEye_saveJson(&cat, "12net.json");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
//				CatsEye_visualize(x+size*i, size, k*3, &pixels[(c/10)*size*10+(c%10)*k*3], k*3*10);
				real *xx = &x[size*i];
				unsigned char *p = &pixels[(c/10)*size*10+(c%10)*k*3];
				for (int y=0; y<k; y++) {
					for (int x=0; x<k; x++) {
						p[(y*k*10+x)*3  ] = xx[y*k+x] * 255.0;
						p[(y*k*10+x)*3+1] = xx[k*k+y*k+x] * 255.0;
						p[(y*k*10+x)*3+2] = xx[2*k*k+y*k+x] * 255.0;
					}
				}
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png("12net_train_wrong.png", k*10, k*10, 3, pixels, 0);

	memset(pixels, 0, size*100);
	int i = frand()*sample;
	CatsEye_visualize(x+size*i, size/3, k, &pixels[100* k*10 +100], k*10);
	int p = _CatsEye_predict(&cat, x+size*i);
	printf("predict %d -> %d [%d]\n", i, p, t[i]);
	for (int n=0; n<cat.layers; n++) {
		CatsEye_layer *l = &cat.layer[n];
		if (l->type == CATS_LINEAR) {
			continue;
		}

		int mch = l->ch > 10 ? 10 : l->ch;
		for (int ch=0; ch<mch; ch++) {
			CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pixels[n*(k+2) +ch*(k+2)*k*10], k*10);
		}
	}
	stbi_write_png("12net_train_predict.png", k*10, k*10, 1, pixels, 0);

/*	memset(pixels, 0, size*100);
	for (int i=0; i<10; i++) {
		CatsEye_forward(&cat, x+size*i);

		// 初段フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 1, 0, &pixels[i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 1, &pixels[k*k*10+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 2, &pixels[k*k*10*2+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 3, &pixels[k*k*10*3+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 4, &pixels[k*k*10*4+i*k], k*10);

		// 2段目フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 2, 0, &pixels[k*k*10*5+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 1, &pixels[k*k*10*6+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 2, &pixels[k*k*10*7+i*k], k*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 3, &pixels[k*k*10*8+i*k], k*10);
	}
	// フィルタ
	for (int i=0; i<u[CHANNEL+LPLEN]; i++) {
		int n = (u[KSIZE+LPLEN]+2);
		CatsEye_visualizeUnits(&cat, 1, 0, i, &pixels[k*k*10*(9+(i*n)/(k*10))+(i*n)%(k*10)], k*10);
	}
	stbi_write_png("cifar10_train.png", k*10, k*10, 1, pixels, 0);*/
	free(pixels);

	free(t);
	free(x);
//	CatsEye__destruct(&cat);

	{
	char *name = "mikarika.jpg";
	int w, h, bpp;
	uint8_t *pixels = stbi_load(name, &w, &h, &bpp, 3);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	real pix[12*12*3];
	for (int y=0; y<h; y+=4) {
		for (int x=0; x<w; x+=4) {
			for (int sy=0; sy<12; sy++) {
				for (int sx=0; sx<12; sx++) {
					pix[12*sy+sx]         = pixels[(w*(y+sy)+x+sx)*3  ] /255.0;
					pix[12*sy+sx+12*12]   = pixels[(w*(y+sy)+x+sx)*3+1] /255.0;
					pix[12*sy+sx+12*12*2] = pixels[(w*(y+sy)+x+sx)*3+2] /255.0;
				}
			}
			int p = _CatsEye_predict(&cat, pix);
			if (p) {
				for (int sy=0; sy<12; sy++) {
					for (int sx=0; sx<12; sx++) {
						pixels[(w*(y+sy)+x+sx)*3] = 0;
					}
				}
			}
		}
	}
	stbi_write_jpg("mikarika_r.jpg", w, h, bpp, pixels, 0);
	}
	CatsEye__destruct(&cat);

	return 0;
}

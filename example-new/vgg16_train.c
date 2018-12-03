//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc vgg16_train.c -o vgg16_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang vgg16_train.c -o vgg16_train -lm -Ofast -march=native -funroll-loops

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

//#define ETA	0.001
#define ETA	1e-5

int main()
{
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;

	CatsEye_layer u[] = {
		{  size, CATS_PADDING, .sx=32, .sy=32, .ich=3, .padding=1 },
//		{  size, CATS_PADDING, .sx=224, .sy=224, .ich=3, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=64, .ich=3 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=64 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // 64,112x112

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=128 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=128 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // 128,56x56

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=256 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=256 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=256 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // 256,28x28

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // 512,14x14

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=512 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // 512,7x7

		{     0, CATS_LINEAR, 0.01, .outputs=4096 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_LINEAR, 0.01, .outputs=4096 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_LINEAR, 0.01, .outputs=label },
		{     0, _CATS_ACT_SIGMOID }, // <- slow learning, but good recognize
		//{ label, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = _CatsEye_loadCifar("../example/data_batch_1.bin", 32*32*3, 1, sample, &t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	_CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	_CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 100/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	static int result[10][10];
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i</*sample*/100; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);
		result[t[i]][p]++;
		if (p==t[i]) r++;
		else {
			if (c<100) {
//				CatsEye_visualize(x+size*i, size, k*3, &pixels[(c/10)*size*10+(c%10)*k*3], k*3*10);

				_CatsEye_visualize(x+size*i, 32*32, 32, &pixels[(c/10)*size*10+(c%10)*k*3], k*10, 3);
				/*real *xx = &x[size*i];
				unsigned char *p = &pixels[(c/10)*size*10+(c%10)*k*3];
				for (int y=0; y<k; y++) {
					for (int x=0; x<k; x++) {
						p[(y*k*10+x)*3  ] = xx[y*k+x] * 255.0;
						p[(y*k*10+x)*3+1] = xx[k*k+y*k+x] * 255.0;
						p[(y*k*10+x)*3+2] = xx[2*k*k+y*k+x] * 255.0;
					}
				}*/

				//CatsEye_visualize(cat.o[0], k*k, k, &pixels[(c/10)*k*k*10+(c%10)*k], k*10);
//				CatsEye_visualizeUnits(&cat, 0, 0, 0, &pixels[(c/10)*k*k*10+(c%10)*k], k*10);
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	for (int i=0; i<10; i++) {
		for (int j=0; j<10; j++) {
			printf("%3d ", result[i][j]);
		}
		printf("\n");
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png("vgg16_train_wrong.png", k*10, k*10, 3/*bpp*/, pixels, 0);

	int n[10]; // 10 classes 
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);

//		CatsEye_visualize(x+size*i, size/3, k, &pixels[p*k*k*10+(n[p]%10)*k], k*10);
		_CatsEye_visualize(x+size*i, 32*32, 32, &pixels[(p*k*k*10+(n[p]%10)*k)*3], k*10, 3);
		n[p]++;
	}
//	stbi_write_png("vgg16_classify.png", k*10, k*10, 1, pixels, 0);
	stbi_write_png("vgg16_classify.png", k*10, k*10, 3, pixels, 0);

	memset(pixels, 0, size*100);
	/*for (int i=0; i<10*10; i++)*/ {
		int p = _CatsEye_predict(&cat, x/*+size*i*/);

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
	}
	stbi_write_png("vgg16_predict.png", k*10, k*10, 1, pixels, 0);
	free(pixels);

	free(t);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

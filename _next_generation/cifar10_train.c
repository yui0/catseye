//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_train.c -o cifar10_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang cifar10_train.c -o cifar10_train -lm -Ofast -march=native -funroll-loops

#define CATS_USE_FLOAT
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#define ETA	0.01
#define ETA	0.0001

int main()
{
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;		// 出力層
	int sample = 10000;

	CatsEye_layer u[] = {	// 52.4%(10), 85.1%(100), 95.8%(1000), 99.7%(2000)
		{  size, CATS_PADDING, .sx=32, .sy=32, .ich=3, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=10, .ich=3 },
//		{     0, CATS_BATCHNORMAL },
		{     0, _CATS_ACT_LEAKY_RELU },
//		{     0, _CATS_ACT_RRELU, .min=-0.1, .max=0.1 },

		{     0, CATS_PADDING, .padding=1 },
		{     0, CATS_CONV,   ETA, .ksize=3, .stride=1, .ch=10, },
//		{     0, CATS_BATCHNORMAL },
		{     0, _CATS_ACT_LEAKY_RELU },
//		{     0, _CATS_ACT_RRELU, .min=-0.1, .max=0.1 },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
//		{     0, CATS_AVGPOOL, .ksize=2, .stride=2 },

		{     0, CATS_LINEAR, 0.01, .outputs=256 },
//		{     0, CATS_BATCHNORMAL },
		{     0, _CATS_ACT_LEAKY_RELU },
//		{     0, _CATS_ACT_RRELU, .min=-0.1, .max=0.1 },

		{     0, CATS_LINEAR, 0.01, .outputs=label },
//		{     0, CATS_BATCHNORMAL },
		//{     0, _CATS_ACT_SIGMOID }, // <- slow learning, but good recognize
		{     0, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};
	CatsEye cat = { .batch=1 };
	CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
	real *x = CatsEye_loadCifar("data_batch_1.bin", 32*32*3, 1, sample, &t);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	printf("Training complete\n");
//	CatsEye_save(&cat, "cifar10.weights");
//	CatsEye_saveJson(&cat, "cifar10.json");
//	CatsEye_saveBin(&cat, "cifar10.bin");

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
				CatsEye_visualize(x+size*i, 32*32, 32, &pixels[(c/10)*size*10+(c%10)*k*3], k*10, 3);
				/*real *xx = &x[size*i];
				uint8_t *p = &pixels[(c/10)*size*10+(c%10)*k*3];
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
	stbi_write_png("cifar10_train_wrong.png", k*10, k*10, 3/*bpp*/, pixels, 0);

	int n[10]; // 10 classes 
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
		int p = CatsEye_predict(&cat, x+size*i);

		CatsEye_visualize(x+size*i, 32*32, 32, &pixels[(p*k*k*10+(n[p]%10)*k)*3], k*10, 3);
		n[p]++;
	}
//	stbi_write_png("cifar10_classify.png", k*10, k*10, 1, pixels, 0);
	stbi_write_png("cifar10_classify.png", k*10, k*10, 3, pixels, 0);

	memset(pixels, 0, size*100);
	/*for (int i=0; i<10*10; i++)*/ {
		int p = CatsEye_predict(&cat, x/*+size*i*/);

		for (int n=0; n<cat.layers; n++) {
			CatsEye_layer *l = &cat.layer[n];
			if (l->type == CATS_LINEAR) {
				continue;
			}

			int mch = l->ch > 10 ? 10 : l->ch;
			for (int ch=0; ch<mch; ch++) {
				CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pixels[n*(k+2) +ch*(k+2)*k*10], k*10, 1);
			}
		}
	}
	stbi_write_png("cifar10_predict.png", k*10, k*10, 1, pixels, 0);

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

/*	memset(pixels, 0, size*100);
	for (int i=0; i<10; i++) {
		memset(cat.o[layers-1], 0, label);
		cat.o[layers-1][i] = 1;
		CatsEye_backpropagate(&cat, layers-2);

		CatsEye_visualizeUnits(&cat, 0, 1, 0, &pixels[i*k], k*10);
	}
	stbi_write_png("cifar10_gen.png", k*10, k*10, 1, pixels, k*10);*/
	free(pixels);

	free(t);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

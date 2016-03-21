//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_train.c -o mnist_cnn_train -lm -fopenmp -lgomp
// clang mnist_cnn_train.c -o mnist_cnn_train -lm -Ofast
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 784;	// 入力層ユニット(28x28)
	int hidden = 64;	// 隠れ層ユニット
	int label = 10;	// 出力層ユニット(0-9)
	int sample = 60000;

#if 0
	int ch = 5;		// チャネル
	int k = 5;		// カーネルサイズ
	int s = 28-(k/2)*2;	// 出力サイズ
	int u[] = {		// 95.98%[k:5] 95.28%[k:3]
		0, 0, 1, size, 0, 0, 0, 0,
		CATS_CONV, CATS_ACT_TANH, 5, 5*s*s, 28, 28, k, 1,	// tanh, 5ch, stride 1
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#endif
#if 0
	int ch = 5;		// チャネル
	int k = 3;		// カーネルサイズ
	int s = 28-(k/2)*2;	// 出力サイズ
	int k2 = 3;		//!!
	int s2 = s/k2;		//!!
	int u[] = {		//
		0, 0, 1, size, 0, 0, 0, 0,
		CATS_CONV, CATS_ACT_TANH, 5, 5*s*s, 28, 28, k, 1,	// tanh, 5ch, stride 1
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,	// 97.5%
//		CATS_LINEAR, CATS_ACT_SOFTMAX, 1, label, 0, 0, 0, 0,	// 86.8%
	};
#else
	int ch = 5;		// チャネル
	int k = 5;		// 1段目のカーネルサイズ
	int s = 28-(k/2)*2;	// 1段目の出力サイズ
	int k2 = 3;		// 2段目のカーネルサイズ
	int s2 = s/k2;		// 2段目の出力サイズ
	int u[] = {
		0, 0, 1, size,    0, 0, 0, 0,

//		CATS_CONV, CATS_ACT_TANH, 1, 26*26, 28, 28, 3, 1,		// 1ch tanh 28x28,26x26 90.3%
//		CATS_CONV, CATS_ACT_TANH, 1, 24*24, 26, 26, 3, 1,

//		CATS_CONV, CATS_ACT_TANH, 2, 2*26*26, 28, 28, 3, 1,		// 2ch tanh 28x28,26x26 94.5%
//		CATS_CONV, CATS_ACT_TANH, 2, 2*24*24, 26, 26, 3, 1,

//		CATS_CONV, CATS_ACT_TANH, 5, 5*26*26, 28, 28, 3, 1,		// 5ch tanh 28x28,26x26 95.9%
//		CATS_CONV, CATS_ACT_TANH, 5, 5*24*24, 26, 26, 3, 1,

//		CATS_CONV, CATS_ACT_TANH, 5, 5*24*24, 28, 28, 5, 1,		// 98.1%
//		CATS_CONV, CATS_ACT_TANH, 5, 5*22*22, 24, 24, 3, 1,
		//CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,		// 97.6%

//		CATS_CONV, CATS_ACT_TANH, 5, 5*24*24, 28, 28, 5, 1,		// 95.9%
//		CATS_MAXPOOL, 0, 5, 5*12*12, 24, 24, 2, 2,
		//CATS_MAXPOOL, 0, 5, 5*8*8, 24, 24, 3, 3,			// 91.9%

		// http://www.kanadas.com/weblog/2015/06/1_1_2.html
//		CATS_CONV, CATS_ACT_TANH, 5, 5*24*24, 28, 28, 5, 1,		// 95.0%
//		CATS_MAXPOOL, 0, 5, 5*8*8, 24, 24, 3, 3,
//		CATS_CONV, CATS_ACT_TANH, 16, 16*6*6, 8, 8, 3, 1,
		//CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,

		// https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html
		CATS_CONV, CATS_ACT_RELU, 8, 8*24*24, 28, 28, 5, 1,		// 97.1%
		CATS_MAXPOOL, 0, 8, 8*12*12, 24, 24, 2, 2,
		CATS_CONV, CATS_ACT_RELU, 16, 16*8*8, 12, 12, 5, 1,
		//CATS_MAXPOOL, 0, 16, 16*4*4, 8, 8, 2, 2,

//		CATS_CONV, CATS_ACT_TANH, 6, 6*24*24, 28, 28, 5, 1,		// 95.3%
//		CATS_MAXPOOL, 0, 6, 6*12*12, 24, 24, 2, 2,
//		CATS_CONV, CATS_ACT_TANH, 16, 16*10*10, 12, 12, 3, 1,
//		CATS_MAXPOOL, 0, 16, 16*5*5, 10, 10, 2, 2,
		//CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,

//		CATS_CONV, CATS_ACT_RELU, 5, 5*24*24, 28, 28, 5, 1,		// 96.6%
//		CATS_CONV, CATS_ACT_RELU, 5, 5*24*24, 24, 24, 1, 1,		// 96.8% Cascaded 1x1 Convolution [Network in Network]
		//CATS_CONV, CATS_ACT_RELU, 5, 5*24*24, 24, 24, 1, 1,		// CCCP layer
//		CATS_CONV, CATS_ACT_RELU, 5, 5*22*22, 24, 24, 3, 1,

//		CATS_CONV, CATS_ACT_RELU, 5, 5*26*26, 28, 28, 3, 1,		// 97.5%
//		CATS_MAXPOOL, 0, 5, 5*13*13, 26, 26, 2, 2,
//		CATS_CONV, CATS_ACT_RELU, 5, 5*11*11, 13, 13, 3, 1,
//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,

//		CATS_CONV, CATS_ACT_SIGMOID, ch, ch*s*s, 28, 28, k, 1,		// tanh, 5ch, stride 1
//		CATS_CONV, CATS_ACT_TANH, ch, ch*s*s, 28, 28, k, 1,		// tanh, 5ch, stride 1 (pl:89%,k11:96%,97.8%)
//		CATS_CONV, CATS_ACT_RELU, ch, ch*s*s, 28, 28, k, 1,		// ReLU, 5ch, stride 1 (pl:71%,98.3%)
//		CATS_CONV, CATS_ACT_LEAKY_RELU, ch, ch*s*s, 28, 28, k, 1,	// Leaky ReLU, 5ch, stride 1 (pl:61%,k11:58%,95%)
//		CATS_MAXPOOL, 0, ch, ch*s2*s2, s, s, k2, k2,			// maxpooling (93.2%)

//		CATS_CONV, CATS_ACT_TANH, 5, 5*6*6, s2, s2, 3, 1,
//		CATS_CONV, CATS_ACT_TANH, 1, 1*6*6, s2, s2, 3, 1,
//		CATS_CONV, CATS_ACT_TANH, 16, 16*6*6, s2, s2, 3, 1,

//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 200, 0, 0, 0, 0,
//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 64, 0, 0, 0, 0,		// linear only:95.6%
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, label, 0, 0, 0, 0,
	};
#endif
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
//	CatsEye__construct(&cat, size, hidden, label, 0);
	CatsEye__construct(&cat, 0, 0, layers, u);

	double *x = malloc(sizeof(double)*size*sample);	// 訓練データ
	int t[sample];			// ラベルデータ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist.weights");
//	CatsEye_saveJson(&cat, "mnist_cnn_train.json");
//	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
				CatsEye_visualize(cat.o[0], 28*28, 28, &pixels[(c/10)*28*28*10+(c%10)*28], 28*10);
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png("mnist_cnn_train_wrong.png", 28*10, 28*10, 1, pixels, 28*10);
	memset(pixels, 0, size*100);

	for (int i=0; i<10; i++) {
		CatsEye_forward(&cat, x+size*i);

		// 初段フィルタ出力
		CatsEye_visualize(cat.o[1], s*s, s, &pixels[i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s], s*s, s, &pixels[28*28*10+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*2], s*s, s, &pixels[28*28*10*2+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*3], s*s, s, &pixels[28*28*10*3+i*28], 28*10);
		CatsEye_visualize(&cat.o[1][s*s*4], s*s, s, &pixels[28*28*10*4+i*28], 28*10);

		// 2段目フィルタ出力
		CatsEye_visualize(cat.o[2], s2*s2, s2, &pixels[28*28*10*5+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][s2*s2], s2*s2, s2, &pixels[28*28*10*6+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][s2*s2*2], s2*s2, s2, &pixels[28*28*10*7+i*28], 28*10);
		CatsEye_visualize(&cat.o[2][s2*s2*3], s2*s2, s2, &pixels[28*28*10*8+i*28], 28*10);
	}
	// フィルタ
	for (int i=0; i<ch; i++) {
		CatsEye_visualize(&cat.w[0][(k*k+1)*i], k*k, k, &pixels[28*10*28*9 + i*(k+2)], 28*10);
	}
	stbi_write_png("mnist_cnn_train.png", 28*10, 28*10, 1, pixels, 28*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_cnn_train.c -o mnist_cnn_train -lm -Ofast -fopenmp -lgomp
// clang mnist_cnn_train.c -o mnist_cnn_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL` -mf16c
#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main()
{
	int size = 784;	// 入力層ユニット(28x28)
	int label = 10;	// 出力層ユニット(0-9)
	int sample = 60000;

	CatsEye_layer u[] = {	// 99.19% (100)
		{  size, CATS_CONV, 0.01, .ksize=7, .stride=1, .ch=32 },
//		{     0, _CATS_ACT_RELU }, // minus??
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
//		{     0, CATS_LINEAR, CATS_ACT_SIGMOID, 0.01 },
		{     0, CATS_LINEAR, 0.01 },
//		{ label, _CATS_ACT_SIGMOID },
		//{ label, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	int16_t t[sample];				// ラベルデータ
	unsigned char *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("../example/train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	fp = fopen("../example/train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	_CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 1500/*random batch*/, sample/10);
	_CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 1500/*random batch*/, 0);
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist.weights");
//	CatsEye_saveJson(&cat, "mnist_cnn_train.json");
//	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		else {
			if (c<100) {
				CatsEye_visualize(x+size*i, size, 28, &pixels[(c/10)*size*10+(c%10)*28], 28*10);
/*				real *xx = &x[size*i];
				unsigned char *p = &pixels[(c/10)*size*10+(c%10)*k*3];
				for (int y=0; y<k; y++) {
					for (int x=0; x<k; x++) {
						p[(y*k*10+x)*3  ] = xx[y*k+x] * 255.0;
						p[(y*k*10+x)*3+1] = xx[k*k+y*k+x] * 255.0;
						p[(y*k*10+x)*3+2] = xx[2*k*k+y*k+x] * 255.0;
					}
				}*/
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png("mnist_cnn_train_wrong.png", 28*10, 28*10, 1, pixels, 28*10);
	memset(pixels, 0, size*100);

/*	for (int i=0; i<10; i++) {
//		CatsEye_forward(&cat, x+size*i);
		_CatsEye_forward(&cat, x+size*i);

		// 初段フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 1, 0, &pixels[i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 1, &pixels[28*28*10+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 2, &pixels[28*28*10*2+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 3, &pixels[28*28*10*3+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 1, 4, &pixels[28*28*10*4+i*28], 28*10);

		// 2段目フィルタ出力
		CatsEye_visualizeUnits(&cat, 0, 2, 0, &pixels[28*28*10*5+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 1, &pixels[28*28*10*6+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 2, &pixels[28*28*10*7+i*28], 28*10);
		CatsEye_visualizeUnits(&cat, 0, 2, 3, &pixels[28*28*10*8+i*28], 28*10);
	}
	// フィルタ
	for (int i=0; i<u[CHANNEL+LPLEN]; i++) {
		int n = (u[KSIZE+LPLEN]+2);
		CatsEye_visualizeUnits(&cat, 1, 0, i, &pixels[28*28*10*(9+(i*n)/(28*10))+(i*n)%(28*10)], 28*10);
	}
	stbi_write_png("mnist_cnn_train.png", 28*10, 28*10, 1, pixels, 28*10);*/
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

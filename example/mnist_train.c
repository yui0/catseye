//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_train.c -o mnist_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang mnist_train.c -o mnist_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL`
#define CATS_USE_FLOAT
#include "../catseye.h"

int main()
{
	int size = 784;	// 入力層ユニット(28x28)
	int hidden = 200;	// 隠れ層ユニット
	int label = 10;	// 出力層ユニット(0-9)
	int sample = 60000;

	CatsEye cat;
	CatsEye__construct(&cat, size, hidden, label, 0);

	// 訓練データの読み込み
	printf("Training data:\n");
	int *t;
	real *x = CatsEye_loadMnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", sample, &t);

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 0.01);	// 94.38%(100), 97.1%(1000)
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist.weights");
	CatsEye_saveJson(&cat, "mnist.json");
//	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

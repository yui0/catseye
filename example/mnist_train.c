//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// clang mnist_train.c -o mnist_train -lm
#include "../catseye.h"

int main()
{
	int size = 784;	// 入力層ユニット(28x28)
	//int hidden = 100;	// 隠れ層ユニット
	int hidden = 64;	// 隠れ層ユニット
	int label = 10;	// 出力層ユニット(0-9)
	int sample = 60000;

	CatsEye cat;
	CatsEye__construct(&cat, size, hidden, label, 0);

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

	// 多層パーセプトロンの訓練
	// 繰り返しの回数
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 0.01);
	printf("Training complete\n");
	CatsEye_save(&cat, "mnist.weights");
	CatsEye_saveJson(&cat, "mnist.json");
	CatsEye_saveBin(&cat, "mnist.bin");

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	free(x);
	return 0;
}

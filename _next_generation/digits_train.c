//---------------------------------------------------------
//	Cat's eye
//
//		©2016,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc digits_train.c -o digits_train -lm -fopenmp -lgomp
// clang digits_train.c -o digits_train -lm

//#define CATS_USE_FLOAT
#include "catseye.h"

#define ETA 0.001	// batch 1

int main()
{
	int size = 64;		// 入力層64ユニット(8x8)
	int label = 10;		// 出力層10ユニット(0-9)
	int sample = 1797;

	CatsEye_layer u[] = {	// 97.8% (100)
		{ size, CATS_LINEAR, ETA },
		{  100, _CATS_ACT_TANH },
		{  100, CATS_LINEAR, ETA },
//		{    0, _CATS_ACT_SIGMOID },
//		{    0, _CATS_ACT_SOFTMAX },
		{label, CATS_LOSS_0_1 },
	};
	CatsEye cat = { .batch=1 };	// 97.8%
	CatsEye__construct(&cat, u);

	real x[size*sample];	// 訓練データ
	int16_t t[sample];	// ラベルデータ

	// CSVの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("digits.csv", "r");
	if (fp==NULL) return -1;
	int n=0;
	while (feof(fp)==0) {
		// データの読み込み
		if (n<3) printf("[%4d]  ", n);
		for (int j=0; j<size; j++) {
			if (!fscanf(fp, "%lf,", x+size*n+j)) {
				// 0-1に正規化
				x[size*n+j] /= 16.0;
			}
			if (n<3) printf("%6.2f  ", x[size*n+j]);
		}
		fscanf(fp, "%hd", t+n);
		if (n<3) printf("<%d>\n", t[n]);
		n++;
	}
	fclose(fp);
	printf("\n");
	sample = n;

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, t, sample-1, 100/*repeat*/, 0.01);
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, sample, sample/10);
	printf("Training complete\n");
//	CatsEye_save(&cat, "digits.weights");
//	CatsEye_saveJson(&cat, "digits.json");

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	CatsEye__destruct(&cat);

	return 0;
}

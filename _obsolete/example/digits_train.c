//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc digits_train.c -o digits_train -lm -fopenmp -lgomp
// clang digits_train.c -o digits_train -lm
#include "../catseye.h"

int main()
{
	int size = 64;		// 入力層64ユニット(8x8)
	int label = 10;	// 出力層10ユニット(0-9)
	int sample = 1797;

	CatsEye cat;
	CatsEye__construct(&cat, size, 100, label, 0);

	double x[size*sample];	// 訓練データ
	int t[sample];		// ラベルデータ

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
		fscanf(fp, "%d", t+n);
		if (n<3) printf("<%d>\n", t[n]);
		n++;
	}
	fclose(fp);
	sample = n;

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample-1, 100/*repeat*/, 0.01);
	printf("Training complete\n");
	CatsEye_save(&cat, "digits.weights");
	CatsEye_saveJson(&cat, "digits.json");

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

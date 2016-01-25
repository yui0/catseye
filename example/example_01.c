//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// clang example_01.c -o example_01 -lm
// ref. http://kivantium.hateblo.jp/entry/2014/12/22/004640
#include "../catseye.h"

int main()
{
	// サンプルの数
	int sample = 150;
	// 入力ベクトルの次元
	int size = 2;
	// ラベルの種類
	int label = 3;

	// 入力ユニット2つ, 隠れユニット3つ, 出力ユニット3つの多層パーセプトロンを作る
	CatsEye cat;
	CatsEye__construct(&cat, size, 3, label);

	// 訓練データ
	double x[size*sample];
	// ラベルデータ
	int t[sample];

	// CSVの読み込み
	FILE *fp = fopen("example_01.csv", "r");
	if (fp==NULL) {
		return -1;
	}
	for (int i=0; i<sample; i++) {
		// データの読み込み
		for (int j=0; j<size; j++) {
			fscanf(fp, "%lf,", x+size*i+j);
		}
		// ラベルの読み込み
		fscanf(fp, "%d", t+i);
	}
	fclose(fp);

	// 多層パーセプトロンの訓練
	// 繰り返しの回数
	int repeat = 500;
	CatsEye_train(&cat, x, t, sample, repeat, 0.1);

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	return 0;
}

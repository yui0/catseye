//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// clang weather.c -o weather -lm
#include "../catseye.h"

int main()
{
	int size = 4;	// 入力ベクトルの次元
	int label = 2;	// ラベルの種類
	int sample = 10000;

	// 入力ユニット4つ, 隠れユニット3つ, 出力ユニット2つの多層パーセプトロンを作る
	CatsEye cat;
	CatsEye__construct(&cat, size, 3, label, 0);

	// 訓練データ
	double x[size*sample];
	// ラベルデータ
	int t[sample];

	// CSVの読み込み
	FILE *fp = fopen("weather_sapporo.train", "r");
	if (fp==NULL) {
		return -1;
	}
	int n=0;
	while (fscanf(fp, "%d ", t+n) != EOF) {
		if (t[n]<0) t[n]=0;

		// データの読み込み
		for (int j=0; j<size; j++) {
			int f;
			fscanf(fp, "%d:%lf,", &f, x+size*n+j);
		}
		//printf("%d %lf,%lf,%lf,%lf\n", t[n], x[size*n], x[size*n+1], x[size*n+2], x[size*n+3]);
		n++;
	}
	fclose(fp);
	sample = n;

	// 多層パーセプトロンの訓練
	// 繰り返しの回数
	int repeat = 1000;
	CatsEye_train(&cat, x, t, sample, repeat, 0.001);

	// 結果の表示
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		if (p==t[i]) r++;
		printf("%d -> %d\n", p, t[i]);
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);

	CatsEye_save(&cat, "weather_sapporo.weights");

	return 0;
}

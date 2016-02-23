//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc quadratic.c -o quadratic -lm -Ofast -fopenmp -lgomp
// clang quadratic.c -o quadratic -lm -Ofast
// ps2pdf quadratic.ps 
#define CATS_LOSS_MSE
#include "../catseye.h"
#include "../pssub.h"

int main()
{
	int size = 1;		// 入力層
	int sample = 360;

	CatsEye cat;
	CatsEye__construct(&cat, size, 100/*hidden*/, 1/*output*/, 0);

	// 訓練データ
	double x[sample];
	for (int i=0; i<sample; i++) x[i] = (double)i/sample*2 -1;
	// ラベルデータ
	double t[sample];
	for (int i=0; i<sample; i++) t[i] = x[i] * x[i];

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 2000/*repeat*/, 0.01);
	printf("Training complete\n");
	//CatsEye_save(&cat, "quadratic.weights");

	// 結果の表示
/*	FILE *fp = fopen("quadratic.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+size*i);
		fprintf(fp, "%d, %lf\n", i, cat.o3[0]);
	}
	fclose(fp);*/

	// postscriptで表示
	PS_init("quadratic.ps");
	PS_viewport(0.2, 0.2, 0.8, 0.8);
	PS_xyworld(-1.2, -0.2, 1.2, 1.2);
	PS_linewidth(1.0);
	PS_linety(1);
	PS_setgray(0.8);
	PS_xaxis(-1.0, 0.0, 1.0, 0.0, 2, 10);

	PS_setrgb(0.0, 0.0, 1.0);	// quadratic function
//	PS_plot(x[0], t[0], 3);
//	for (int i=1; i<sample; i++) {
	for (int i=1; i<sample; i+=16) {
//		PS_plot(x[i], t[i], 2);
		PS_circ(x[i], t[i], 0.05);
		PS_stroke();
	}
//	PS_stroke();

	PS_linewidth(1.5);		// output
	//PS_setgray(0.0);
	PS_setrgb(1.0, 0.0, 0.0);
	CatsEye_forward(&cat, x);
	PS_plot(x[0], cat.o3[0], 3);
	for (int i=1; i<sample; i++) {
		CatsEye_forward(&cat, x+size*i);
		PS_plot(x[i], cat.o3[0], 2);
	}
	PS_stroke();
	PS_fin();
	system("ps2pdf quadratic.ps");

	return 0;
}

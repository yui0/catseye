//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc sin.c -o sin -lm -fopenmp -lgomp
// clang sin.c -o sin -lm
// ps2pdf sin.ps (ghostscript-core)
#include "../catseye.h"
#include "../pssub.h"

int main()
{
	int sample = 360;
	int u[] = {
		CATS_LINEAR, CATS_ACT_IDENTITY, 1/*ch*/,   1/*input*/,  0, 0, 0, 500,
		CATS_LINEAR, CATS_ACT_SIGMOID,  1/*ch*/, 100/*hidden*/, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_IDENTITY, 1/*ch*/,   1/*output*/, 0, 0, 0, CATS_LOSS_MSE,
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	// 訓練データ
	double x[sample];
	for (int i=0; i<sample; i++) x[i] = 2.0*M_PI / sample * i;
	double t[sample];
	for (int i=0; i<sample; i++) t[i] = sin(x[i]);

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 2000/*repeat*/, 0.01);
	printf("Training complete\n");
//	CatsEye_save(&cat, "sin.weights");

	// 結果の表示
	FILE *fp = fopen("/tmp/sin.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		fprintf(fp, "%d, %lf\n", i, cat.o[2][0]);
	}
	fclose(fp);

	// postscriptで表示
	PS_init("/tmp/sin.ps");
	PS_viewport(0.2, 0.2, 0.8, 0.8);
	PS_xyworld(-0.5, -1.2, 2.0*M_PI+0.5, 1.2);
	PS_linewidth(1.0);
	PS_linety(1);
	PS_setgray(0.8);
	PS_xaxis(0.0, 0.0, 2.0*M_PI, 0.0, 2, 10);
/*	PS_rect(0.0, -1.0, 2.0*M_PI, 1.0);
	PS_line(0.0, 0.0, 2.0*M_PI, 0.0);
	PS_line(2.0*M_PI/2, -1.0, 2.0*M_PI/2, 1.0);
	PS_stroke();*/

	PS_setrgb(0.0, 0.0, 1.0);	// sin
//	PS_plot(x[0], t[0], 3);
//	for (int i=1; i<sample; i++) {
	for (int i=1; i<sample; i+=8) {
//		PS_plot(x[i], t[i], 2);
		PS_circ(x[i], t[i], 0.05);
		PS_stroke();
	}
//	PS_stroke();

	PS_linewidth(1.5);		// output
	//PS_setgray(0.0);
	PS_setrgb(1.0, 0.0, 0.0);
	CatsEye_forward(&cat, x);
	PS_plot(x[0], cat.o[2][0], 3);
	for (int i=1; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		PS_plot(x[i], cat.o[2][0], 2);
	}
	PS_stroke();
	PS_fin();
	system("ps2pdf /tmp/sin.ps sin_.pdf");

	CatsEye__destruct(&cat);

	return 0;
}

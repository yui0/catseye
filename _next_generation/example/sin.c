//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc sin.c -o sin -lm -fopenmp -lgomp
// clang sin.c -o sin -lm
// ps2pdf sin.ps (ghostscript-core)

//#define CATS_TEST
//#define CATS_USE_FLOAT
//#define CATS_OPENGL
#include "catseye.h"
#include "pssub.h"

#define ETA	1e-2

int main()
{
	int sample = 360;

/*	CatsEye_layer u[] = {
		{   1, CATS_LINEAR, ETA }, // input layer
		{ 100, CATS_ACT_SIGMOID },
		{ 100, CATS_LINEAR, ETA }, // hidden layer
		{   1, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};*/
	CatsEye_layer u[] = {
		{   1, CATS_LINEAR, ETA },
		{  10, CATS_ACT_TANH },
		{  10, CATS_LINEAR, ETA },
		{   1, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};
/*	CatsEye_layer u[] = {
		{   1, CATS_LINEAR, ETA },
		{  10, CATS_ACT_RELU },
		{  10, CATS_LINEAR, ETA },
		{  10, CATS_ACT_RELU },
		{  10, CATS_LINEAR, ETA },
		{   1, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};*/
	CatsEye cat = CATS_INIT;
	CatsEye__construct(&cat, u);

	// 訓練データ
	real x[sample];
	for (int i=0; i<sample; i++) x[i] = 2.0*M_PI / sample * i;
	real t[sample];
	for (int i=0; i<sample; i++) t[i] = sin(x[i]);

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
//	CatsEye_train(&cat, x, t, sample, 2000/*repeat*/, sample, 0);
	CatsEye_train(&cat, x, t, sample, 200/*repeat*/, sample, 0);
//	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, sample, 0);
//	CatsEye_train(&cat, x, t, 1, 1/*repeat*/, 1, 0);
	printf("Training complete\n");
//	CatsEye_save(&cat, "sin.weights");

	// 結果の表示
	FILE *fp = fopen("/tmp/sin.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		fprintf(fp, "%d, %lf\n", i, cat.layer[cat.layers-1].x[0]);
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
	PS_plot(x[0], cat.layer[cat.layers-1].x[0], 3);
	for (int i=1; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		PS_plot(x[i], cat.layer[cat.layers-1].x[0], 2);
	}
	PS_stroke();
	PS_fin();
	system("ps2pdf /tmp/sin.ps");

	CatsEye__destruct(&cat);

	return 0;
}

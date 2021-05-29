//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc wavecnn.c -o wavecnn -lm -fopenmp -lgomp
// clang wavecnn.c -o wavecnn -lm
// ps2pdf wavecnn.ps (ghostscript-core)
#define CATS_USE_FLOAT
#include "../catseye.h"
#include "../pssub.h"

#define TIME	441

int main()
{
	int sample = TIME*1000;

/*	CatsEye_layer u[] = {
//		{     0, CATS_CONV,   0, 0.01, .ksize=1, .stride=1, .ch=1, .ich=1 },
		{  TIME, CATS_LINEAR, 0, 0.01 },
		{   896, _CATS_ACT_RELU },
//		{   256, _CATS_ACT_SIGMOID },
		{   896, CATS_LINEAR, 0, 0.01 },
		{   256, _CATS_ACT_RELU },
//		{   256, _CATS_ACT_SIGMOID },
		{   256, CATS_LINEAR, 0, 0.01 },
//		{   256, _CATS_ACT_SIGMOID },
//		{   256, _CATS_ACT_SOFTMAX },
		{   256, CATS_LOSS_0_1 },
	};*/
/*	CatsEye_layer u[] = {
		{  TIME, CATS_LINEAR, 0.01 },
//		{   512, _CATS_ACT_SIGMOID },
		{   512, _CATS_ACT_LEAKY_RELU },
		{   512, CATS_LINEAR, 0.01 },
//		{   128, _CATS_ACT_SIGMOID },
		{   128, _CATS_ACT_LEAKY_RELU },
		{   128, CATS_LINEAR, 0.01 },
//		{  2048, _CATS_ACT_LEAKY_RELU },
//		{  2048, CATS_LINEAR, 0.01 },
		{   256, _CATS_ACT_SIGMOID },
//		{   256, _CATS_ACT_SOFTMAX },
		{   256, CATS_LOSS_0_1 },
	};*/
	CatsEye_layer u[] = {
		{  TIME, CATS_LINEAR, 0.01, .outputs=512 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_LINEAR, 0.01, .outputs=512 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_LINEAR, 0.01, .outputs=1024 },
		{     0, _CATS_ACT_LEAKY_RELU },
		{     0, CATS_LINEAR, 0.01, .outputs=256 },
//		{     0, _CATS_ACT_SIGMOID },
		{     0, _CATS_ACT_SOFTMAX },
		{   256, CATS_LOSS_0_1 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データ
	real x[sample];		// -1 - 1
//	for (int i=0; i<sample; i++) x[i] = sin(2.0*M_PI / (TIME/2) * i *10);
	for (int i=0; i<sample; i++) x[i] = sin(2.0*M_PI / TIME * i);
	//for (int i=0; i<sample; i++) x[i] = sin(2.0*M_PI / (TIME/2) * i);
	//for (int i=0; i<sample; i++) x[i] = sin(2.0*M_PI / TIME * i*10);
	int16_t t[sample];	// 0 - 255
	for (int i=0; i<sample; i++) t[i] = (int16_t)(x[i+TIME+1]*127+128);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	cat.slide = 1;
	_CatsEye_train(&cat, x, t, sample/TIME, 20/*repeat*/, 1000/*random batch*/, 0);
	printf("Training complete\n");

	// 結果の表示
	FILE *fp = fopen("/tmp/sin.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<TIME; i++) {
//		_CatsEye_forward(&cat, x+i);
//		fprintf(fp, "%d, %lf\n", i, cat.layer[cat.layers-1].x[0]);
		int p = _CatsEye_predict(&cat, x+i);
		fprintf(fp, "%d, %d [%lf/%d]\n", i, p, x[i], t[i]);
	}
	fclose(fp);

	// postscriptで表示 [x: -0.5 - 2*3.14+0.5, y: -1.2 - 1.2 ]
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
	for (int i=1; i<TIME; i+=8) {
//		PS_circ(x[i], t[i]/256.0, 0.05);
		PS_circ(2*M_PI*i/TIME, t[i]/128.0-1, 0.05);
		PS_stroke();
	}
//	PS_stroke();

	PS_linewidth(1.5);		// output
	//PS_setgray(0.0);
	PS_setrgb(1.0, 0.0, 0.0);
//	_CatsEye_forward(&cat, x);
//	PS_plot(x[0], cat.layer[cat.layers-1].x[0], 3);
	int p = _CatsEye_predict(&cat, x);
//	PS_plot(x[0], p/256.0, 3);
	PS_plot(2*M_PI*0/TIME, p/128.0-1, 3);
	for (int i=1; i<TIME; i++) {
//		_CatsEye_forward(&cat, x+i);
//		PS_plot(x[i], cat.layer[cat.layers-1].x[0], 2);
		int p = _CatsEye_predict(&cat, x+i);
//		PS_plot(x[i], p/256.0, 2);
		PS_plot(2*M_PI*i/TIME, p/128.0-1, 2);
	}
	PS_stroke();
	PS_fin();
	system("ps2pdf /tmp/sin.ps");

	CatsEye__destruct(&cat);

	return 0;
}

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
#ifdef PS
#include "pssub.h"
#else
#include "svg.h"
#endif

#define ETA	1e-2
#define BATCH	1
#define DATA	64
#define SAMPLE	(360+DATA)

int main()
{
	const int sample = 360;

/*	CatsEye_layer u[] = {
		{   1, CATS_LINEAR, ETA },
		{  10, CATS_ACT_TANH },
		{  10, CATS_LINEAR, ETA },
		{   1, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};*/
	// https://www.kabuku.co.jp/developers/visualize_intermidiate_conv1d_output
/*	CatsEye_layer u[] = {
		// 64(1ch) >> 32(64ch)
		{DATA, CATS_CONV1D, ETA, .ksize=9, .stride=2, .padding=4, .ch=64 },
		{   0, CATS_ACT_RELU },

		// 32(64ch) >> 16(64ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=2, .padding=4, .ch=64 },
		{   0, CATS_ACT_RELU },

		// 16(64ch) >> 16(32ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=1, .padding=4, .ch=32 },
		{   0, CATS_ACT_RELU },

		// 16(32ch) >> 16(1ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=1, .padding=4, .ch=1 },
		{   0, CATS_ACT_TANH },

		{   0, CATS_LINEAR, ETA },

		{   1, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE
	};*/
	// https://qiita.com/niisan-tokyo/items/a94dbd3134219f19cab1
	CatsEye_layer u[] = {
		// 64(1ch) >> 32(64ch)
		{  64, CATS_CONV1D, ETA, .ksize=9, .stride=2, .padding=4, .ch=64 },
		{   0, CATS_ACT_RELU },

		// 32(64ch) >> 16(64ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=2, .padding=4, .ch=64 },
		{   0, CATS_ACT_RELU },

		// 16(64ch) >> 16(32ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=1, .padding=4, .ch=32 },
		{   0, CATS_ACT_RELU },

		// 16(32ch) >> 16(1ch)
		{   0, CATS_CONV1D, ETA, .ksize=9, .stride=1, .padding=4, .ch=1 },

//		{   0, CATS_ACT_TANH },
//		{  16, CATS_LOSS_IDENTITY_MSE }, // 回帰なのでMSE

		{   0, CATS_LINEAR, ETA },
		{   1, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データ
	real x[SAMPLE];
	for (int i=0; i<SAMPLE; i++) x[i] = 2.0*M_PI / sample * i;
	real t[SAMPLE];
	for (int i=0; i<SAMPLE; i++) t[i] = sin(x[i]);

	// 多層パーセプトロンの訓練
	printf("Starting training...\n");
	cat.slide = 1;
	CatsEye_train(&cat, t, t+DATA, sample, 10/*repeat*/, sample, 0);
//	CatsEye_train(&cat, x, t, sample-64, 100/*repeat*/, sample, 0);
//	CatsEye_train(&cat, x, t, sample/64, 100/*repeat*/, sample/64, 0);
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

#ifdef PS
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
//	CatsEye_forward(&cat, x);
	CatsEye_forward(&cat, t);
	PS_plot(x[0], cat.layer[cat.layers-1].x[0], 3);
	for (int i=1; i<sample; i++) {
//		CatsEye_forward(&cat, x+i);
		CatsEye_forward(&cat, t+i);
		PS_plot(x[i], cat.layer[cat.layers-1].x[0], 2);
	}
	PS_stroke();
	PS_fin();
	system("ps2pdf /tmp/sin.ps");
#else
	real ax[DATA+sample], z[DATA+sample];
	for (int i=0; i<DATA; i++) {
		z[i] = t[i];
		ax[i] = i; // axis
	}
	for (int i=0; i<sample; i++) {
//		CatsEye_forward(&cat, x+i);
//		CatsEye_forward(&cat, t+i);
		CatsEye_forward(&cat, z+i);
		z[DATA+i] = cat.layer[cat.layers-1].x[0];
		ax[DATA+i] = DATA+i;
	}

	// SVGで表示
	svg *psvg = svg_create(512, 512);
	//if (!psvg) return;
	svg_scatter(psvg, ax, z, DATA+sample, 0, 0, SVG_FRAME);
	svg_scatter(psvg, ax, t/*+64*/, DATA+sample, 0, 0, SVG_NONFILL);
	svg_finalize(psvg);
	svg_save(psvg, "/tmp/sin.svg");
	svg_free(psvg);
#endif

	CatsEye__destruct(&cat);

	return 0;
}

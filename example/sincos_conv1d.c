//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc sin.c -o sin -lm -fopenmp -lgomp
// clang sin.c -o sin -lm
// ps2pdf sin.ps (ghostscript-core)

#ifdef PS
#include "pssub.h"
#else
#include "svg.h"
#endif

#define ETA	1e-2
#define BATCH	1
#define DATA	64
#define SAMPLE	(360+DATA)

#define NAME	"sincos"

//#define CATS_TEST
//#define CATS_USE_FLOAT
//#define CATS_OPENGL
#include "catseye.h"

int main()
{
	const int sample = 360;

	// https://qiita.com/niisan-tokyo/items/a94dbd3134219f19cab1
	CatsEye_layer u[] = {
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

		{   0, CATS_LINEAR, ETA },
		{   1, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データ
	real x[SAMPLE];
	for (int i=0; i<SAMPLE; i++) x[i] = 2.0*M_PI / sample * i;
	real t[SAMPLE];
//	for (int i=0; i<SAMPLE; i++) t[i] = sin(x[i]);
	for (int i=0; i<SAMPLE; i++) t[i] = (sin(x[i]) +sin(3*x[i]) +sin(10*x[i]) +cos(5*x[i]) +cos(7*x[i]));

	// 多層パーセプトロンの訓練
	printf("Starting training...\n");
	cat.slide = 1;
	CatsEye_train(&cat, t, t+DATA, sample, 30/*repeat*/, sample, 0);
	printf("Training complete\n");
//	CatsEye_save(&cat, NAME".weights");

	// 結果の表示
	FILE *fp = fopen("/tmp/"NAME".csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		fprintf(fp, "%d, %lf\n", i, cat.layer[cat.layers-1].x[0]);
	}
	fclose(fp);

	real ax[DATA+sample], z[DATA+sample];
	for (int i=0; i<DATA; i++) {
		z[i] = t[i];
		ax[i] = i; // axis
	}
	for (int i=0; i<sample; i++) {
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
	svg_save(psvg, "/tmp/"NAME".svg");
	svg_free(psvg);

	CatsEye__destruct(&cat);

	return 0;
}

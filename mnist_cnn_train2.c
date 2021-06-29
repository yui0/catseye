//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_cnn_train.c -o mnist_cnn_train -lm -Ofast -fopenmp -lgomp
// clang mnist_cnn_train.c -o mnist_cnn_train -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL` -mf16c

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include"svg.h"

//#define CATS_USE_ADAM
//#define ETA	1e-4
#define ETA	0.01	// 77.0% (batch 1 with SGD)
//#define ETA	0.001	// batch 64 with SGD
//#define BATCH	64
#define BATCH	1
#define NAME	"mnist_cnn_train2"

#define CATS_CHECK
#define CATS_USE_FLOAT
//#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"

int main()
{
	const int size = 28*28;
	const int label = 10;
	const int sample = 60000;

	// https://cpp-learning.com/center-loss/
	// https://testpy.hatenablog.com/entry/2020/01/04/225231
	CatsEye_layer u[] = {
		{  size, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=/*32*/16 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=5, .stride=2, .padding=2, .ch=/*32*/16 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=/*64*/32 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=5, .stride=2, .padding=2, .ch=/*64*/32 },
		{     0, CATS_ACT_RRELU },
/*		{     0, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=128 },
		{     0, CATS_ACT_RRELU },
		{     0, CATS_CONV, ETA, .ksize=5, .stride=2, .padding=2, .ch=128 },
		{     0, CATS_ACT_RRELU },*/
		{     0, CATS_LINEAR, ETA, .outputs=2, .name="feature" }, // ip1
		{     0, CATS_ACT_RRELU },
		{     0, CATS_LINEAR, ETA, .outputs=label }, // ip2
		{ label, CATS_ACT_SOFTMAX },
		{ label, CATS_SOFTMAX_CE },
	};
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	int16_t t[sample];				// ラベルデータ
	uint8_t *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data: loading...");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
//	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0 *2-1; // calc err!
	fclose(fp);
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);
	printf("OK\n");

	// 訓練
	printf("Starting training...\n");
	CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 1500/*random batch*/, sample/10);
//	CatsEye_train(&cat, x, t, sample, 20/*repeat*/, 1500/*random batch*/, sample/10);
//	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 1500/*random batch*/, sample/10);
	printf("Training complete\n");

	// 結果の表示
	static int result[10][10];
	uint8_t *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = CatsEye_predict(&cat, x+size*i);
		result[t[i]][p]++;
		if (p==t[i]) r++;
		else {
			if (c<100) {
				CatsEye_visualize(x+size*i, size, 28, &pixels[(c/10)*size*10+(c%10)*28], 28*10, 1);
			}
			c++;
		}
//		printf("%d -> %d\n", p, t[i]);
	}
	printf("\n");
	for (int i=0; i<10; i++) {
		for (int j=0; j<10; j++) {
			printf("%3d ", result[i][j]);
		}
		printf("\n");
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
//	stbi_write_png(NAME"_wrong.png", 28*10, 28*10, 1, pixels, 28*10);
	memset(pixels, 0, size*100);

	for (int i=0; i<10; i++) {
		CatsEye_forward(&cat, x+size*i);

		// 初段フィルタ出力
		CatsEye_layer *l = &cat.layer[0];
		int s = l->ox*l->oy;
		CatsEye_visualize(l->z+s*0, s, l->ox, &pixels[28*28*10*0+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*1, s, l->ox, &pixels[28*28*10*1+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*2, s, l->ox, &pixels[28*28*10*2+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*3, s, l->ox, &pixels[28*28*10*3+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*4, s, l->ox, &pixels[28*28*10*4+i*28], 28*10, 1);

		// 2段目フィルタ出力
		l = &cat.layer[1];
		s = l->ox*l->oy;
		CatsEye_visualize(l->z+s*0, s, l->ox, &pixels[28*28*10*5+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*1, s, l->ox, &pixels[28*28*10*6+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*2, s, l->ox, &pixels[28*28*10*7+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*3, s, l->ox, &pixels[28*28*10*8+i*28], 28*10, 1);
		CatsEye_visualize(l->z+s*4, s, l->ox, &pixels[28*28*10*9+i*28], 28*10, 1);
	}
	// フィルタ
	{
	CatsEye_layer *l = &cat.layer[1];
	for (int i=0; i<l->ch; i++) {
		int s = l->ksize*l->ksize;
		int n = l->ksize+2;
		CatsEye_visualize(l->w+s*i, s, l->ksize, &pixels[28*28*10*(9+(i*n)/(28*10))+(i*n)%(28*10)], 28*10, 1);
	}
	}
	stbi_write_png(NAME"_train.png", 28*10, 28*10, 1, pixels, 28*10);
	free(pixels);

	// 潜在変数
	double xs[sample], ys[sample];
	for (int i=0; i<sample/50; i++) {
		CatsEye_forward(&cat, x+size*i);
		int e = CatsEye_getLayer(&cat, "feature");
		CatsEye_layer *l = &cat.layer[e];
		xs[i] = l->z[0];
		ys[i] = l->z[1];
	}
	svg *psvg = svg_create(512, 512);
	//if (!psvg) return;
	svg_scatter(psvg, xs, ys, sample/50, t, 10);
	svg_finalize(psvg);
	svg_save(psvg, NAME".svg");
	svg_free(psvg);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

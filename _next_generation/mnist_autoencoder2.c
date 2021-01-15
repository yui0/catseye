//---------------------------------------------------------
//	Cat's eye
//
//		©2016,2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast -fopenmp -lgomp
// clang mnist_autoencoder.c -o mnist_autoencoder -lm -Ofast

#define CATS_USE_FLOAT
#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#include "pbPlots.h"
#include"svg.h"

#define ETA 1e-3

int main()
{
	const int wh = 28;
	const int size = 28*28;	// 入出力層(28x28)
	const int sample = 60000;

	// https://www.sejuku.net/blog/63331
	CatsEye_layer u[] = {	// epoch 20/ eta 1e-5
		{ size, CATS_LINEAR, ETA },
		{   32, CATS_ACT_RELU },
		{   32, CATS_LINEAR, ETA },
		{    2, CATS_ACT_RELU },
		{    2, CATS_LINEAR, ETA },
		{   32, CATS_ACT_RELU },
		{   32, CATS_LINEAR, ETA },
		{ size, CATS_ACT_SIGMOID },
		{ size, CATS_LOSS_MSE },
	};
//	CatsEye cat = { .batch=1 };	// 0.0%
	CatsEye cat = { .batch=256 };	// 0.0%
	CatsEye__construct(&cat, u);

	int16_t t[sample];	// ラベルデータ
	real *x = malloc(sizeof(real)*size*sample);	// 訓練データ
	uint8_t *data = malloc(sample*size);

	// 訓練データの読み込み
	printf("Training data:\n");
	FILE *fp = fopen("train-images-idx3-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 16, 1, fp);		// header
	fread(data, size, sample, fp);	// data
	for (int i=0; i<sample*size; i++) x[i] = data[i] / 255.0;
	fclose(fp);
	fp = fopen("train-labels-idx1-ubyte", "rb");
	if (fp==NULL) return -1;
	fread(data, 8, 1, fp);		// header
	fread(data, 1, sample, fp);	// data
	for (int i=0; i<sample; i++) t[i] = data[i];
	fclose(fp);
	free(data);

	// 多層パーセプトロンの訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, x, sample, 20/*epoch*/, sample, 0);
//	CatsEye_train(&cat, x, x, sample-1, 100, 1e-2);		// SGD[h64/3.3], SGD[h64+s/7.4/OK]
	printf("Training complete\n");
//	CatsEye_save(&cat, "mnist_autoencoder.weights");
//	CatsEye_saveJson(&cat, "mnist_autoencoder.json");

	// 結果の表示
	uint8_t *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		CatsEye_forward(&cat, x+size*i);

		CatsEye_layer *l = &cat.layer[cat.end-1];
		double mse = 0;
		uint8_t *p = &pixels[(i/10)*size*10 + (i%10)*wh];
		for (int j=0; j<size; j++) {
			p[(j/wh)*wh*10+(j%wh)] = l->z[j] * 255.0;
			mse += (x[size*i+j]-l->z[j])*(x[size*i+j]-l->z[j]);

			p[5*size*10+(j/wh)*wh*10+(j%wh)] = x[size*i+j] * 255.0;
		}
//		printf("%d mse %lf\n", t[i], mse);
	}
	stbi_write_png("mnist_autoencoder2.png", wh*10, wh*10, 1, pixels, wh*10);

	memset(pixels, 0, size*100);
/*	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, 28, &pixels[(n/10)*28*28*10 + (n%10)*28], 28*10);
	}
	stbi_write_png("mnist_autoencoder_weights.png", 28*10, 28*10, 1, pixels, 28*10);*/
	free(pixels);

	// 潜在変数
	double xs[sample], ys[sample];
/*	double xmin = 0;
	double xmax = 0;
	double ymin = 0;
	double ymax = 0;*/
	for (int i=0; i<sample/50; i++) {
		CatsEye_forward(&cat, x+size*i);
		CatsEye_layer *l = &cat.layer[3];
		xs[i] = l->z[0];
		ys[i] = l->z[1];
/*		if (xmin>xs[i]) xmin = xs[i];
		if (xmax<xs[i]) xmax = xs[i];
		if (ymin>ys[i]) ymin = ys[i];
		if (ymax<ys[i]) ymax = ys[i];*/
	}
/*	double ax = 1.0/(xmax-xmin)*360;
	double ay = 1.0/(ymax-ymin)*350;
	for (int i=0; i<sample/100; i++) {
		xs[i] = (xs[i]-xmin)*ax;
		ys[i] = (ys[i]-ymin)*ay;
	}*/
	svg *psvg = svg_create(512, 512);
	//if (!psvg) return;
	svg_scatter(psvg, xs, ys, sample/50, t, 10);
/*	svg_rectangle(psvg, 360, 350, 40, 0, "#C0C0FF", "black", 1, 0, 0); // 枠線
	svg_line(psvg, "#000000", 2, 100, 0, 100, 350); // 縦線
	svg_line(psvg, "#000000", 2, 160, 0, 160, 350);
	svg_line(psvg, "#000000", 2, 220, 0, 220, 350);
	svg_line(psvg, "#000000", 2, 280, 0, 280, 350);
	svg_line(psvg, "#000000", 2, 340, 0, 340, 350);
	svg_line(psvg, "#000000", 2, 40, 50, 400, 50); // 横線
	svg_line(psvg, "#000000", 2, 40, 100, 400, 100);
	svg_line(psvg, "#000000", 2, 40, 150, 400, 150);
	svg_line(psvg, "#000000", 2, 40, 200, 400, 200);
	svg_line(psvg, "#000000", 2, 40, 250, 400, 250);
	svg_line(psvg, "#000000", 2, 40, 300, 400, 300);
	svg_text(psvg, 0, 350, "sans-serif", 16, "#000000", "#000000", "0"); // 縦の目盛り
	svg_text(psvg, 0, 300, "sans-serif", 16, "#000000", "#000000", "50");
	svg_text(psvg, 0, 250, "sans-serif", 16, "#000000", "#000000", "100");
	svg_text(psvg, 0, 200, "sans-serif", 16, "#000000", "#000000", "150");
	svg_text(psvg, 0, 150, "sans-serif", 16, "#000000", "#000000", "200");
	svg_text(psvg, 0, 100, "sans-serif", 16, "#000000", "#000000", "250");
	svg_text(psvg, 0, 50, "sans-serif", 16, "#000000", "#000000", "300");
	svg_text(psvg, 60, 380, "sans-serif", 16, "#000000", "#000000", "10"); // 横の目盛り
	svg_text(psvg, 120, 380, "sans-serif", 16, "#000000", "#000000", "20");
	svg_text(psvg, 180, 380, "sans-serif", 16, "#000000", "#000000", "30");
	svg_text(psvg, 240, 380, "sans-serif", 16, "#000000", "#000000", "40");
	svg_text(psvg, 300, 380, "sans-serif", 16, "#000000", "#000000", "50");
	svg_text(psvg, 360, 380, "sans-serif", 16, "#000000", "#000000", "60");
	for (int i=0; i<sample/100; i++) {
		char col[256];
		sprintf(col, "#%02d0000", t[i]*20);
		svg_circle(psvg, "#000080", 0, col, 3, xs[i], ys[i]);
	}*/
	svg_finalize(psvg);
	svg_save(psvg, "mnist_autoencoder2.svg");
	svg_free(psvg);
#if 0
	RGBABitmapImageReference *canvasReference = CreateRGBABitmapImageReference();
	double xs[sample], ys[sample];
	for (int i=0; i<sample/100; i++) {
		CatsEye_forward(&cat, x+size*i);
		CatsEye_layer *l = &cat.layer[3];
//		xs[i] = l->z[0];
//		ys[i] = l->z[1];
		RGBA col;
		col.a = 0.8;
		col.r = t[i]*20;
		col.g = 0;
		col.b = 0;
		DrawPixel(canvasReference, l->z[0], l->z[1], &col);
	}
//	RGBABitmapImageReference *canvasReference = CreateRGBABitmapImageReference();
//	DrawScatterPlot(canvasReference, 600, 400, xs, sample/100, ys, sample/100);
	size_t length;
	double *pngdata = ConvertToPNG(&length, canvasReference->image);
	WriteToFile(pngdata, length, "mnist_autoencoder2_z.png");
	DeleteImage(canvasReference->image);
#endif

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

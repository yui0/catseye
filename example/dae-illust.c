//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc dae-illust.c -o dae-illust -lm -Ofast -fopenmp -lgomp
// clang dae-illust.c -o dae-illust -lm -Ofast
//#define CATS_AUTOENCODER
#define CATS_DENOISING_AUTOENCODER
#include "../catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <dirent.h>
//#include <sys/stat.h>
double *load(char *path, int w, int h, int *c)
{
	int n = 0;
	struct dirent *de;
	DIR *d = opendir(path);
	while ((de = readdir(d))) {
/*		struct stat st;
		if (stat(de->d_name, &st) == 0 && st.st_mode & S_IFREG)*/ n++;
	}
	n -= 2;

	*c = n;
	double *x = malloc(sizeof(double)*3*w*h*n);
	double *p = x;

	rewinddir(d);
	while ((de = readdir(d))) {
		unsigned char *pixels;
		int width, height, bpp;
		char name[256];
		snprintf(name, 256, "%s/%s", path, de->d_name);
		pixels = stbi_load(name, &width, &height, &bpp, 3);
		if (!pixels) { (*c)--; continue; }

		int x = (width-w)/2;
		int y = (height-h)/2;
		//printf("%x  %s %d  %d %d  %d %d\n",pixels,name,n,width,height,x,y);
		for (int i=0; i<h; i++) {
			for (int j=0; j<w; j++) {
				int a = (y+i)*width+(x+j);
				*p++ = pixels[a*3  ]/255.0;
//				*p++ = pixels[a*3+1]/255.0;
//				*p++ = pixels[a*3+2]/255.0;
			}
		}
		free(pixels);
	}
	closedir(d);

	return x;
}

int main()
{
	int w = 120;
	int h = 120;
	int size = 1*w*h;	// 入出力層
	int hidden = 256;	// 100
//	int hidden = 128;	// 40, ×100
//	int hidden = 64;	// ×
	int sample;

	// データの読み込み
	double *x = load("./illust/", w, h, &sample);
//	sample = 100;

#if 1
	int u[] = {
		0, 0, 1, size, 0, 0, 0, sample,				// mini batch size is 40 by random
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, hidden, 0, 0, 0, 0,
		CATS_LINEAR, 0, 1, size, 0, 0, 0, 1,			// use mse (sakura)
	};
#else
	int u[] = {
//		0, 0, 1, size, 0, 0, 0, sample<70?sample:70,		// mini batch size is 50 by random
		0, 0, 1, size, 0, 0, 0, sample,				// mini batch size is 40 by random

		CATS_CONV, CATS_ACT_TANH, 4, 0, 0, 0, 3, 1,
//		CATS_CONV, CATS_ACT_TANH, 1, 0, 0, 0, 3, 1,

//		CATS_CONV, CATS_ACT_LEAKY_RELU, 32, 0, 0, 0, 3, 1,
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 1, 0, 0, 0, 3, 1,

//		CATS_CONV, CATS_ACT_LEAKY_RELU, 4, 0, 0, 0, 3, 1,
//		CATS_MAXPOOL, 0, 4, 0, 0, 0, 2, 2,
//		CATS_CONV, CATS_ACT_LEAKY_RELU, 2, 0, 0, 0, 3, 1,
		//CATS_MAXPOOL, 0, 2, 0, 0, 0, 2, 2,

//		CATS_LINEAR, CATS_ACT_SIGMOID, 1, size, 0, 0, 0, 1,	// use mse
		CATS_LINEAR, 0, 1, size, 0, 0, 0, 1,			// use mse (sakura)
	};
#endif
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, x, sample, 1000/*repeat*/, 0.001);
	printf("Training complete\n");
//	CatsEye_save(&cat, "dae-illust.weights");
//	CatsEye_saveJson(&cat, "dae-illust.json");

	// 結果の表示
	unsigned char *pixels = malloc(size*100);
	for (int i=0; i<50; i++) {
		double mse = 0;
		CatsEye_forward(&cat, x+size*i);

		CatsEye_visualize(cat.o[2], w*h, w, &pixels[(i/10)*w*h*10 + (i%10)*w], w*10);
		CatsEye_visualize(&x[size*i], w*h, w, &pixels[5*w*h*10+(i/10)*w*h*10 + (i%10)*w], w*10);
		for (int j=0; j<size; j++) {
			mse += (x[size*i+j]-cat.o[2][j])*(x[size*i+j]-cat.o[2][j]);
		}
		printf("mse %lf\n", mse/size);
	}
	stbi_write_png("dae-illust.png", w*10, w*10, 1, pixels, w*10);

	memset(pixels, 0, size*100);
	int m = (hidden<100 ? hidden : 100);
	for (int n=0; n<m; n++) {
		CatsEye_visualizeWeights(&cat, n, w, &pixels[(n/10)*w*h*10 + (n%10)*w], w*10);
//		CatsEye_visualize(&cat.w[0][n], w*h, w, &pixels[(n/10)*w*h*10 + (n%10)*w], w*10);
	}
	stbi_write_png("dae-illust_weights.png", w*10, w*10, 1, pixels, w*10);

	srand((unsigned)(time(0)));
	memset(pixels, 0, size*100);
	for (int i=0; i<100; i++) {
		for (int n=0; n<hidden; n++) {
			cat.o[1][n] = (rand()/(RAND_MAX+1.0)) * 1.0;
//			cat.o[1][n] = rand();
		}
		CatsEye_propagate(&cat, 1);

		CatsEye_visualize(cat.o[2], w*h, w, &pixels[(i/10)*w*h*10 + (i%10)*w], w*10);
	}
	stbi_write_png("dae-illust_gen.png", w*10, w*10, 1, pixels, w*10);
	free(pixels);

	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

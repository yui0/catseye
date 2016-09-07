//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc paint.c -o paint -lm -Ofast -fopenmp -lgomp -march=native -funroll-loops `pkg-config --libs --cflags OpenCL`
// clang paint.c -o paint -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL`
#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main(int argc,char *argv[])
{
	if (argc<2) {
		printf("%s *.png\n", argv[0]);
		return 0;
	}

	// データの読み込み
	unsigned char *pixels;
	int width, height, bpp;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 3);
	if (!pixels) return 0;
	int sample = width * height;

	numerus x[sample*2];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			x[(i*width+j)*2  ] = i / (numerus)width;
			x[(i*width+j)*2+1] = j / (numerus)height;
		}
	}
	numerus t[sample*3];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			t[(i*width+j)*3  ] = pixels[(i*width+j)*3  ] / 255.0;
			t[(i*width+j)*3+1] = pixels[(i*width+j)*3+1] / 255.0;
			t[(i*width+j)*3+2] = pixels[(i*width+j)*3+2] / 255.0;
		}
	}
	free(pixels);

	int neurons = 20;	// 20
#if 1
	int u[] = {	// http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
		0, 0, 1, 2/*xy*/, 0, 0, 0, sample,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,	// sigmoid×
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 3/*RGB*/, 0, 0, 0, 1,
	};
#else
	int u[] = {
		0, 0, 1, 2/*xy*/, 0, 0, 0, sample,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_TANH, 1, neurons, 0, 0, 0, 0,
//		CATS_LINEAR, 0, 1, 3/*RGB*/, 0, 0, 0, 1,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 3/*RGB*/, 0, 0, 0, 1,
	};
#endif
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	numerus scale = 2.0;
	width *= scale;
	height *= scale;

	/*unsigned char * */pixels = malloc(width*height*3);
	for (int n=0; n<500; n++) {
		// 訓練
		printf("Starting training using (stochastic) gradient descent\n");
		CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 0.001);
		printf("Training complete\n");
//		CatsEye_save(&cat, "paint.weights");
//		CatsEye_saveJson(&cat, "paint.json");

		// 結果の表示
		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
//				CatsEye_forward(&cat, x+2*(i*width+j));
				numerus xx[2];
				xx[0] = i / (numerus)width;
				xx[1] = j / (numerus)height;
				CatsEye_forward(&cat, xx);
				pixels[(i*width+j)*3  ] = cat.o[layers-1][0] * 255.0;
				pixels[(i*width+j)*3+1] = cat.o[layers-1][1] * 255.0;
				pixels[(i*width+j)*3+2] = cat.o[layers-1][2] * 255.0;
			}
		}
		char name[256];
		snprintf(name, 256, "/tmp/paint%04d.png", n);
		stbi_write_png(name, width, height, 3, pixels, 0);
	}
	printf("ffmpeg -r 30 -i paint%%4d.png -pix_fmt yuv420p paint.mp4\n");
	system("ffmpeg -r 30 -i /tmp/paint%4d.png -pix_fmt yuv420p paint.mp4");
	system("mv /tmp/paint0499.png .");
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}

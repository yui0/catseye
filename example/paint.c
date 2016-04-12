//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc paint.c -o paint -lm -Ofast -fopenmp -lgomp
// clang paint.c -o paint -lm -Ofast
#include "../catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

int main(int argc,char *argv[])
{
	// データの読み込み
	unsigned char *pixels;
	int width, height, bpp;
	if (argc<2) return 0;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 3);
	int sample = width * height;

	double x[sample*2];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			x[(i*width+j)*2  ] = i / (double)width;
			x[(i*width+j)*2+1] = j / (double)height;
		}
	}
	double t[sample*3];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			t[(i*width+j)*3  ] = pixels[(i*width+j)*3  ] / 255.0;
			t[(i*width+j)*3+1] = pixels[(i*width+j)*3+1] / 255.0;
			t[(i*width+j)*3+2] = pixels[(i*width+j)*3+2] / 255.0;
		}
	}
	free(pixels);

	int u[] = {
		0, 0, 1, 2/*xy*/, 0, 0, 0, sample,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_LEAKY_RELU, 1, 20, 0, 0, 0, 0,

/*		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,*/
/*		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, 20, 0, 0, 0, 0,*/
		CATS_LINEAR, 0, 1, 3/*RGB*/, 0, 0, 0, 1,
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	CatsEye_train(&cat, x, t, sample, 100/*repeat*/, 0.001);
	printf("Training complete\n");
//	CatsEye_save(&cat, "paint.weights");
//	CatsEye_saveJson(&cat, "paint.json");

	// 結果の表示
	/*unsigned char * */pixels = malloc(width*height*3);
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			CatsEye_forward(&cat, x+2*(i*width+j));
			pixels[(i*width+j)*3  ] = cat.o[layers-1][0] * 255.0;
			pixels[(i*width+j)*3+1] = cat.o[layers-1][1] * 255.0;
			pixels[(i*width+j)*3+2] = cat.o[layers-1][2] * 255.0;
		}
	}
	stbi_write_png("paint.png", width, height, 3, pixels, 0);
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}

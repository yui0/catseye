//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// clang mnist_test.c -o mnist_test -lm
#include "../catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

int main(int argc, char *argv[])
{
	if (argc!=2) return 1;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, 0, "mnist_test.weights");

	// 入力データ
	double x[cat.in];

	unsigned char *pixels;
	int width, height, bpp;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 1/*Grey*/);
	int max = 0;
	for (int i=0; i<28*28; i++) if (pixels[i]>max) max = pixels[i];
	for (int i=0; i<28*28; i++) {
		x[i] = pixels[i]/(double)max;
	}

	int p = CatsEye_predict(&cat, x);
	printf("%d\n", p);

	CatsEye__destruct(&cat);

	return 0;
}

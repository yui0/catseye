//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc 12net_test.c -o 12net_test -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang 12net_test.c -o 12net_test -lm -Ofast -march=native -funroll-loops

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

int main(int argc, char *argv[])
{
	int k = 12;
	int size = 12*12*3;	// 入力層
//	int sample = 18925;
	int sample = 789431;	// 93%(10), 94%(100)

	// https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf
	CatsEye_layer u[] = {	// 12-net 99.9%(1000)
		{   size, CATS_CONV,   CATS_ACT_LEAKY_RELU,  0.01, .ksize=3, .stride=1, /*.padding=1,*/ .ch=16, .ich=3 },
		{      0, CATS_MAXPOOL,                  0,  0.01, .ksize=2, .stride=2 },
		{      0, CATS_LINEAR, CATS_ACT_LEAKY_RELU,  0.01 },	// 16 outputs (Fully-connected layer)
		{     16, CATS_LINEAR,   CATS_ACT_IDENTITY,  0.01 },	// face / non-face
		{      2, CATS_LOSS,         CATS_LOSS_0_1,  0.01 },
	};
	CatsEye cat;
	_CatsEye__construct(&cat, u);
	CatsEye_saveCats(&cat, "12net.cats");

	char *name = argv[1];//"mikarika.jpg";
	int w, h, bpp;
	uint8_t *pixels = stbi_load(name, &w, &h, &bpp, 3);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	real pix[12*12*3];
	for (int y=0; y<h; y+=4) {
		for (int x=0; x<w; x+=4) {
			for (int sy=0; sy<12; sy++) {
				for (int sx=0; sx<12; sx++) {
					pix[12*sy+sx]         = pixels[(w*(y+sy)+x+sx)*3  ] /255.0;
					pix[12*sy+sx+12*12]   = pixels[(w*(y+sy)+x+sx)*3+1] /255.0;
					pix[12*sy+sx+12*12*2] = pixels[(w*(y+sy)+x+sx)*3+2] /255.0;
				}
			}
			int p = _CatsEye_predict(&cat, pix);
			if (p) {
				for (int sy=0; sy<12; sy++) {
					for (int sx=0; sx<12; sx++) {
						pixels[(w*(y+sy)+x+sx)*3] = 0;
					}
				}
			}
		}
	}
	stbi_write_jpg("mikarika_r.jpg", w, h, bpp, pixels, 0);

	CatsEye__destruct(&cat);

	return 0;
}

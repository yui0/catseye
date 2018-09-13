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
	int size = 12*12*3;	// 入力層

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
	CatsEye_loadCats(&cat, "12net.cats");

	char *name = argv[1];//"mikarika.jpg";
	int w, h, bpp;
	uint8_t *pixels = stbi_load(name, &w, &h, &bpp, 3);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	real pix[12*12*3];
	for (int y=0; y<h-12-3; y+=4) {
		for (int x=0; x<w-12-3; x+=4) {
			for (int sy=0; sy<12; sy++) {
				for (int sx=0; sx<12; sx++) {
					pix[12*sy+sx]         = pixels[(w*(y+sy)+x+sx)*3  ] /255.0;
					pix[12*sy+sx+12*12]   = pixels[(w*(y+sy)+x+sx)*3+1] /255.0;
					pix[12*sy+sx+12*12*2] = pixels[(w*(y+sy)+x+sx)*3+2] /255.0;
				}
			}
			//memset(cat.odata, 0, cat.osize*sizeof(real));
			int p = _CatsEye_predict(&cat, pix);
			if (p) {
				for (int sy=0; sy<12; sy++) {
					for (int sx=0; sx<12; sx++) {
						pixels[(w*(y+sy)+x+sx)*3] = 0;
					}
				}
			}

			int k = 12;
			static unsigned char pic[12*12*3/*size*/*100];
			memset(pic, 0, size*100);
			CatsEye_visualize(pix+12*12, size/3, k, &pic[100* k*10 +100], k*10);
			CatsEye_visualize(pix+12*12*2, size/3, k, &pic[80* k*10 +100], k*10);
			for (int n=0; n<cat.layers; n++) {
				CatsEye_layer *l = &cat.layer[n];
				if (l->type == CATS_LINEAR) continue;

				int mch = l->ch > 10 ? 10 : l->ch;
				for (int ch=0; ch<mch; ch++) {
					CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pic[n*(k+2) +ch*(k+2)*k*10], k*10);
				}
			}
			static int num = 0;
			char buff[256];
			sprintf(buff, "/tmp/%010d_%d.jpg", num++, p);
			stbi_write_jpg(buff, k*10, k*10, 1, pic, 0);
		}
	}
	stbi_write_jpg("mikarika_r.jpg", w, h, bpp, pixels, 0);
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}

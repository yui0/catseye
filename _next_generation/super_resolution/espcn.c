//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

#define CATS_USE_FLOAT
//#define CATS_OPENCL
//#define CATS_OPENGL
#include "catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NAME	"espcn"
#define ETA	1e-3	// ADAM,AdaGrad (batch 1,3)

int main(int argc,char *argv[])
{
	const int k = 96;
	const int k2 = k/2;
	const int size = k*k*3;
	const int sample = 946-1;

	// https://nykergoto.hatenablog.jp/entry/2019/05/28/%E7%94%BB%E5%83%8F%E3%81%AE%E8%B6%85%E8%A7%A3%E5%83%8F%E5%BA%A6%E5%8C%96%3A_ESPCN_%E3%81%AE_pytorch_%E5%AE%9F%E8%A3%85_/_%E5%AD%A6%E7%BF%92
	// https://qiita.com/nekono_nekomori/items/08ec250ceb09a0004768
	CatsEye_layer u[] = {
		{size/4, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=64, .ich=3 },
//		{size/4, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=64, .ich=3 },
//		{  size, CATS_CONV, ETA, .ksize=5, .stride=1, .padding=2, .ch=64, .ich=3 },
//		{     0, CATS_ACT_TANH },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RELU },
//		{     0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

/*		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=64 },
//		{     0, CATS_ACT_TANH },
		{     0, CATS_ACT_RELU },*/

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=32 },
//		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=32 },
//		{     0, CATS_ACT_TANH },
//		{     0, CATS_BATCHNORMAL },
		{     0, CATS_ACT_RELU },
//		{     0, CATS_ACT_LEAKY_RELU, .alpha=0.2 },

//		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=3, .name="Output" },
//		{  size, CATS_LOSS_IDENTITY_MSE },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=4*3 },
//		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=4*3 }, // Good!!
		{     0, CATS_PIXELSHUFFLER, .r=2, .ch=3, .name="Output" },

		{  size, CATS_LOSS_IDENTITY_MSE },
	};
	CatsEye cat = { .batch=1 };
	CatsEye__construct(&cat, u);
	int output = CatsEye_getLayer(&cat, "Output");

	if (CatsEye_loadCats(&cat, NAME".cats")) {
		return -1;
	}

	uint8_t *pixels;
	int width, height, bpp;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 3);
	if (!pixels) return 0;
	printf("%dx%d\n", width, height);

	uint8_t *pix = calloc(3, width*height*3 *2*2);
	int c = 0;
	int r = 0;
	for (int sy=0; sy<height; sy+=k2-2) {
		for (int sx=0; sx<width; sx+=k2-2) {
			//printf("%d,%d\n",sx,sy);
			real p[k2*k2*3];
			for (int i=0; i<k2; i++) {
				for (int j=0; j<k2; j++) {
//					if (sx+j>width || sy+i>height) continue;
					int n = (sy+i)*width+sx +j;
					if (n>width*height) continue;
					p[i*k2+j] = pixels[n*3  ] / 255.0;
					p[i*k2+j+k2*k2] = pixels[n*3+1] / 255.0;
					p[i*k2+j+k2*k2*2] = pixels[n*3+2] / 255.0;
				}
			}
			CatsEye_forward(&cat, p);

			for (int y=0; y<k; y++) {
				for (int x=0; x<k; x++) {
					if (sx+x>width*2 || sy+y>height*2) continue;
					int r = (int)(cat.layer[output].z[y*k+x] *255);
					int g = (int)(cat.layer[output].z[(y*k+x)+k*k] *255);
					int b = (int)(cat.layer[output].z[(y*k+x)+k*k*2] *255);
					pix[((sy*2+y)*width*2+sx*2 +x)*3  ] = r>255 ? 255 : r<0 ? 0 : r;
					pix[((sy*2+y)*width*2+sx*2 +x)*3+1] = g>255 ? 255 : g<0 ? 0 : g;
					pix[((sy*2+y)*width*2+sx*2 +x)*3+2] = b>255 ? 255 : b<0 ? 0 : b;
				}
			}
		}
	}
	stbi_write_png(NAME".png", width*2, height*2, 3, pix, 0);
	free(pix);

	CatsEye__destruct(&cat);

	return 0;
}

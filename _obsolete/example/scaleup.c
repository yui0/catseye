//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

// gcc scaleup.c -o scaleup -lm -Ofast -fopenmp -lgomp
// clang scaleup.c -o scaleup -lm -Ofast
#include "../catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define BPP	3

void bilinear(unsigned char *o, unsigned char *in, int inWidth, int inHeight, int inChannel, double mag)
{
	int i, j;
	double inX, inY;

	int height = (int)((double)inHeight * mag);
	int width = (int)((double)inWidth * mag);

	int inStep = inWidth * inChannel;
	int outStep = width * inChannel;

	// ループ中にif文を書くと遅いためループ前にビット数の条件分岐
	switch (inChannel) {
	case 1:
		// 8bit画像
		for (i=0; i<height; ++i) {
			inY = (double)i / mag;
			// 入力画像の高さを超えないようにする
			if ((int)inY >= inHeight) --inY;

			for (j=0; j<width; ++j) {
				// 入力画像の幅を超えないようにする
				inX = (double)j / mag;
				if ((int)inX >= inWidth) --inX;
				o[i*outStep + j] = (unsigned char)(((int)inX + 1 - inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + (int)inX]
					+ (inX - (int)inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + (int)inX + 1]
					+ ((int)inX + 1 - inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + (int)inX]
					+ (inX - (int)inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + (int)inX + 1]);
			}
		}
		break;
	case 3:
		// 24bit画像
		for (i=0; i<height; i++) {
			inY = (double)i / mag;
			// 入力画像の高さを超えないようにする
			if ((int)inY >= inHeight) --inY;

			for (j=0; j<width; j++) {
				// 入力画像の幅を超えないようにする
				inX = (double)j / mag;
				if ((int)inX >= inWidth) --inX;
				o[i*outStep + j*3] = (unsigned char)(((int)inX + 1 - inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + (int)inX * 3]
						+ (inX - (int)inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + ((int)inX + 1) * 3]
						+ ((int)inX + 1 - inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + (int)inX * 3]
						+ (inX - (int)inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + ((int)inX + 1) * 3]);
				o[i*outStep + j*3+1] = (unsigned char)(((int)inX + 1 - inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + (int)inX * 3 + 1]
						+ (inX - (int)inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + ((int)inX + 1) * 3 + 1]
						+ ((int)inX + 1 - inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + (int)inX * 3 + 1]
						+ (inX - (int)inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + ((int)inX + 1) * 3 + 1]);
				o[i*outStep + j*3+2] = (unsigned char)(((int)inX + 1 - inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + (int)inX * 3 + 2]
						+ (inX - (int)inX) * ((int)inY + 1 - inY) * (double)in[(int)inY * inStep + ((int)inX + 1) * 3 + 2]
						+ ((int)inX + 1 - inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + (int)inX * 3 + 2]
						+ (inX - (int)inX) * (inY - (int)inY) * (double)in[((int)inY + 1) * inStep + ((int)inX + 1) * 3 + 2]);
			}
		}
		break;
	case 4:
		// 32bit画像
		break;
	default:
		break;
	}
}

void diff(/*unsigned */char *out, unsigned char *in, unsigned char *in2, int width, int height)
{
	char *p = out;
	char *s1 = (char*)in;
	char *s2 = (char*)in2;

	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			/**p++ = (int)(*s2++) - (int)(*s1++);
			*p++ = (int)(*s2++) - (int)(*s1++);
			*p++ = (int)(*s2++) - (int)(*s1++);*/

			*p++ = 128 + (int)(*s2++) - (int)(*s1++);
			*p++ = 128 + (int)(*s2++) - (int)(*s1++);
			*p++ = 128 + (int)(*s2++) - (int)(*s1++);

			/*int diff = (*s2++) - (*s1++);
			diff += (*s2++) - (*s1++);
			diff += (*s2++) - (*s1++);
			*p++ = diff;
			*p++ = diff;
			*p++ = diff;*/

			/**p++ = sqrt((*s2)*(*s2) - (*s1)*(*s1));
			s1++; s2++;
			*p++ = sqrt((*s2)*(*s2) - (*s1)*(*s1));
			s1++; s2++;
			*p++ = sqrt((*s2)*(*s2) - (*s1)*(*s1));
			s1++; s2++;*/
		}
	}
}

// http://koujinz.cocolog-nifty.com/blog/2009/05/post-45f8.html
void sres(unsigned char *p, unsigned char *s1, unsigned char *s2, int width, int height)
{
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			int a;
/*			a = (int)(*s1++) + (((int)*s2++) -128);
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);
			a = (int)(*s1++) + (((int)*s2++) -128);
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);
			a = (int)(*s1++) + (((int)*s2++) -128);
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);*/

			a = (int)(*s1++) + (((int)*s2++) -128) *1.58;
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);
			a = (int)(*s1++) + (((int)*s2++) -128) *1.58;
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);
			a = (int)(*s1++) + (((int)*s2++) -128) *1.58;
			*p++ = a<0 ? 0 : (a>255 ? 255 : a);
/*			*p++ = ((int)(*s1++) + ((((int)*s2++) -128) *1.58)) / 1.58;
			*p++ = ((int)(*s1++) + ((((int)*s2++) -128) *1.58)) / 1.58;
			*p++ = ((int)(*s1++) + ((((int)*s2++) -128) *1.58)) / 1.58;*/

/*			*p++ = (int)(*s1++) + (((int)*s2++) -128);
			*p++ = (int)(*s1++) + (((int)*s2++) -128);
			*p++ = (int)(*s1++) + (((int)*s2++) -128);*/
//			*p++ = (*s1++) + (*s2++)*1;
		}
	}
}

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stb_image_resize.h"
int main(int argc,char *argv[])
{
	// 読み込み
	unsigned char *pixels;
	int width, height, bpp;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 3);
	if (!pixels) {
		printf("Can't load %s.\n", argv[1]);
		return 0;
	}

	// 整数倍だと差分が出ない
	// x1.6
	unsigned char *image = calloc(width*1.6 * height*1.6 * BPP, sizeof(char));
//	bilinear(image, pixels, width, height, 3, 1.6);
	bilinear(image, pixels, width, height-1, 3, 1.6);
//	stbir_resize_uint8(pixels, width, height, 0, image, width*1.6, height*1.6, 0, 3);
	stbi_write_png("1.png", width*1.6, height*1.6, BPP, image, 0);

	// x0.625
	unsigned char *image2 = calloc(width * height * BPP, sizeof(char));
	bilinear(image2, image, width*1.6, height*1.6, 3, 0.625);
//	stbir_resize_uint8(image, width*1.6, height*1.6, 0, image2, width, height, 0, 3);
	stbi_write_png("2.png", width, height, BPP, image2, 0);
	printf("+\n");

	// 差分
	unsigned char *image3 = calloc(width * height * BPP, sizeof(char));
	diff((char*)image3, image2, pixels, width, height);
	stbi_write_png("3.png", width, height, BPP, image3, 0);

	// 差分拡大
	unsigned char *image4 = calloc(width*1.6 * height*1.6 * BPP, sizeof(char));
	bilinear(image4, image3, width, height, 3, 1.6);
//	stbir_resize_uint8(image3, width, height, 0, image4, width*1.6, height*1.6, 0, 3);
	stbi_write_png("4.png", width*1.6, height*1.6, BPP, image4, 0);

	// 超解像度
	unsigned char *image5 = calloc(width*1.6 * height*1.6 * BPP, sizeof(char));
	sres(image5, image, image4, width*1.6, height*1.6);
	stbi_write_png("5.png", width*1.6, height*1.6, BPP, image5, 0);

	free(image5);
	free(image4);
	free(image3);
	free(image2);
	free(image);

	free(pixels);

	return 0;
}

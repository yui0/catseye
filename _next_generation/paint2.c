//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2021 Yuichiro Nakada
//---------------------------------------------------------

// gcc paint.c -o paint -lm -Ofast -fopenmp -lgomp -march=native -funroll-loops `pkg-config --libs --cflags OpenCL`
// clang paint.c -o paint -lm -Ofast -march=native -funroll-loops `pkg-config --libs --cflags OpenCL`
#define CATS_USE_FLOAT
#include "catseye.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"
#include "imgp.h"

//#define ETA	0.01 // epoch 70
//#define ETA	0.005
#define ETA	0.001

int main(int argc,char *argv[])
{
	if (argc<2) {
		printf("%s *.png\n", argv[0]);
		return 0;
	}

	// データの読み込み
	uint8_t *pixels;
	int width, height, bpp;
	pixels = stbi_load(argv[1], &width, &height, &bpp, 3);
	if (!pixels) return 0;
	int sample = width * height;

	real x[sample*2];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			x[(i*width+j)*2  ] = i / (real)width;
			x[(i*width+j)*2+1] = j / (real)height;
		}
	}
	real t[sample*3];
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			t[(i*width+j)*3  ] = pixels[(i*width+j)*3  ] / 255.0 *2-1;
			t[(i*width+j)*3+1] = pixels[(i*width+j)*3+1] / 255.0 *2-1;
			t[(i*width+j)*3+2] = pixels[(i*width+j)*3+2] / 255.0 *2-1;
		}
	}

	// resize
	real scale = 2.0;
	int sx = (int)(width * scale);
	int sy = (int)(height * scale);
	uint8_t *pix = malloc(sx*sy*bpp);
	stbir_resize_uint8_srgb(pixels, width, height, 0, pix, sx, sy, 0, bpp, -1, 0);
	stbi_image_free(pixels);

	uint8_t *gray = malloc(sx*sy*4);
	uint8_t *dilated = gray+sx*sy;
	uint8_t *diff = gray+sx*sy*2;
	uint8_t *contour = gray+sx*sy*3;
	imgp_gray(pix, sx, sy, sx, gray, sx);
	imgp_dilate(gray, sx, sy, 1, dilated);
//	stbi_write_jpg("dilated.jpg", w, h, 1, dilated, 0);
	imgp_absdiff(gray, dilated, sx, sy, contour);
//	stbi_write_jpg("diff.jpg", w, h, 1, diff, 0);
//	imgp_reverse(diff, sx, sy, contour);
//	stbi_write_jpg("contour.jpg", w, h, 1, contour, 0);
//	free(gray);

	CatsEye_layer u[] = {
		{   2, CATS_LINEAR, ETA }, // x,y
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
		{  20, CATS_ACT_TANH },
		{  20, CATS_LINEAR, ETA },
//		{   3, CATS_ACT_SIGMOID },
		{   3, CATS_LOSS_IDENTITY_MSE }, // RGB
	};
//	CatsEye cat = { .batch=1 };
	CatsEye cat = { .batch=128 };
	CatsEye__construct(&cat, u);

//	real scale = 2.0;
	width = (int)(scale * width);
	height = (int)(scale * height);

	printf("Starting training...\n");
	pixels = malloc(width*height*3);
	for (int n=0; n</*500*/90; n++) {
		// 訓練
		CatsEye_train(&cat, x, t, sample, 10/*epoch*/, sample, 0);

		// 結果の表示
		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
//				CatsEye_forward(&cat, x+2*(i*width+j));
				real xx[2];
				xx[0] = i / (real)width;
				xx[1] = j / (real)height;
				CatsEye_forward(&cat, xx);
				CatsEye_layer *l = &cat.layer[cat.end-1];
				real r = ((l->z[0]+1)/2 * 255.0) +contour[i*width+j];
				real g = ((l->z[1]+1)/2 * 255.0) +contour[i*width+j];
				real b = ((l->z[2]+1)/2 * 255.0) +contour[i*width+j];
				pixels[(i*width+j)*3  ] = r>255 ? 255: r<0 ? 0 : (uint8_t)r;
				pixels[(i*width+j)*3+1] = g>255 ? 255: g<0 ? 0 : (uint8_t)g;
				pixels[(i*width+j)*3+2] = b>255 ? 255: b<0 ? 0 : (uint8_t)b;
			}
		}
		char name[256];
		snprintf(name, 256, "/tmp/paint%04d.png", n);
		stbi_write_png(name, width, height, 3, pixels, 0);
	}
	printf("Training complete\n");
	printf("ffmpeg -r 30 -i paint%%4d.png -pix_fmt yuv420p paint.mp4\n");
	system("ffmpeg -r 30 -i /tmp/paint%4d.png -pix_fmt yuv420p paint.mp4");
//	system("mv /tmp/paint0499.png .");
	system("mv /tmp/paint0089.png .");
	free(gray);
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}
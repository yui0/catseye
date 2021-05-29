//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// gcc dae-yuichan.c -o dae-yuichan -lm -Ofast -fopenmp -lgomp
// clang dae-yuichan.c -o dae-yuichan -lm -Ofast
// http://joisino.hatenablog.com/entry/2015/09/09/224157
#define CATS_AUTOENCODER
#define CATS_DENOISING_AUTOENCODER
#define CATS_SIGMOID_CROSSENTROPY
#include "../catseye.h"
//#define STB_IMAGE_IMPLEMENTATION
//#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

double _rand()
{
	return (rand()/(RAND_MAX+1.0)) * 1.0;
}
double normRand(double m, double s)
{
	double a = 1 - _rand();
	double b = 1 - _rand();
	double c = sqrt(-2 * log(a));
	if (0.5 - _rand() > 0) {
		return c * sin(M_PI * 2 * b) * s + m;
	} else {
		return c * cos(M_PI * 2 * b) * s + m;
	}
}

int main()
{
	int size = 96*96*3;
	int hidden = 128;

	int u[] = {
		0, 0, 1, size, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, hidden, 0, 0, 0, 0,
		CATS_LINEAR, CATS_ACT_SIGMOID, 1, size, 0, 0, 0, CATS_LOSS_MSE,
	};
	int layers = sizeof(u)/sizeof(int)/LPLEN;

	CatsEye cat;
	CatsEye__construct(&cat, 0, 0, layers, u);

	FILE *fp = fopen("dae-yuichan.dat", "r");
	if (fp==NULL) return -1;
//	while (feof(fp)==0) {
		for (int i=0; i<hidden; i++) {
			for (int j=0; j<size; j++) {
//				fscanf(fp, "%lf,", cat.w[1]+(size+1)*i+j);
				fscanf(fp, "%lf,", cat.w[1]+size*i+j);
			}
		}
		for (int i=0; i<hidden; i++) {
//			fscanf(fp, "%lf,", cat.w[1]+(size+1)*i+size);
			fscanf(fp, "%lf,", cat.w[1]+size*hidden+i);
		}
//	}
	fclose(fp);

	unsigned char *pixels = malloc(size*5);
	memset(pixels, 0, size*5);
	srand((unsigned)(time(0)));
	for (int i=0; i<5; i++) {
		for (int n=0; n<hidden; n++) {
//			cat.o[1][n] = (rand()/(RAND_MAX+1.0)) * 1.0;
//			cat.o[1][n] = (rand()/(RAND_MAX+1.0)) * 0.2;
			cat.o[1][n] = normRand(0.0, 0.2);
		}
//		CatsEye_layer_forward[u[LPLEN+ACT]](cat.o[1], cat.w[1], cat.z[1], cat.o[2], &u[LPLEN*(1+1)]);
		CatsEye_propagate(&cat, 1);

		/*double *o = cat.o[2];
		double max = o[0];
		double min = o[0];
		for (int i=1; i<size; i++) {
			if (max < o[i]) max = o[i];
			if (min > o[i]) min = o[i];
		}
		for (int i=0; i<size; i++) {
			o[i] = (o[i] - min) / (max - min);
		}*/

		unsigned char *p = &pixels[(i/5)*size*5 + (i%5)*96*3];
		for (int y=0; y<96; y++) {
			for (int x=0; x<96; x++) {
				p[(y*96*5+x)*3  ] = cat.o[2][(y*96+x)*3  ] * 255.0;
				p[(y*96*5+x)*3+1] = cat.o[2][(y*96+x)*3+1] * 255.0;
				p[(y*96*5+x)*3+2] = cat.o[2][(y*96+x)*3+2] * 255.0;
			}
		}
/*		for (int j=0; j<size; j++) {
			p[(j/(96*3))*96*5+(j%(96*3))] = cat.o[2][j] * 255.0;
		}*/
	}
	stbi_write_png("dae_yuichan.png", 96*5, 96, 3, pixels, 96*3*5);
//	stbi_write_png("dae_yuichan.png", 96, 96, 3, pixels, 96*3);
	free(pixels);

	CatsEye__destruct(&cat);

	return 0;
}

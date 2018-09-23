//---------------------------------------------------------
//	Cat's eye
//
//		©2018 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os -o make_dataset make_dataset.c -lm
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "../stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#include <stdio.h>
#include <string.h>
#include "ls.h"

uint8_t *pix;

void make_dataset(FILE *fp, char *name, int sx, int sy, int label)
{
	uint8_t *pixels;
	int w, h, bpp;
	pixels = stbi_load(name, &w, &h, &bpp, 3);
//	printf("[%s]\n", name);
	assert(pixels);
	printf("#%d %s %dx%d %d\n", label, name, w, h, bpp);
	bpp = 3;

	// resize
	uint8_t *p = pix;
	uint8_t *s = pix +sx*sy*bpp;
	stbir_resize_uint8_srgb(pixels, w, h, 0, s, sx, sy, 0, bpp, -1, 0);
	stbi_image_free(pixels);

	for (int y=0; y<sy; y++) {
		for (int x=0; x<sx; x++) {
			*p           = *s++;
			*(p+sx*sy)   = *s++;
			*(p+sx*sy*2) = *s++;
			p++;
		}
	}

	fwrite(pix, sx*sy*bpp, 1, fp);
}

int main(int argc, char* argv[])
{
	char *p;
	char *dir = ".";
	char *target = "datasets.bin";
	static int32_t l[1000], lmax[1000];

	for(int i=0; i<argc; i++) {
		p = argv[i];
		if (*p=='-') { // option
			p++;
			switch (*p) {
			case 'o':
				target = ++p;
				break;
			case 'l':
				lmax[0] = atoi(++p);
				break;
			}
		} else {
			dir = p;
		}
	}

	FILE *fp = fopen(target, "wb");
	char *name = dir;
	int num;

	pix = malloc(12*12*3 *2);
	LS_LIST *ls = ls_dir(name, 1, &num);
//	fwrite(&num, sizeof(int32_t), 1, fp);
	for (int i=0; i<num; i++) {
//		printf("\n%s\n", ls[i].d_name);
/*		char buff[256];
		strcpy(buff, ls[i].d_name);
		char *p = strrchr(buff, '/');
		*p = 0;
		p = strrchr(buff, '/');
//		printf("%s\n", p+1);
		int16_t label = atoi(p+1);*/
		char *p = strstr(ls[i].d_name, "/label_");
		int16_t label = atoi(p+7);
		if (lmax[label] && lmax[label] <= l[label]) continue;
		l[label]++;
		fwrite(&label, sizeof(int16_t), 1, fp);

		make_dataset(fp, ls[i].d_name, 12, 12, label);
	}
	free(ls);
	free(pix);

	printf("Total %d\n", num);
	for (int i=0; i<1000; i++) if (l[i]) printf("  label#%d %d\n", i, l[i]);
	fclose(fp);
}

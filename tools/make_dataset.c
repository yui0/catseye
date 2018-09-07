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

void make_dataset(FILE *fp, char *name, int sx, int sy)
{
	uint8_t *pixels;
	int w, h, bpp;
	pixels = stbi_load(name, &w, &h, &bpp, 3);
//	printf("[%s]\n", name);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	bpp = 3;

	// resize
	uint8_t *pix = malloc(sx*sy*bpp);
	stbir_resize_uint8_srgb(pixels, w, h, 0, pix, sx, sy, 0, bpp, -1, 0);
	stbi_image_free(pixels);

	fwrite(pix, sx*sy*bpp, 1, fp);
}

int main(int argc, char* argv[])
{
	FILE *fp = fopen("datasets.bin", "wb");
	char *name = argv[1];
	int32_t num;

	LS_LIST *ls = ls_dir(name, 1, &num);
	fwrite(&num, sizeof(int32_t), 1, fp);
	for (int i=0; i<num; i++) {
//		printf("\n%s\n", ls[i].d_name);
		make_dataset(fp, ls[i].d_name, 12, 12);
	}
	for (int i=0; i<num; i++) {
		char buff[256];
		strcpy(buff, ls[i].d_name);
		char *p = strrchr(buff, '/');
		*p = 0;
		p = strrchr(buff, '/');
//		printf("%s\n", p+1);
		int16_t label = atoi(p+1);
		fwrite(&label, sizeof(int16_t), 1, fp);
	}
	free(ls);

	fclose(fp);
}

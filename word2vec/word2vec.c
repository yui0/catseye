//---------------------------------------------------------
//	Cat's eye
//
//		©2023 Yuichiro Nakada
//---------------------------------------------------------

// gcc word2vec.c -o word2vec -lm -fopenmp -lgomp
// https://ie.u-ryukyu.ac.jp/~tnal/2021/dm/static/4-nlp/nlp2.html

//#include <stdio.h>
//#include <stdlib.h>
//#include <stdint.h>
//#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <string.h>

#define ETA	1e-2
#define BATCH	1

//#define CATS_TEST
#define CATS_USE_FLOAT
//#define CATS_OPENGL
#include "catseye.h"

#define DICSIZE		8192
#include "ht.h"
#include "sentencepiece.h"

#define CACHE_FILE	"/tmp/word2vec.bin"
int fexist(const char* path)
{
	FILE* fp = fopen(path, "r");
	if (fp == NULL) return 0;
	fclose(fp);
	return 1;
}

real* read_txt(const char *name, int *c, ht* dic)
{
	int fd;
	struct stat sb;

	fd = open(name, O_RDONLY);
	fstat(fd, &sb);
//	printf("Size: %lu\n", (uint64_t)sb.st_size);
	close(fd);

	if (fexist(CACHE_FILE)) {
		fd = open(CACHE_FILE, O_RDONLY);
		fstat(fd, &sb);
		real *wi = malloc(sb.st_size);
		read(fd, wi, sb.st_size);
		close(fd);
		*c = sb.st_size/sizeof(real);
		return wi;
	}

	int count = 0;
	real *wi = malloc(sizeof(real)*sb.st_size);

	char str[8192];
	FILE *fp = fopen(name, "r");
	if (!fp) return 0;
	while (fgets(str, sizeof(str), fp) != NULL) {
		size_t* a = word2int(str, dic);
		if (!a) { printf("mem err\n"); return 0; }
		for (int i=0; a[i]!=SIZE_MAX; i++) {
//			printf(" %zu\r", a[i]);
			wi[count++] = a[i];
		}
		free(a);
//		printf(" %s\r", str);
		printf("line: %d\r", count);
	}
	printf("\n");
	fclose(fp);
	*c = count;

	fp = fopen(CACHE_FILE, "wb");
	if (!fp) return 0;
	fwrite(wi, sizeof(real)*count, 1, fp);
	fclose(fp);
	return wi;
}

int window_size = 2;
void set_data(CatsEye *this, CatsEye_layer *l, int n, int b)
{
//	printf("set data: %d\n", n);
	// n, n+2
	real *x = l->x +l->inputs*b;
	memset(x, 0, sizeof(real)*DICSIZE *window_size);
	x[(int)this->learning_data[n]] = 1.0;
	x[DICSIZE +(int)this->learning_data[(n+2)]] = 1.0;

	// n+1
	real *a = this->label +this->label_size*b;
	memset(a, 0, sizeof(real)*DICSIZE);
	a[(int)this->learning_data[(n+1)]] = 1.0;
	printf("sample:%d\r", n);
}

int main(int argc, char *argv[])
{
	CatsEye_layer u[] = {
		{ DICSIZE*2, CATS_LINEAR, ETA },
		{      2000, CATS_ACT_TANH },
		{   DICSIZE, CATS_SOFTMAX_CE },
	};
/*	CatsEye_layer u[] = {
		{ DICSIZE*2, CATS_LINEAR, ETA },
		{      2000, CATS_ACT_SIGMOID },
		{   DICSIZE, CATS_SIGMOID_BCE },
	};*/
	CatsEye cat = { .batch=BATCH };
	CatsEye__construct(&cat, u);

	// 訓練データ
	ht* dic = read_dic("sentencepiece.dic");
//	int sample = ht_length(dic);
//	int sample = DICSIZE;
	int sample;
	cat.set_data = set_data;
	real *x = read_txt(argv[1], &sample, dic);
	printf("samples: %d\n\n", sample);

	// 訓練
	printf("Starting training...\n");
//	CatsEye_train(&cat, x, x, sample/100, 1/*epoch*/, 1500/*random batch*/, sample/1000);
	CatsEye_train(&cat, x, x, sample/100, 2/*epoch*/, 1500/*random batch*/, 0);
	printf("Training complete\n");
//	CatsEye_save(&cat, "word2vec.weights");

	// 結果の表示
/*	FILE *fp = fopen("/tmp/sin.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		fprintf(fp, "%d, %lf\n", i, cat.layer[cat.layers-1].x[0]);
	}
	fclose(fp);*/

	CatsEye__destruct(&cat);

	return 0;
}

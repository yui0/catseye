//---------------------------------------------------------
//	Cat's eye
//
//		©2023 Yuichiro Nakada
//---------------------------------------------------------

// gcc -I.. word2vec.c -o word2vec -lm -fopenmp -lgomp  -O0 -D_FORTIFY_SOURCE=2 -g
// ./word2vec /tmp/ldcc.txt

// valgrind --tool=memcheck --leak-check=yes ./word2vec /tmp/ldcc.txt
// -fmudflap -lmudflap

// https://ie.u-ryukyu.ac.jp/~tnal/2021/dm/static/4-nlp/nlp2.html

//#include <stdio.h>
//#include <stdlib.h>
//#include <stdint.h>
//#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <string.h>

#define DICSIZE		8192
//#define VECSIZE		2000
#define VECSIZE		200
#include "ht.h"
#include "sentencepiece.h"

#define CATS_USE_ADAM
//#define ETA	1e-2
#define ETA	1e-3
#define BATCH	1

//#define CATS_TEST
#define CATS_USE_FLOAT
//#define CATS_OPENGL
#include "catseye.h"

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

	if (fexist(CACHE_FILE)) {
		fd = open(CACHE_FILE, O_RDONLY);
		fstat(fd, &sb);
		real *wi = malloc(sb.st_size);
		read(fd, wi, sb.st_size);
		close(fd);
		*c = sb.st_size/sizeof(real);
		return wi;
	}

	fd = open(name, O_RDONLY);
	fstat(fd, &sb);
//	printf("Size: %lu\n", (uint64_t)sb.st_size);
	close(fd);

	int count = 0;
	real *wi = malloc(/*sizeof(real)**/sb.st_size);

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

real cos_similarity(real a[], real b[], int size)
{
	real aa = 0, bb = 0, vector, hypotenuse;

	for (int i = 0; i < size; i++) {
		vector += a[i] * b[i];
		aa += pow(a[i], 2);
		bb += pow(b[i], 2);	
	}

	hypotenuse = sqrt(aa) * sqrt(bb);
//	printf("%f %f \n\n", vector, hypotenuse);

	return vector / hypotenuse;
}

int window_size = 1;
void set_data(CatsEye *this, CatsEye_layer *l, int n, int b)
{
//	printf("set data: %d\n", n);
	// n, n+2
	real *x = l->x +l->inputs*b;
	memset(x, 0, sizeof(real)*DICSIZE *window_size*2);
	x[(int)this->learning_data[n]] = 1.0;
	x[DICSIZE +(int)this->learning_data[(n+2)]] = 1.0;

	// n+1
/*	real *a = this->label +this->label_size*b;
	memset(a, 0, sizeof(real)*DICSIZE);
	a[(int)this->learning_data[(n+1)]] = 1.0;*/
	this->clasify[b] = (int16_t)this->learning_data[(n+1)];
	printf("sample:%d\r", n);
}

void get_word_vec(char *w, ht* dic, CatsEye *cat)
{
	size_t *a = word2int(w, dic);
	printf("[%s] id:%d\n", w, a[0]);
	real x[DICSIZE*window_size*2];
	memset(x, 0, sizeof(real)*DICSIZE *window_size*2);
	x[(int)a[0]] = 1.0;
	CatsEye_forward(cat, x);
	for (int i=0; i<VECSIZE; i++) {
		printf(" %f", cat->layer[1].x[i]);
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	// CBOW
	CatsEye_layer u[] = {
		{ DICSIZE*2, CATS_LINEAR, ETA },
//		{   VECSIZE, CATS_ACT_TANH },
		{   VECSIZE, CATS_ACT_SIGMOID },
		{   VECSIZE, CATS_LINEAR, ETA },// Embedding (hidden) layer
//		{   DICSIZE, CATS_ACT_TANH },
		{   DICSIZE, CATS_ACT_SOFTMAX },
		{   DICSIZE, CATS_SOFTMAX_CE },
	};
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
//	CatsEye_train(&cat, x, x, sample/100, 3/*epoch*/, 1500/*random batch*/, 0);
	CatsEye_train(&cat, x, x, sample, 30/*epoch*/, 150/*random batch*/, 0);
	printf("Training complete\n");
//	CatsEye_saveCats(&cat, "/tmp/word2vec.weights");

	free(x);

	// 結果の表示
/*	FILE *fp = fopen("/tmp/sin.csv", "w");
	if (fp==NULL) return -1;
	for (int i=0; i<sample; i++) {
		CatsEye_forward(&cat, x+i);
		fprintf(fp, "%d, %lf\n", i, cat.layer[cat.layers-1].x[0]);
	}
	fclose(fp);*/

	get_word_vec("男", dic, &cat);
	get_word_vec("女", dic, &cat);

	CatsEye__destruct(&cat);

	return 0;
}

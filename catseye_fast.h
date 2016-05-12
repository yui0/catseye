#ifdef CATS_SSE

#include <xmmintrin.h>
#include <immintrin.h>
void *_malloc(int x)
{
	void *p;
	posix_memalign((void**)&p, 32, x);
	return p;
}
#define malloc(x)	_malloc(x)
#ifdef CATS_USE_FLOAT
/*float dot(float *vec1, float *vec2, int n)
{
	int i;
	__m128 u = {0};
	for (i=0; i<n; i+=4) {
		__m128 w = _mm_load_ps(&vec1[i]);	// load 4 values
		__m128 x = _mm_load_ps(&vec2[i]);
		x = _mm_mul_ps(w, x);
		u = _mm_add_ps(u, x);
	}
	__attribute__((aligned(16))) float t[4] = {0};
	_mm_store_ps(t, u);

	float s = 0;
	for (; i<n; i++) {
		s += (*vec1++) * (*vec2++);
	}
	return t[0] + t[1] + t[2] + t[3] + s;
}
float dotT(float *mat1, float *vec1, int r, int c)
{
	int i;
	__attribute__((aligned(16))) float t[4] = {0};
	__m128 u = {0};
	for (i=0; i<r; i+=4) {
		t[0] = *mat1;
		mat1 += c;
		t[1] = *mat1;
		mat1 += c;
		t[2] = *mat1;
		mat1 += c;
		t[3] = *mat1;
		mat1 += c;
		__m128 w = _mm_load_ps(t);
		__m128 x = _mm_load_ps(vec1);
		vec1 += 4;
		x = _mm_mul_ps(w, x);
		u = _mm_add_ps(u, x);
	}
	_mm_store_ps(t, u);

	float s = 0;
	for (; i<r; i++) {
		s += (*mat1) * (*vec1++);
		mat1 += c;
	}
	return t[0] + t[1] + t[2] + t[3] + s;
}*/
float dot(float *vec1, float *vec2, int n)
{
	int i;
	__m256 u = {0};
	for (i=0; i<n; i+=8) {
		__m256 w = _mm256_load_ps(&vec1[i]);
		__m256 x = _mm256_load_ps(&vec2[i]);
		x = _mm256_mul_ps(w, x);
		u = _mm256_add_ps(u, x);
	}
	__attribute__((aligned(32))) float t[8];
	_mm256_store_ps(t, u);

	float s = 0;
	for (; i<n; i++) {
		s += (*vec1++) * (*vec2++);
	}
	return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] + s;
}
float dotT(float *mat1, float *vec1, int r, int c)
{
	int i;
	__attribute__((aligned(32))) float t[8];
	__m256 u = {0};
	for (i=0; i<r; i+=8) {
//	for (int i=r/8; i>0; i--) {
		t[0] = *mat1;	mat1 += c;
		t[1] = *mat1;	mat1 += c;
		t[2] = *mat1;	mat1 += c;
		t[3] = *mat1;	mat1 += c;
		t[4] = *mat1;	mat1 += c;
		t[5] = *mat1;	mat1 += c;
		t[6] = *mat1;	mat1 += c;
		t[7] = *mat1;	mat1 += c;
		__m256 w = _mm256_load_ps(t);
		__m256 x = _mm256_load_ps(vec1);
		vec1 += 8;
		x = _mm256_mul_ps(w, x);
		u = _mm256_add_ps(u, x);
	}
	_mm256_store_ps(t, u);

	float s = 0;
	for (; i<r; i++) {
		s += (*mat1) * (*vec1++);
		mat1 += c;
	}
	return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] + s;
}
void muladd(float *vec1, float *vec2, float a, int n)
{
	int i;
	__m256 alpha = _mm256_set1_ps(a);
	__m256 beta = _mm256_set1_ps(1e-8);
	for (i=0; i<n; i+=8) {
		__m256 d = _mm256_load_ps(&vec1[i]);
		__m256 w = _mm256_load_ps(&vec2[i]);
		d = _mm256_mul_ps(alpha, d);
		w = _mm256_add_ps(w, d);
		d = _mm256_mul_ps(beta, w);
		w = _mm256_add_ps(w, d);
//		_mm256_store_ps(&vec2[i], w);
		_mm256_storeu_ps(&vec2[i], w);
	}

	for (; i<n; i++) {
		vec2[i] += a * vec1[i] + vec2[i]*1e-8;
	}
}
#else
double dot(double *vec1, double *vec2, int n)
{
	__m128d u = {0};
	for (int i=0; i<n; i+=2) {
		__m128d w = _mm_load_pd(vec1);	// load 2 values
		__m128d x = _mm_load_pd(vec2);
		vec1 += 2;
		vec2 += 2;
		x = _mm_mul_pd(w, x);
		u = _mm_add_pd(u, x);
	}
	__attribute__((aligned(16))) double t[2] = {0};
	_mm_store_pd(t, u);
	return t[0] + t[1];
}
double dotT(double *mat1, double *vec1, int r, int c)
{
	__attribute__((aligned(16))) double t[2] = {0};
	__m128d u = {0};
	for (int i=0; i<r; i+=2) {
		t[0] = *mat1;
		mat1 += c;
		t[1] = *mat1;
		mat1 += c;
		__m128d w = _mm_load_pd(t);
		__m128d x = _mm_load_pd(vec1);
		vec1 += 2;
		x = _mm_mul_pd(w, x);
		u = _mm_add_pd(u, x);
	}
	_mm_store_pd(t, u);
	return t[0] + t[1];
}
#endif

#endif

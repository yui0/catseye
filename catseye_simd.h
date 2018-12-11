#ifdef CATS_SSE

#if defined(_MSC_VER)
#define _aligned(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define _aligned(x) __attribute__ ((aligned(x)))
#endif
#endif

#define _aligned_type(t,x) typedef t _aligned(x)

_aligned_type(double, 32) _double;

#include <xmmintrin.h>
#include <immintrin.h>
void *_malloc(size_t x)
{
	void *p;
	posix_memalign((void**)&p, 32, x);
	return p;
}
void *_calloc(size_t n, size_t size)
{
	void *p = _malloc(n*size);
	if (p) memset(p, 0, n*size);
	return p;
}
#define malloc(x)	_malloc(x)
#define calloc(n, x)	_calloc(n, x)

#ifdef CATS_USE_FLOAT
#ifndef CATS_AVX
float dot(float *vec1, float *vec2, int n)
{
	int i, m = n/4;
	__m128 u = {0};
	for (i=m; i>0; i--) {
		__m128 w = _mm_loadu_ps(vec1);	// load 4 values
		__m128 x = _mm_loadu_ps(vec2);
		x = _mm_mul_ps(w, x);
		u = _mm_add_ps(u, x);
		vec1 += 4;
		vec2 += 4;
	}
	__attribute__((aligned(16))) float t[4] = {0};
	_mm_store_ps(t, u);

	float s = 0;
	for (i=m*4; i<n; i++) {
		s += (*vec1++) * (*vec2++);
	}
	return t[0] + t[1] + t[2] + t[3] + s;
}
float dotT(float *mat1, float *vec1, int r, int c)
{
	int i, m = r/4;
	__attribute__((aligned(16))) float t[4] = {0};
	__m128 u = {0};
	for (i=m; i>0; i--) {
		t[0] = *mat1;
		mat1 += c;
		t[1] = *mat1;
		mat1 += c;
		t[2] = *mat1;
		mat1 += c;
		t[3] = *mat1;
		mat1 += c;
		__m128 w = _mm_loadu_ps(t);
		__m128 x = _mm_loadu_ps(vec1);
		vec1 += 4;
		x = _mm_mul_ps(w, x);
		u = _mm_add_ps(u, x);
	}
	_mm_store_ps(t, u);

	float s = 0;
	for (i=m*4; i<r; i++) {
		s += (*mat1) * (*vec1++);
		mat1 += c;
	}
	return t[0] + t[1] + t[2] + t[3] + s;
}
void muladd(float *vec1, float *vec2, float a, int n)
{
	int i, m = n/4;
	__m128 alpha = _mm_set1_ps(a);
//	__m128 beta = _mm_set1_ps(1e-8);
	for (i=m; i>0; i--) {
		__m128 w = _mm_loadu_ps(vec1);
		__m128 d = _mm_loadu_ps(vec2);
		d = _mm_mul_ps(alpha, d);	// d *= a;
		w = _mm_add_ps(w, d);		// w += d;
//		d = _mm_mul_ps(beta, w);	// d = w*1e-8;
//		w = _mm_add_ps(w, d);		// w += d;
		_mm_storeu_ps(vec1, w);
		vec1 += 4;
		vec2 += 4;
	}

	for (i=m*4; i<n; i++) {
//		vec1[i] += a * vec2[i] + vec1[i] * 1e-8;
		*vec1++ += a * (*vec2++);
	}
}
#else // !CATS_AVX
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
	int i, m = n/8;
	__m256 alpha = _mm256_set1_ps(a);
//	__m256 beta = _mm256_set1_ps(1e-8);
	for (i=m; i>0; i--) {
		__m256 w = _mm256_loadu_ps(vec1);
		__m256 d = _mm256_loadu_ps(vec2);
		d = _mm256_mul_ps(alpha, d);	// d *= a;
		w = _mm256_add_ps(w, d);	// w += d;
//		d = _mm256_mul_ps(beta, w);	// d = w*1e-8;
//		w = _mm256_add_ps(w, d);	// w += d;
//		_mm256_store_ps(&vec2[i], w);
		_mm256_storeu_ps(vec1, w);
		vec1 += 8;
		vec2 += 8;
	}

	for (i=m*4; i<n; i++) {
//		vec1[i] += a * vec2[i] + vec1[i] * 1e-8;
		*vec1++ += a * (*vec2++);
	}
}
#endif // !CATS_AVX
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
#endif // CATS_USE_FLOAT

#endif // CATS_SSE

void gemm_cpu(
	char		major,
	char		transa,
	char		transb,
	const int	M,
	const int	N,
	const int	K,
	const real	alpha,
	const real	*A,
	const int	lda,
	const real	*B,
	const int	ldb,
	const real	beta,
	real		*C,
	const int	ldc)
{
	if (major == 'R') {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[n + k * ldb];
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		} else if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[m + k * lda] * B[n + k * ldb];
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		} else if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[k + n * ldb];
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		} else if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[m + k * lda] * B[k + n * ldb];
					}
					C[n + m * ldc] = alpha * sum + beta * C[n + m * ldc];
				}
			}
		}
	} else {
		if (transa == 'N' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[m + k * lda] * B[k + n * ldb];
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		} else if (transa == 'T' && transb == 'N') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[k + n * ldb];
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		} else if (transa == 'N' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[m + k * lda] * B[n + k * ldb];
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		} else if (transa == 'T' && transb == 'T') {
			for (int m=0; m<M; m++) {
				for (int n=0; n<N; n++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[n + k * ldb];
					}
					C[m + n * ldc] = alpha * sum + beta * C[m + n * ldc];
				}
			}
		}
	}
}

real _dot_8(const real *x, const real *y, int n)
{
	int i, n8 = n>>3<<3;
	real s, t[8];
	t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = t[6] = t[7] = 0.0;
	for (i=0; i<n8; i+=8) {
		t[0] += x[i+0] * y[i+0];
		t[1] += x[i+1] * y[i+1];
		t[2] += x[i+2] * y[i+2];
		t[3] += x[i+3] * y[i+3];
		t[4] += x[i+4] * y[i+4];
		t[5] += x[i+5] * y[i+5];
		t[6] += x[i+6] * y[i+6];
		t[7] += x[i+7] * y[i+7];
	}
	for (s=0.0; i<n; i++) s += x[i] * y[i];
	return s + t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}
float dot_avx_2(const float *vec1, const float *vec2, int n)
{
	int i, n16 = n>>4<<4;
	__m256 u1 = {0};
	__m256 u2 = {0};
	for (i=0; i<n16; i+=16) {
		__m256 w1 = _mm256_load_ps(&vec1[i]);
		__m256 w2 = _mm256_load_ps(&vec1[i+8]);
		__m256 x1 = _mm256_load_ps(&vec2[i]);
		__m256 x2 = _mm256_load_ps(&vec2[i+8]);

		x1 = _mm256_mul_ps(w1, x1);
		x2 = _mm256_mul_ps(w2, x2);
		u1 = _mm256_add_ps(u1, x1);
		u2 = _mm256_add_ps(u2, x2);
	}
	u1 = _mm256_add_ps(u1, u2);

	__attribute__((aligned(32))) static float t[8];// = {0};
	_mm256_store_ps(t, u1);

	for (; i<n; i++) t[0] += vec1[i] * vec2[i];
	return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}
float dot8_avx256(const float *vec1, const float *vec2, int n)
{
}
//#define BLOCK_SIZE 50
void gemm_(
	char		major,
	char		transa,
	char		transb,
	const int	M,
	const int	N,
	const int	K,
	const real	alpha,
	const real	*A,
	const int	lda,
	const real	*B,
	const int	ldb,
	const real	beta,
	real		*C,
	const int	ldc)
{
	memset(C, 0, ldc*M*sizeof(real));
	if (transa == 'N' && transb == 'T') {
		// RNT
		/*for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
				C[n+m*ldc] += _dot_8(&A[m*lda], &B[n*ldb], K);
//				C[n+m*ldc] += dot_avx_2(&A[m*lda], &B[n*ldb], K);
//				C[n+m*ldc] += dot(&A[m*lda], &B[n*ldb], K);
			}
		}*/
		int k, n8 = K>>3<<3;
		__m256 va, vb, vtemp;
		__m128 vlow, vhigh, vresult;
		for (int i=0; i<M; i++) {
			for (int j=0; j<N; j++) {
				register real s = C[i*ldc+j];
				for (k=0; k<n8; k+=8) {
					// load
					va = _mm256_loadu_ps(A+(i*lda)+k); // matrix_a[i][k]
					vb = _mm256_loadu_ps(B+(j*ldb)+k); // matrix_b[j][k]

					// multiply
					vtemp = _mm256_mul_ps(va, vb);

					// add
					// extract higher four floats
					vhigh = _mm256_extractf128_ps(vtemp, 1); // high 128
					// add higher four floats to lower floats
					vresult = _mm_add_ps(_mm256_castps256_ps128(vtemp), vhigh);
					// horizontal add of that result
					vresult = _mm_hadd_ps(vresult, vresult);
					// another horizontal add of that result
					vresult = _mm_hadd_ps(vresult, vresult);

					// store
					s += _mm_cvtss_f32(vresult);
				}
				for (; k<K; k++) s += A[k+i*lda] * B[k+j*ldb];
				C[i*ldc+j] = s;
			}
		}
	} else if (transa == 'N' && transb == 'N') {
		// RNN
		/*for (int n=0; n<N; n++) {
			for (int i=0; i<K; i++) {
				C[n+m*ldc] = alpha * sum + beta * C[n+m*ldc];
			}
			for (int m=0; m<M; m++) {
					register real sum = 0.0;
					for (int k=0; k<K; k++) {
						sum += A[k + m * lda] * B[n + k * ldb];
					}
					C[n+m*ldc] = alpha * sum + beta * C[n+m*ldc];
			}
		}*/
	}
#if 0
//	int i, j, k;
	memset(C, 0, ldc*M*sizeof(real));
	if (transa == 'N' && transb == 'T') {
		/*for (i=0; i<M; i++) {
			for (k=0; k<K; k++) {
				for (j=0; j<N; j++) {
					C[i*ldc+j] += A[i*lda+k] * B[k*ldb+j];
				}
			}
		}*/
		for (int m=0; m<M; m++) {
			for (int n=0; n<N; n++) {
				for (int k=0; k<K; k++) {
					C[n+m*ldc] += A[k+m*lda] * B[k+n*ldb];
				}
			}
		}
	}
#endif
	/*int i, j, k, ii, jj, kk;
	for (i=0; i<N; i+=BLOCK_SIZE) {
		for (j=0; j<N; j+=BLOCK_SIZE) {
			for (k=0; k<N; k+=BLOCK_SIZE) {
				for (ii=i; ii<(i+BLOCK_SIZE); ii+=2) {
					for (jj=j; jj<(j+BLOCK_SIZE); jj++) {
						register real s0 = 0.0;
						register real s1 = 0.0;
						for (kk=k; kk<(k+BLOCK_SIZE); kk+=5) {
							s0 += a[ii][kk] * b[kk][jj];
							s0 += a[ii][kk+1] * b[kk+1][jj];
							s0 += a[ii][kk+2] * b[kk+2][jj];
							s0 += a[ii][kk+3] * b[kk+3][jj];
							s0 += a[ii][kk+4] * b[kk+4][jj];

							s1 += a[ii+1][kk] * b[kk][jj];
							s1 += a[ii+1][kk+1] * b[kk+1][jj];
							s1 += a[ii+1][kk+2] * b[kk+2][jj];
							s1 += a[ii+1][kk+3] * b[kk+3][jj];
							s1 += a[ii+1][kk+4] * b[kk+4][jj];
						}
						c[ii][jj] += s0;
						c[ii+1][jj] += s1;
					}
				}
			}
		}
	}*/
}

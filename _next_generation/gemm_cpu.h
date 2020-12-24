/* public domain Simple, Minimalistic, Fast GEMM library
 *	©2020 Yuichiro Nakada
 *
 * Basic usage:
 *	gemm_cpu('N', 'N', M, N, K, A, B, C);
 * */

#ifndef real
#define real		float
#endif

inline void gemm_rnn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	#pragma omp parallel for
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = K;
	const int ldb = N;
	const int ldc = N;*/
	for (int m=0; m<M; ++m) { // fast
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m*K+k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}

inline void gemm_rnt(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	#pragma omp parallel for
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = K;
	const int ldb = K;
	const int ldc = N;*/
	for (int m=0; m<M; ++m) {
		for (int n=0; n<N; ++n) {
			register real sum = 0;
			for (int k=0; k<K; ++k) {
				sum += A[m*K+k] * B[k+K*n];
			}
			C[m*N+n] += alpha * sum;
		}
	}
}

inline void gemm_rtn(int M, int N, int K, real alpha, real *A, real *B, real beta, real *C)
{
	#pragma omp parallel for
	if (beta==0.0) {
		memset(C, 0, M*N*sizeof(real));
	} else if (beta!=1.0) {
		for (int i=0; i<M*N; i++) C[i] *= beta;
	}
/*	const int lda = M;
	const int ldb = N;
	const int ldc = N;*/
	for (int m=0; m<M; ++m) {
		for (int k=0; k<K; ++k) {
			register real A_PART = alpha * A[m+M*k];
			for (int n=0; n<N; ++n) {
				C[m*N+n] += A_PART * B[k*N+n];
			}
		}
	}
}


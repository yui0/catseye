#define MC  384
#define KC  384
#define NC  4096

#define MR  4
#define NR  4

//  Local buffers for storing panels from A, B and C
static real GEMM(_A)[MC*KC] /*__attribute__ ((aligned(32)))*/;
static real GEMM(_B)[KC*NC] /*__attribute__ ((aligned(32)))*/;
static real GEMM(_C)[MR*NR] /*__attribute__ ((aligned(32)))*/;

//  Packing complete panels from A (i.e. without padding)
static void GEMM(_pack_MRxk)(int k, const real *A, int incRowA, int incColA, real *buffer)
{
	int i, j;

	for (j=0; j<k; ++j) {
		for (i=0; i<MR; ++i) {
			buffer[i] = A[i*incRowA];
		}
		buffer += MR;
		A      += incColA;
	}
}

//  Packing panels from A with padding if required
static void GEMM(_pack_A)(int mc, int kc, const real *A, int incRowA, int incColA, real *buffer)
{
	int mp  = mc / MR;
	int _mr = mc % MR;

	int i, j;

	for (i=0; i<mp; ++i) {
		GEMM(_pack_MRxk)(kc, A, incRowA, incColA, buffer);
		buffer += kc*MR;
		A      += MR*incRowA;
	}
	if (_mr>0) {
		for (j=0; j<kc; ++j) {
			for (i=0; i<_mr; ++i) {
				buffer[i] = A[i*incRowA];
			}
			for (i=_mr; i<MR; ++i) {
				buffer[i] = 0.0;
			}
			buffer += MR;
			A      += incColA;
		}
	}
}

//  Packing complete panels from B (i.e. without padding)
static void GEMM(_pack_kxNR)(int k, const real *B, int incRowB, int incColB, real *buffer)
{
	int i, j;

	for (i=0; i<k; ++i) {
		for (j=0; j<NR; ++j) {
			buffer[j] = B[j*incColB];
		}
		buffer += NR;
		B      += incRowB;
	}
}

//  Packing panels from B with padding if required
static void GEMM(_pack_B)(int kc, int nc, const real *B, int incRowB, int incColB, real *buffer)
{
	int np  = nc / NR;
	int _nr = nc % NR;

	int i, j;

	for (j=0; j<np; ++j) {
		GEMM(_pack_kxNR)(kc, B, incRowB, incColB, buffer);
		buffer += kc*NR;
		B      += NR*incColB;
	}
	if (_nr>0) {
		for (i=0; i<kc; ++i) {
			for (j=0; j<_nr; ++j) {
				buffer[j] = B[j*incColB];
			}
			for (j=_nr; j<NR; ++j) {
				buffer[j] = 0.0;
			}
			buffer += NR;
			B      += incRowB;
		}
	}
}

//  Micro kernel for multiplying panels from A and B.
static void GEMM(_micro_kernel)(
	int kc, real alpha, const real *A, const real *B,
	real beta, real *C, int incRowC, int incColC)
{
	real AB[MR*NR];

	int i, j, l;

	//  Compute AB = A*B
	for (l=0; l<MR*NR; ++l) {
		AB[l] = 0;
	}
	for (l=0; l<kc; ++l) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				AB[i+j*MR] += A[i]*B[j];
			}
		}
		A += MR;
		B += NR;
	}

	//  Update C <- beta*C
	if (beta==0.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] = 0.0;
			}
		}
	} else if (beta!=1.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] *= beta;
			}
		}
	}

	//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
	//                                  the above layer dgemm_nn)
	if (alpha==1.0) {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] += AB[i+j*MR];
			}
		}
	} else {
		for (j=0; j<NR; ++j) {
			for (i=0; i<MR; ++i) {
				C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
			}
		}
	}
}

//  Compute Y += alpha*X
static void GEMM(_geaxpy)(
	int m,
	int n,
	real alpha,
	const real *X,
	int incRowX,
	int incColX,
	real *Y,
	int incRowY,
	int incColY)
{
	int i, j;

	if (alpha!=1.0) {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
			}
		}
	} else {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
			}
		}
	}
}

//  Compute X *= alpha
static void GEMM(_gescal)(int m, int n, real alpha, real *X, int incRowX, int incColX)
{
	int i, j;

	if (alpha!=0.0) {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				X[i*incRowX+j*incColX] *= alpha;
			}
		}
	} else {
		for (j=0; j<n; ++j) {
			for (i=0; i<m; ++i) {
				X[i*incRowX+j*incColX] = 0.0;
			}
		}
	}
}

//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
static void GEMM(_macro_kernel)(
	int     mc,
	int     nc,
	int     kc,
	real  alpha,
	real  beta,
	real  *C,
	int     incRowC,
	int     incColC)
{
	int mp = (mc+MR-1) / MR;
	int np = (nc+NR-1) / NR;

	int _mr = mc % MR;
	int _nr = nc % NR;

	int mr, nr;
	int i, j;

	for (j=0; j<np; ++j) {
		nr    = (j!=np-1 || _nr==0) ? NR : _nr;

		for (i=0; i<mp; ++i) {
			mr    = (i!=mp-1 || _mr==0) ? MR : _mr;

			if (mr==MR && nr==NR) {
				GEMM(_micro_kernel)(kc, alpha, &GEMM(_A)[i*kc*MR], &GEMM(_B)[j*kc*NR], beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
			} else {
				GEMM(_micro_kernel)(kc, alpha, &GEMM(_A)[i*kc*MR], &GEMM(_B)[j*kc*NR], 0.0, GEMM(_C), 1, MR);
				GEMM(_gescal)(mr, nr, beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
				GEMM(_geaxpy)(mr, nr, 1.0, GEMM(_C), 1, MR, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
			}
		}
	}
}

//  Compute C <- beta*C + alpha*A*B
void GEMM(_nn_cpu)(
	int m, int n, int k,
	real alpha,
	const real *A, int incRowA, int incColA,
	const real *B, int incRowB, int incColB,
	real beta,
	real *C, int incRowC, int incColC)
{
	int mb = (m+MC-1) / MC;
	int nb = (n+NC-1) / NC;
	int kb = (k+KC-1) / KC;

	int _mc = m % MC;
	int _nc = n % NC;
	int _kc = k % KC;

	int mc, nc, kc;
	int i, j, l;

	real _beta;

	if (alpha==0.0 || k==0) {
		GEMM(_gescal)(m, n, beta, C, incRowC, incColC);
		return;
	}

	for (j=0; j<nb; ++j) {
		nc = (j!=nb-1 || _nc==0) ? NC : _nc;

		for (l=0; l<kb; ++l) {
			kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
			_beta = (l==0) ? beta : 1.0;

			GEMM(_pack_B)(kc, nc, &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB, GEMM(_B));

			for (i=0; i<mb; ++i) {
				mc = (i!=mb-1 || _mc==0) ? MC : _mc;

				GEMM(_pack_A)(mc, kc, &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA, GEMM(_A));

				GEMM(_macro_kernel)(mc, nc, kc, alpha, _beta, &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC);
			}
		}
	}
}

void GEMM(_c)(
	char		major,
	char		transA,
	char		transB,
	const int	m,
	const int	n,
	const int	k,
	const real	alpha,
	const real	*A,
	const int	ldA,
	const real	*B,
	const int	ldB,
	const real	beta,
	real		*C,
	const int	ldC)
{
	int i, j;

	//  Quick return if possible
	if (m==0 || n==0 || ((alpha==0.0 || k==0) && (beta==1.0))) {
		return;
	}

	//  And if alpha is exactly zero
	if (alpha==0.0) {
		if (beta==0.0) {
			for (j=0; j<n; j++) {
				for (i=0; i<m; i++) {
					C[i+j*ldC] = 0.0;
				}
			}
		} else {
			for (j=0; j<n; j++) {
				for (i=0; i<m; i++) {
					C[i+j*ldC] *= beta;
				}
			}
		}
		return;
	}

	//  Start the operations
	if (major == 'C') {
		if (transB=='N') {
			if (transA=='N') {
				// Form  C := alpha*A*B + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, 1, ldC);
			} else {
				// Form  C := alpha*A**T*B + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, 1, ldC);
			}
		} else {
			if (transA=='N') {
				// Form  C := alpha*A*B**T + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, 1, ldC);
			} else {
				// Form  C := alpha*A**T*B**T + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, 1, ldC);
			}
		}
	} else {
		if (transB=='N') {
			if (transA=='N') {
				// Form  C := alpha*A*B + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, ldA, 1, B, ldB, 1, beta, C, ldC, 1);
			} else {
				// Form  C := alpha*A**T*B + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, 1, ldA, B, ldB, 1, beta, C, ldC, 1);
			}
		} else {
			if (transA=='N') {
				// Form  C := alpha*A*B**T + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, ldA, 1, B, 1, ldB, beta, C, ldC, 1);
			} else {
				// Form  C := alpha*A**T*B**T + beta*C
				GEMM(_nn_cpu)(m, n, k, alpha, A, 1, ldA, B, 1, ldB, beta, C, ldC, 1);
			}
		}
	}
}

#undef MC
#undef KC
#undef NC

#undef MR
#undef NR

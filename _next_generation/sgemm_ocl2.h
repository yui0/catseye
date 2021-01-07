/* public domain Simple, Minimalistic, Fast GEMM library
 *	©2019-2021 Yuichiro Nakada
 *
 * Basic usage:
 *	sgemm_ocl_init(platform, device, max_buffer_size);
 *	sgemm_ocl('N', 'N', M, N, K, A, B, C);
 *	sgemm_ocl_finish();
 * */

#include "ocl.h"

#define TS 16		// Threadblock sizes

char sgemm_kcode[] = OCLSTRINGIFY(

// Tiled and coalesced version
__kernel void gemm_rnn(__global float* restrict gm, const int8 _info, const float4 _param)
{
	const int M = _info.s0;
	const int N = _info.s1;
	const int K = _info.s2;
	__global float* restrict A = (__global float* restrict)(gm + _info.s3);
	__global float* restrict B = (__global float* restrict)(gm + _info.s4);
	__global float* restrict C = (__global float* restrict)(gm + _info.s5);

	// Thread identifiers
	const int row = get_local_id(0); // Local row ID (max: TS)
	const int col = get_local_id(1); // Local col ID (max: TS)
	const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
	const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

	// Local memory to fit a tile of TS*TS elements of A and B
	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	// Initialise the accumulation register
	float acc = 0.0f;

	// Loop over all tiles
	const int numTiles = K/TS /*+1*/;
	for (int t=0; t<numTiles; t++) {
		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;
//		Asub[col][row] = A[tiledCol*M + globalRow]; // Column major
//		Bsub[col][row] = B[globalCol*K + tiledRow];
		Asub[col][row] = A[tiledCol + globalRow*K]; // Row major
		Bsub[col][row] = B[globalCol + N*tiledRow];

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k=0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
//			acc += Asub[row][k] * Bsub[k][col];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	{
		int t = numTiles;
		// Load one tile of A and B into local memory
		const int tiledRow = TS*t + row;
		const int tiledCol = TS*t + col;
		Asub[col][row] = (tiledCol>=K || globalRow>=M) ? 0 : A[tiledCol + globalRow*K]; // Row major
		Bsub[col][row] = (tiledRow>=K || globalCol>=N) ? 0 : B[globalCol + N*tiledRow];
//		Asub[col][row] = (tiledCol>=K || globalRow>=M) ? 0 : col + row*K;

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k=0; k<TS; k++) {
			acc += Asub[k][row] * Bsub[col][k];
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (globalRow >= M || globalCol >= N) return;

	// Store the final result in C
//	C[globalCol*M + globalRow] = acc; // Column major
	float z = _param.s1;
	if (z) z *= C[globalCol + globalRow*N];
	C[globalCol + globalRow*N] = _param.s0 * acc + z; // Row major
//	C[globalCol + globalRow*N] = Asub[col][row];
}

#define TRANSPOSEX 16
#define TRANSPOSEY 16
// Simple transpose kernel for a P * Q matrix
__kernel void transpose(__global float* gm, const int8 _info, const float4 _param)
{
	const int P = _info.s0;
	const int Q = _info.s1;
	__global float* input = (__global float*)(gm + _info.s2);
	__global float* output = (__global float*)(gm + _info.s3);

	// Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
	const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

	// Set-up the local memory for shuffling
	__local float buffer[TRANSPOSEX][TRANSPOSEY];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < P && ID1 < Q) {
		buffer[ty][tx] = input[ID1*P + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
	const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

	// Store the transposed result (coalesced)
	if (newID0 < Q && newID1 < P) {
		output[newID1*Q + newID0] = buffer[tx][ty];
	}
}

);

//#define OPENCL_SVM
int _info[8];
float _param[4];
args_t _args[] = {
#ifdef OPENCL_SVM
	{ CL_MEM_READ_WRITE|CL_MEM_SVM_FINE_GRAIN_BUFFER, 0, 0, OCL_SVM },
#else
	{ CL_MEM_READ_WRITE, 0, 0, OCL_BUFFER },
#endif
	{ 0, sizeof(int)*8, _info },
	{ 0, sizeof(float)*4, _param },
	{ 0, 0, 0, 0, 0 },
};
ocl_t _kernel[] = {
	// global: m*MDIMC/MWG, n*NDIMC/NWG
	{ _args, "gemm_rnn", 0, 2,{TS,TS} },

	// global: k, n
	{ _args, "transpose", 0, 2,{TRANSPOSEX,TRANSPOSEY} },
//	{ _args, "im2col", 0, 1,{16} },
};
int _ksz = sizeof(_kernel)/sizeof(_kernel[0]);
#define KGEMM_RNN	_kernel[0]
#define KTRANSPOSE	_kernel[1]
#define KIM2COL		_kernel[2]

void sgemm_ocl_init(int platform, int device, size_t size)
{
	_args[0].size = size;

	oclSetup(platform, device);
	oclKernel(_kernel, _ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", sgemm_kcode);
	oclKernelArgs(_kernel, _ksz);
}
static inline void sgemm_ocl(char ta, char tb, int m, int n, int k, float alpha, float *a, float *b, float beta, float *c)
{
	int mk = m*k;
	int kn = k*n;
	int mn = m*n;
	int off_a = 0;
	int off_b = mk;

#ifndef OPENCL_SVM
	oclWrite(_args[0].p, 0, sizeof(float)*mk, a);
	oclWrite(_args[0].p, sizeof(float)*mk, sizeof(float)*kn, b);
	if (beta!=0) oclWrite(_args[0].p, sizeof(float)*(mk+kn), sizeof(float)*mn, c);
#endif

	if (ta=='T') {
		_info[0] = m;	// a
		_info[1] = k;	// ta
		_info[2] = 0;	// input a
		_info[3] = off_a = mk +kn +mn;
		KTRANSPOSE.global_size[0] = ceil_int(m, TRANSPOSEX);
		KTRANSPOSE.global_size[1] = ceil_int(k, TRANSPOSEY);

		oclRun(&KTRANSPOSE);
	}
	if (tb=='T') {
		_info[0] = k;	// b
		_info[1] = n;	// tb
		_info[2] = mk;	// input b
		_info[3] = off_b = mk +kn +mn +mk;
		KTRANSPOSE.global_size[0] = ceil_int(k, TRANSPOSEX);
		KTRANSPOSE.global_size[1] = ceil_int(n, TRANSPOSEY);

		oclRun(&KTRANSPOSE);
	}

	_info[0] = m;
	_info[1] = n;
	_info[2] = k;
	_info[3] = off_a;	// a
	_info[4] = off_b;	// b
	_info[5] = mk +kn;	// c
	_param[0] = alpha;
	_param[1] = beta;
	KGEMM_RNN.global_size[0] = ceil_int(m, TS);
	KGEMM_RNN.global_size[1] = ceil_int(n, TS);
//	KGEMM_RNN.global_size[0] = ((m+1)/TS)*TS;
//	KGEMM_RNN.global_size[1] = ((n+1)/TS)*TS;
//	printf("M:%zu N:%zu ", KGEMM_RNN.global_size[0], KGEMM_RNN.global_size[1]);

	oclRun(&KGEMM_RNN);
#ifndef OPENCL_SVM
	oclRead(_args[0].p, sizeof(float)*(mk+kn), sizeof(float)*mn, c);
#endif
}
void sgemm_ocl_finish()
{
	oclReleaseKernel(_kernel, _ksz);
	oclFinish();
}


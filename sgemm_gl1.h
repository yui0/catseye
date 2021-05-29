//---------------------------------------------------------
//	Cat's eye
//
//		©2020 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os gpgpu_gl4.c -o gpgpu_gl4 `pkg-config --libs --cflags gl egl gbm` -lglfw
// dnf install mesa-libgbm-devel libdrm-devel mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel glfw-
#include "gpgpu_gl4.h"

// https://www.ibiblio.org/e-notes/webgl/gpu/mul/sgemm.htm
/*static const char _gemm1_cnn[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
	float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
	float B[];
};
layout (std430, binding = 2) writeonly buffer ssbC {
	float C[];
};
uniform float param[16]; // 0:M 1:N 2:K

void main() {
	int M = int(param[0]);
	int N = int(param[1]);
	int K = int(param[2]);

	// Thread identifiers
	uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
	uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

	if (M<=globalRow || N<=globalCol) return;

	// Compute a single element (loop over K)
	float acc = 0.0;
	for (uint k=0u; k<K; k++) {
		acc += A[k*M + globalRow] * B[globalCol*K + k];
	}

	// Store the result
	C[globalCol*M + globalRow] = acc;
}

);*/

static const char _gemm1_simple_rnn[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
	float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
	float B[];
};
layout (std430, binding = 2) writeonly buffer ssbC {
	float C[];
};
uniform int param[16]; // 0:M 1:N 2:K

void main() { // C := A * B
	int M = int(param[0]);
	int N = int(param[1]);
	int K = int(param[2]);

	// Thread identifiers
	uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
	uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

	if (M<=globalRow || N<=globalCol) return;

	// Compute a single element (loop over K)
	float acc = 0.0;
	for (uint k=0u; k<K; k++) {
		acc += A[k + globalRow*K] * B[globalCol + N*k]; // RNN
	}

	// Store the result
	C[globalCol + globalRow*N] = acc; // Row major
}

);

static const char _gemm1_rnn[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
	float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
	float B[];
};
layout (std430, binding = 2) buffer ssbC {
	float C[];
};
//uniform int param[16]; // 0:M 1:N 2:K
uniform float param[16]; // 0:M 1:N 2:K

void main() { // C := A * B
//	int M = param[0];
//	int N = param[1];
//	int K = param[2];
	int M = int(param[0]);
	int N = int(param[1]);
	int K = int(param[2]);
	float alpha = param[3];
	float beta = param[4];

	// Thread identifiers
	uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
	uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

	if (M<=globalRow || N<=globalCol) return;

	// Compute a single element (loop over K)
	float acc = 0.0;
	for (uint k=0u; k<K; k++) {
		acc += A[k + globalRow*K] * B[globalCol + N*k]; // RNN
	}

	// Store the result
//	C[globalCol + globalRow*N] = acc; // Row major
//	C[globalCol + globalRow*N] *= beta + alpha * acc; // Row major
	C[globalCol + globalRow*N] = alpha * acc + beta * C[globalCol + globalRow*N]; // Row major
}

);

static const char _gemm1_rnt[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
	float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
	float B[];
};
layout (std430, binding = 2) buffer ssbC {
	float C[];
};
uniform float param[16]; // 0:M 1:N 2:K

void main() {
	int M = int(param[0]);
	int N = int(param[1]);
	int K = int(param[2]);
	float alpha = param[3];
	float beta = param[4];

	// Thread identifiers
	uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
	uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

	if (M<=globalRow || N<=globalCol) return;

	// Compute a single element (loop over K)
	float acc = 0.0;
	for (uint k=0u; k<K; k++) {
		acc += A[k + globalRow*K] * B[globalCol*K + k]; // RNT
	}

	// Store the result
	C[globalCol + globalRow*N] = alpha * acc + beta * C[globalCol + globalRow*N]; // Row major
}

);

static const char _gemm1_rtn[] = STRINGIFY(

\n#version 430\n

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout (std430, binding = 0) readonly buffer ssbA {
	float A[];
};
layout (std430, binding = 1) readonly buffer ssbB {
	float B[];
};
layout (std430, binding = 2) buffer ssbC {
	float C[];
};
uniform float param[16]; // 0:M 1:N 2:K

void main() {
	int M = int(param[0]);
	int N = int(param[1]);
	int K = int(param[2]);
	float alpha = param[3];
	float beta = param[4];

	// Thread identifiers
	uint globalRow = gl_GlobalInvocationID.x; // Row ID of C (0..M)
	uint globalCol = gl_GlobalInvocationID.y; // Col ID of C (0..N)

	if (M<=globalRow || N<=globalCol) return;

	// Compute a single element (loop over K)
	float acc = 0.0;
	for (uint k=0u; k<K; k++) {
		acc += A[M*k + globalRow] * B[globalCol + N*k]; // RTN
	}

	// Store the result
	C[globalCol + globalRow*N] = alpha * acc + beta * C[globalCol + globalRow*N]; // Row major
}

);

#define GEMM1_RNN	0
#define GEMM1_RNT	1
#define GEMM1_RTN	2
#define GEMM1_SRNN	3
GLuint sgemm_program[10];
void sgemm_gl_init(int s1, int s2, int s3)
{
	coInit();
//	sgemm_program[0] = coCreateShaderProgram(_gemm1_cnn);
	sgemm_program[GEMM1_RNN] = coCreateShaderProgram(_gemm1_rnn);
	sgemm_program[GEMM1_RNT] = coCreateShaderProgram(_gemm1_rnt);
	sgemm_program[GEMM1_RTN] = coCreateShaderProgram(_gemm1_rtn);

	sgemm_program[GEMM1_SRNN] = coCreateShaderProgram(_gemm1_simple_rnn);

	int size[] = {s1, s2, s3};
	coCreateBuffer(size, 3);
}
void sgemm_gl_finish()
{
	coDeleteBuffer();
	coDeleteProgram(sgemm_program[GEMM1_RNN]);
	coDeleteProgram(sgemm_program[GEMM1_RNT]);
	coDeleteProgram(sgemm_program[GEMM1_RTN]);

	coDeleteProgram(sgemm_program[GEMM1_SRNN]);
}

/*inline*/ void sgemm_gl(int type, int m, int n, int k, float alpha, float *a, float *b, float beta, float *c)
{
	float param[16];
	param[0] = m;
	param[1] = n;
	param[2] = k;
	param[3] = alpha;
	param[4] = beta;
	coWrite(0, m*k*sizeof(float), a);
	coWrite(1, k*n*sizeof(float), b);
/*	if (beta!=0)*/ coWrite(2, m*n*sizeof(float), c);
//	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // Sync here to make writes visible
	coRun(sgemm_program[type], m/8+1, n/8+1, 1, param);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // Sync here to make writes visible
//	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	coRead(2, m*n*sizeof(float), c);

/*	printf("sgemm_gl%d: %d,%d,%d\n", type, m, n, k);
	for (int i=0; i<10; i++) printf("%f ", c[i]);
	printf("\n");
	coRead(0, m*k*sizeof(float), a);
	for (int i=0; i<10; i++) printf("%f ", a[i]);
	printf("\n");
	coRead(0, k*n*sizeof(float), b);
	for (int i=0; i<10; i++) printf("%f ", b[i]);
	printf("\n");*/
}

#ifndef CATS_OPENGL
static inline void im2col(const float *im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, float *col)
{
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int channels_col = channels * kernel_h * kernel_w;

	for (int c=0; c<channels_col; c++) {
		int w_offset = c % kernel_w;
		int h_offset = (c / kernel_w) % kernel_h;
		int c_im = c / kernel_h / kernel_w;
		for (int h=0; h<height_col; h++) {
			for (int w=0; w<width_col; w++) {
				int h_pad = h * stride_h - pad_h + h_offset;
				int w_pad = w * stride_w - pad_w + w_offset;
				if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
					col[(c * height_col + h) * width_col + w] =
						im[(c_im * height + h_pad) * width + w_pad];
				else
					col[(c * height_col + h) * width_col + w] = 0;
			}
		}
	}
}
float workspace[256*256*128*64];
static inline void gl_convolution_LReLU(float *inputs, int ich, int w, int h, float *weights, int k, int pad, int stride, float *outputs, int ch, float *bias)
{
	// im2col(pix, 3, h, w, 4, 4, 2, 2, 1, 1, workspace);
	im2col(inputs, ich, h, w, k, k, pad, pad, stride, stride, workspace);
	int hcol = (h + 2 * pad - k) / stride + 1;
	int wcol = (w + 2 * pad - k) / stride + 1;

	// gemm('N', 'N', ch, wcol*hcol*batch, k*k*ich, magic_kernel, workspace, pix);
	// https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
	int M, N, K;
	float param[16];
	param[0] = M = ch;		// M
	param[1] = N = wcol*hcol /* *batch */;// N
	param[2] = K = k*k*ich;	// K
	coWrite(0, M*K*sizeof(float), weights); // a
	coWrite(1, K*N*sizeof(float), workspace); // b
	coRun(sgemm_program[GEMM1_SRNN], M/8+1, N/8+1, 1, param);
	coRead(2, M*N*sizeof(float), outputs); // c

	float *p = outputs;
	for (int i=0; i<ch; i++) {
		for (int n=0; n<wcol*hcol; n++) {
			*p += bias[i];
			*p = *p>0 ? (*p) : (*p)*0.1;
			p++;
		}
	}
}
#endif

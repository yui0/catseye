//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

char kernel_code[] =
#include "catseye.cl"

cl_mem d_mem[6];
cl_uint param[8];
args_t args[] = {
	{ CL_MEM_READ_WRITE, 0, &d_mem[0], 0, -1, 0 },	// x
	{ CL_MEM_READ_WRITE, 0, &d_mem[1], 0, 1, 1 },	// w
	{ CL_MEM_READ_WRITE, 0, &d_mem[2], 0, -1, 1 },	// o
	{ CL_MEM_READ_WRITE, 0, &d_mem[3], 0, 1, 1 },	// d
	{ CL_MEM_READ_WRITE, 0, &d_mem[4], 0, -1, 0 },	// t
	{ CL_MEM_READ_WRITE, 0, &d_mem[5], 0, 0, 0 },	// sync
	{ 0, sizeof(param), &param, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
	{ "forward",	0, {256,0,0,},{256,0,0,}, args },
	{ "train",	0, {1024,0,0,},{256,0,0,}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

char acts[][20] = {
	"identity",
	"softmax",
	"sigmoid",
	"normal_tanh",
	"scaled_tanh",
	"relu",
	"LeakyReLU",
};

char *strrep(char *s, char *b, char *a)
{
	size_t b_len = strlen(b);
	size_t a_len = strlen(a);

	char *sub = strstr(s, b);
	if (!sub) return strdup(s);

	char *newstr = malloc(strlen(s) + a_len - b_len + 1);
	if (!newstr) return 0;
	*newstr = '\0';

	strncat(newstr, s, sub - s);
	strcat(newstr, a);
	strcat(newstr, sub + b_len);
	return newstr;
}

void CatsEye_clSetup(CatsEye *this)
{
	// generate dynamic code
	#define BUFSIZE	4096
	char code[4][BUFSIZE];
	int osize = 0, dsize, in, out;
	int wsize = 0;
	code[0][0] = code[1][0] = code[2][0] = 0;
	for (int i=0; i<this->layers-1; i++) {
		int *u = &this->u[LPLEN*(i+1)]; 
		in = u[SIZE-LPLEN];
		out = u[SIZE];
		dsize = osize-this->u[SIZE]-1;

		switch (u[TYPE]) {
		case CATS_CONV:
			if (i==0) {
				snprintf(code[3], BUFSIZE, "\t\tconvolutional_layer_forward3x3(p, w, o+oo+%d, %d, %d, %d, %d);\n\t\tconvolutional_layer_%s(o+oo+%d, %d, %d, %d);\n",
					in+1, u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN], u[CHANNEL], acts[u[ACT]], in+1, u[XSIZE], u[YSIZE], u[CHANNEL]);
				strcat(code[0], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tconvolutional_layer_update3x3(eta, p, w, d+dd, %d, %d, %d, %d);\n", u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN], u[CHANNEL]);
				strcat(code[2], code[3]);
			} else {
				snprintf(code[3], BUFSIZE, "\t\tconvolutional_layer_forward3x3(o+oo+%d, w+%d, o+oo+%d, %d, %d, %d, %d);\n\t\tconvolutional_layer_%s(o+oo+%d, %d, %d, %d);\n",
					osize, wsize, osize+in+1, u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN], u[CHANNEL], acts[u[ACT]], in+1, u[XSIZE], u[YSIZE], u[CHANNEL]);
				strcat(code[0], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tconvolutional_layer_backward3x3(w+%d, d+dd+%d, d+dd+%d, %d, %d, %d, %d);\n\t\tconvolutional_layer_d%s(o+oo+%d, %d, %d, %d);\n", wsize, dsize, dsize+in+1, u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN], u[CHANNEL], acts[u[ACT-LPLEN]], osize, u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN]);
				//strcat(code[1], code[3]);
				strcat(code[3], code[1]);
				strcpy(code[1], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tconvolutional_layer_update3x3(eta, o+oo+%d, w+%d, d+dd+%d, %d, %d, %d, %d);\n", osize, wsize, dsize+in+1, u[XSIZE], u[YSIZE], u[CHANNEL-LPLEN], u[CHANNEL]);
				strcat(code[2], code[3]);
			}
			break;
		case CATS_MAXPOOL:
			break;
		default:
			if (i==0) {
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
//	for (int i=0; i<SIZE(0); i++) {
//		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
//	}
#endif
				snprintf(code[3], BUFSIZE, "\t\tlinear_forward_%s(p, w, o+oo+%d, %d, %d);\n",
					acts[u[ACT]], in+1, in, out);
				strcat(code[0], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tlinear_update(eta, p, w, d+dd, %d, %d);\n", in, out);
				strcat(code[2], code[3]);
			} else {
				snprintf(code[3], BUFSIZE, "\t\tlinear_forward_%s(o+oo+%d, w+%d, o+oo+%d, %d, %d);\n",
					acts[u[ACT]], osize, wsize, osize+in+1, in, out);
				strcat(code[0], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tlinear_backward_%s(o+oo+%d, w+%d, d+dd+%d, d+dd+%d, %d, %d);\n", acts[u[ACT-LPLEN]], osize, wsize, dsize, dsize+in+1, in, out);
				//strcat(code[1], code[3]);
				strcat(code[3], code[1]);
				strcpy(code[1], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tlinear_update(eta, o+oo+%d, w+%d, d+dd+%d, %d, %d);\n", osize, wsize, dsize+in+1, in, out);
				strcat(code[2], code[3]);
			}
		}
		//osize += in+1;
		osize += u[SIZE]+1;
		wsize += this->ws[i];
	}
	int a = this->layers-1;
	int loss = this->u[a*LPLEN+STRIDE];
	if (loss) {
		snprintf(code[3], BUFSIZE, "\t\tloss_mse(o+oo+%d, d+dd+%d, t+seed*%d, %d);\n%s", osize, dsize+in+1, out, out/*this->u[a*LPLEN+SIZE]*/, code[1]);
	} else {
		snprintf(code[3], BUFSIZE, "\t\tloss_0_1(o+oo+%d, d+dd+%d, label, %d);\n%s", osize, dsize+in+1, out, code[1]);
	}
	strcpy(code[1], code[3]);

	snprintf(code[3], BUFSIZE, "\t\tglobal const float *p = x + seed*%d;\n\n\t\tuint dd = m*%d;\n\t\tuint oo = m*%d;\n\n", this->u[SIZE], osize-this->u[SIZE]+out-1, osize+out/*+1*/);
	strcat(code[3], code[0]);
	strcat(code[3], code[1]);
	strcat(code[3], code[2]);
	printf("%s", code[3]);
	char *s = strrep(kernel_code, "\%GEN_CODE\%", code[3]);
//	snprintf(code[3], BUFSIZE, "\t\tuint size = %d;\n\t\tvdot(acc, d, d, mse, %d);\n", SIZE(1), SIZE(1));
	snprintf(code[3], BUFSIZE, "\t\tuint size = %d;\n\t\tvdot(acc, d+%d, d+%d, mse, %d);\n", SIZE(2), SIZE(1)+1, SIZE(1)+1, SIZE(2));
	//char *kcode = strrep(s, "\%MSE_CODE\%", code[3]);
//	printf("%s", code[3]);
	char *kcode = strrep(s, "\%PREPROCESSOR\%", ""/*"#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable\n#pragma OPENCL SELECT_ROUNDING_MODE rte\n"*/);
	free(s);
//	printf("%s", kcode);

	// set arguments
	args[0].size = 0;	// x, set at CatsEye_train
	args[1].size = sizeof(numerus)*this->wsize *CATS_MBATCH;
//	for (int i=1; i<8; i++) memcpy(this->wdata+this->wsize*i, this->wdata, sizeof(numerus)*this->wsize);
	args[1].s = this->wdata;
	args[2].size = sizeof(numerus)*this->osize *CATS_MBATCH;
	args[2].s = this->odata;
	args[3].size = sizeof(numerus)*this->dsize *CATS_MBATCH;
	args[3].s = this->ddata;
	args[4].size = 0;	// t, set at CatsEye_train
	args[5].size = sizeof(cl_int)*1024*2;

	// compile the code
	// http://dhruba.name/2012/12/24/opencl-cookbook-10-tips-for-high-performance-kernels/
	oclSetup(0, 0);
	oclKernel(kernel, ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", kcode);
	oclKernelArgs(kernel, ksz);
	free(kcode);
}

void CatsEye_clFinish()
{
	oclReleaseKernel(kernel, ksz);
	oclFinish();
}

#if 0
void CatsEye_forward(CatsEye *this, numerus *x)
{
#if 0
	// calculation of input layer
	memcpy(this->o[0], x+n, SIZE(0)*sizeof(numerus));
	this->o[0][SIZE(0)] = 1;	// for bias
#endif

	int n = x - this->xdata;
	args[0].s = this->xdata;
	param[0] = n;

	oclKernelArgsWrite(args);
	oclRun(&kernel[0]);
	oclKernelArgsRead(args);

/*	oclKernelArgsWrite(args);
	param[0] = 1;
	param[1] = n;
	param[2] = 0;
	param[3] = 784+1;
	param[4] = 784;
	param[5] = 200;
	oclRun(&kernel[3]);
	param[0] = 0;
	param[1] = 784+1;
	param[2] = 785*200;
	param[3] = 784+1+200+1;
	param[4] = 200;
	param[5] = 10;
	oclRun(&kernel[2]);
	oclKernelArgsRead(args);*/
}
#endif

void CatsEye_train(CatsEye *this, numerus *x, void *t, int N, int repeat, numerus eta)
{
	this->xdata = x;
	this->xsize = N;

	int batch = N;			// for random
	if (RANDOM) batch = RANDOM;

	int a = this->layers-1;
	int loss = this->u[a*LPLEN+STRIDE];
//	if (!loss && x==t) loss = 1;
	if (loss) {
		args[4].size = sizeof(numerus)*SIZE(a)*N;	// (SIZE(a)+1)
	} else {
		args[4].size = sizeof(numerus)*N;
	}
	args[0].size = sizeof(numerus)*SIZE(0)*N;		// (SIZE(0)+1)
	args[0].s = x;
	args[4].s = t;
	param[0] = N;
	param[1] = batch;
	param[4] = eta*1e8;
	oclKernelArgs(kernel, ksz);
	oclKernelArgsWrite(args);

	struct timeval start, stop;
	gettimeofday(&start, NULL);
	for (int times=0; times<repeat; times++) {
		param[2] = xor128();
		param[3] = times;
		oclRun(&kernel[1]);
		oclKernelArgsRead(args);
/*#ifdef CATS_AUTOENCODER
			// tied weight
			numerus *dst = this->w[1];
			for (int i=0; i<SIZE(1); i++) {
				for (int j=0; j<SIZE(0); j++) {
					this->w[1][j + SIZE(1)*i] = this->w[0][SIZE(1)*j + i];
				}
			}
#endif*/
#ifdef CL_DEBUG
		memcpy(&this->wdata[this->wsize*2], this->wdata, this->wsize*sizeof(numerus));
		memcpy(this->wdata, &this->wdata[this->wsize], this->wsize*sizeof(numerus));
		//memcpy(&this->odata[this->osize], &this->odata[0], this->osize*sizeof(numerus));
		/*for (int i=0; i<this->dsize; i++) {
			printf("%f/%d ", this->ddata[this->dsize+i], i);
		}*/

		void CatsEye_forward(CatsEye *this, numerus *x);
		void CatsEye_loss_mse(CatsEye *this, int c, void *t, int n);
		CatsEye_forward(this, x+1*SIZE(0));
		CatsEye_loss_mse(this, a, t, 1);
		for (int i=this->layers-2; i>0; i--) {
			CatsEye_layer_backward[TYPE(i+1)](this->o[i], this->w[i], this->d[i-1], this->d[i], &this->u[LPLEN*(i+1)]);
		}

		for (int i=SIZE(0); i<this->osize; i++) {
			if (fabs(this->odata[i] - this->odata[this->osize+i]) > 1e-7) printf("%f/%f/%d ", this->odata[i], this->odata[this->osize+i], i);
			/*if (fabs(this->odata[i] - this->odata[this->osize+i]) > 1e-7) {
				union Num {
					int i;
					float f;
				};
				union Num a; a.f = this->odata[i];
				union Num b; b.f = this->odata[this->osize+i];
				printf("%f(%x)/%f(%x)/%d ", a.f, a.i, b.f, b.i, i);
			}*/
		}
//		float *w = this->w[0];
//		printf("\nx:[%f,%f,%f],w:[%f,%f,%f]\n",x[2],x[3],x[4],w[0],w[20*1],w[20*2]);
		for (int i=0; i<this->dsize; i++) {
			if (fabs(this->ddata[i] - this->ddata[this->dsize+i]) > 1e-7) printf("%f/%f/%d ", this->ddata[i], this->ddata[this->dsize+i], i);
		}
		/*for (int i=0; i<this->wsize; i++) {
			if (fabs(this->wdata[i] - this->wdata[this->wsize*2+i]) > 1e-7) printf("%f/%f/%d ", this->wdata[i], this->wdata[this->wsize*2+i], i);
		}*/
		exit(0);
#endif

		// calculate the mean squared error
		numerus err = 0;
		numerus mse = 0;
		for (int i=0; i<SIZE(2); i++) {
			mse += 0.5 * (this->d[1][i] * this->d[1][i]);
		}
		err = 0.5 * (err + mse);

		printf("epochs %d, mse %f", times, err);
		gettimeofday(&stop, NULL);
		printf(" [%.2fs]", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);
		printf("\n");
	}
}

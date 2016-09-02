//---------------------------------------------------------
//	Cat's eye
//
//		©2016 Yuichiro Nakada
//---------------------------------------------------------

#include "ocl.h"

char kernel_code[] =
#include "catseye.cl"

cl_mem d_mem[6];
unsigned int param[8];
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
	#define BUFSIZE	2048
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
//				snprintf(code[3], BUFSIZE, "\t\tlinear_update(%f, p, w, d+dd, %d, %d);\n", 0.01, in, out);
				snprintf(code[3], BUFSIZE, "\t\tlinear_update(eta, p, w, d+dd, %d, %d);\n", in, out);
				strcat(code[2], code[3]);
			} else {
				snprintf(code[3], BUFSIZE, "\t\tlinear_forward_%s(o+oo+%d, w+%d, o+oo+%d, %d, %d);\n",
					acts[u[ACT]], osize, wsize, osize+in+1, in, out);
				strcat(code[0], code[3]);
				snprintf(code[3], BUFSIZE, "\t\tlinear_backward_%s(o+oo+%d, w+%d, d+dd+%d, d+dd+%d, %d, %d);\n", acts[u[ACT]], osize, wsize, dsize, dsize+in+1, in, out);
				strcat(code[1], code[3]);
//				snprintf(code[3], BUFSIZE, "\t\tlinear_update(%f, o+oo+%d, w+%d, d+dd+%d, %d, %d);\n", 0.01, osize, wsize, dsize+in+1, in, out);
				snprintf(code[3], BUFSIZE, "\t\tlinear_update(eta, o+oo+%d, w+%d, d+dd+%d, %d, %d);\n", osize, wsize, dsize+in+1, in, out);
				strcat(code[2], code[3]);
			}
		}
		osize += in+1;
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

	snprintf(code[3], BUFSIZE, "\t\tglobal const float *p = x + seed*%d;\n\n\t\tuint dd = m*(%d);\n\t\tuint oo = m*(%d);\n\n", this->u[SIZE], osize-this->u[SIZE]+out, osize+out+1);
	strcat(code[3], code[0]);
	strcat(code[3], code[1]);
	strcat(code[3], code[2]);
	printf("%s", code[3]);
	char *s = strrep(kernel_code, "\%GEN_CODE\%", code[3]);
//	snprintf(code[3], BUFSIZE, "\t\tuint size = %d;\n\t\tvdot(acc, d, d, mse, %d);\n", SIZE(1), SIZE(1));
	snprintf(code[3], BUFSIZE, "\t\tuint size = %d;\n\t\tvdot(acc, d+%d, d+%d, mse, %d);\n", SIZE(2), SIZE(1)+1, SIZE(1)+1, SIZE(2));
	char *kcode = strrep(s, "\%MSE_CODE\%", code[3]);
//	printf("%s", code[3]);
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
#ifdef CATS_DENOISING_AUTOENCODER
	// Denoising Autoencoder (http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html)
	for (int i=0; i<SIZE(0); i++) {
		this->o[0][i] *= binomial(/*0.7(30%)*/0.5);
	}
#endif
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

//	memcpy(this->o[0], x+n, SIZE(0)*sizeof(numerus));

/*	for (int i=0; i<200; i++) printf("%f ", this->o[1][i]);
	printf("\n%d %f\n",SIZE(0),this->o[0][0]);
	CatsEye_layer_forward[TYPE(1)](this->o[0], this->w[0], this->z[0], this->o[1], &this->u[LPLEN*(1)]);
	for (int i=0; i<200; i++) printf("%f ", this->o[1][i]);
	printf("\n");
	exit(0);*/
/*	for (int i=0; i<10; i++) printf("%f ", this->o[2][i]);
	printf("\n%d %f\n",SIZE(0),this->o[1][0]);
	CatsEye_layer_forward[TYPE(2)](this->o[1], this->w[1], this->z[1], this->o[2], &this->u[LPLEN*(2)]);
	for (int i=0; i<10; i++) printf("%f ", this->o[2][i]);
	printf("\n");
	exit(0);*/
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
//		args[4].size = sizeof(numerus)*(SIZE(a)+1)*N;
		args[4].size = sizeof(numerus)*SIZE(a)*N;
	} else {
		args[4].size = sizeof(numerus)*N;
	}
//	args[0].size = sizeof(numerus)*(SIZE(0)+1)*N;
	args[0].size = sizeof(numerus)*SIZE(0)*N;
	oclKernelArgs(kernel, ksz);

	args[0].s = x;//this->xdata;
	args[4].s = t;
	param[0] = N;
	param[1] = batch;
	param[4] = eta*1e8;
//	for (int i=0; i<60; i++) printf("%f ",this->wdata[i]);
//printf("\n%x\n",args[1].p);
oclKernelArgsWrite(args);
//sleep(2);

#ifdef CATS_TIME
	struct timeval start, stop;
	gettimeofday(&start, NULL);
#endif
	for (int times=0; times<repeat; times++) {
		param[2] = xor128();
		param[3] = times;
#ifdef CATS_TIME
		gettimeofday(&stop, NULL);
		param[5] = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec);
#endif
//		oclKernelArgsWrite(args);
		oclRun(&kernel[1]);
//		oclKernelArgsRead(args);

		/*oclKernelArgsWrite(args);
		for (int n=0; n<batch; n++) {
			int sample = RANDOM ? (frand()*N) : n;
			param[0] = 1;
			param[1] = sample*784;
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

			param[0] = 784+1+200+1;
			param[1] = 200+1;
			param[2] = ((int*)t)[sample];
			param[3] = 10;
			oclRun(&kernel[7]);
			param[0] = 784+1;
			param[1] = 785*200;
			param[2] = 0;
			param[3] = 200+1;
			param[4] = 200;
			param[5] = 10;
			oclRun(&kernel[4]);

			param[0] = 1;
			param[1] = sample*784;
			param[2] = 0;
			param[3] = 0;
			param[4] = 784;
			param[5] = 200;
			oclRun(&kernel[6]);
			param[0] = 0;
			param[1] = 784+1;
			param[2] = 785*200;
			param[3] = 200+1;
			param[4] = 200;
			param[5] = 10;
			oclRun(&kernel[6]);
		}
		oclKernelArgsRead(args);*/

/*#ifdef CATS_AUTOENCODER
			// tied weight
			numerus *dst = this->w[1];
			for (int i=0; i<SIZE(1); i++) {
				for (int j=0; j<SIZE(0); j++) {
					this->w[1][j + SIZE(1)*i] = this->w[0][SIZE(1)*j + i];
				}
			}
#endif*/
		// calculate the mean squared error
		numerus err = 0;
		numerus mse = 0;
		for (int i=0; i<SIZE(2); i++) {
			mse += 0.5 * (this->d[1][i] * this->d[1][i]);
		}
		err = 0.5 * (err + mse);

		printf("epochs %d, mse %f", times, err);
#ifdef CATS_TIME
		gettimeofday(&stop, NULL);
		printf(" [%.2fs]", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec)*0.001*0.001);
#endif
		printf("\n");
	}
	oclKernelArgsRead(args);
}

// Gradient check: the analytic gradients of every layer type against central finite differences.
// (The conv layer averages its weight gradient over the batch; linear/deconv/bn sum it.)
#define CATS_QUIET
#include "../catseye.h"

static double fwd_loss(CatsEye *c, real *X, real *T, int16_t *cls)
{
	CatsEye_layer *l0 = &c->layer[0];
	memcpy(l0->x, X, sizeof(real)*l0->inputs*c->batch);
	CatsEye_layer *last = &c->layer[c->layers-1];
	if (T) memcpy(c->label, T, sizeof(real)*last->inputs*c->batch);
	if (cls) memcpy(c->clasify, cls, sizeof(int16_t)*c->batch);
	for (int i=0; i<c->layers; i++) c->layer[i].forward(&c->layer[i]);
	double L = 0;
	if (T) {
		if (last->type==CATS_LOSS_BCE || last->type==CATS_SIGMOID_BCE) for (int i=0; i<last->inputs*c->batch; i++) { double y=last->x[i]; L += -T[i]*log(y+1e-12) - (1-T[i])*log(1-y+1e-12); }
		else for (int i=0; i<last->inputs*c->batch; i++) { double d = last->x[i]-T[i]; L += 0.5*d*d; }
	} else {
		for (int b=0; b<c->batch; b++) L += -log(last->x[last->inputs*b + cls[b]]);
	}
	return L;
}

static int g_fail;
static int check_(const char *name, CatsEye_layer *u, int n, int batch, int classes)
{
	CatsEye c = { .batch=batch };
	_CatsEye__construct(&c, u, n);
	CatsEye_layer *l0 = &c.layer[0], *last = &c.layer[n-1];
	int in = l0->inputs*batch, out = last->inputs*batch;
	for (int i=0; i<n; i++) if (c.layer[i].type==CATS_BATCHNORMAL) { c.layer[i].w[2*c.layer[i].r] = 1.3; c.layer[i].w[2*c.layer[i].r+1] = 0.2; }
	real *X = malloc(sizeof(real)*in), *T = malloc(sizeof(real)*out);
	int16_t cls[64];
	for (int i=0; i<in; i++) X[i] = rand_normal(0, 1);
	for (int i=0; i<out; i++) T[i] = frand();
	for (int b=0; b<batch; b++) cls[b] = (int16_t)(b % (classes?classes:1));
	// nonzero BN gamma/beta etc are left default
	real *Tp = classes ? NULL : T;
	for (int i=0; i<n; i++) c.layer[i].eta = c.layer[i].type==CATS_LINEAR ? 1 : 0;
	// analytic
	real *w0 = malloc(sizeof(real)*c.wsize);
	memcpy(w0, c.wdata, sizeof(real)*c.wsize);
	fwd_loss(&c, X, Tp, classes?cls:NULL);
	memset(c.wdata + c.wsize, 0, sizeof(real)*c.wsize);
	CatsEye_layer *l = &c.layer[c.end];
	for (int i=c.end; i>=c.start; i--) { if (i>c.stop) l->backward(l); if (!(l->fix&1)) l->update(l); l--; }
	real *an = malloc(sizeof(real)*c.wsize);
	memcpy(an, c.wdata + c.wsize, sizeof(real)*c.wsize);
	// linear bias: grad = old - new
	for (int i=0; i<n; i++) {
		CatsEye_layer *q = &c.layer[i];
		if (q->type==CATS_LINEAR) {
			int off = q->w - c.wdata;
			for (int k=0; k<q->outputs; k++) an[off + q->inputs*q->outputs + k] = w0[off + q->inputs*q->outputs + k] - q->w[q->inputs*q->outputs + k];
		}
	}
	memcpy(c.wdata, w0, sizeof(real)*c.wsize);
	double maxerr = 0; int worst = -1; double worst_an=0, worst_nu=0;
	for (int i=0; i<n; i++) {
		CatsEye_layer *q = &c.layer[i];
		if (!q->wsize) continue;
		int off = q->w - c.wdata;
		int tot = c.ws[i];
		double scale = 1; // conv/deconv update divides by batch for conv only
		(void)scale;
		for (int k=(q->type==CATS_BATCHNORMAL?2*q->r:0); k<tot; k++) {
			double h = 1e-5, o = c.wdata[off+k];
			c.wdata[off+k] = o+h; double Lp = fwd_loss(&c, X, Tp, classes?cls:NULL);
			c.wdata[off+k] = o-h; double Lm = fwd_loss(&c, X, Tp, classes?cls:NULL);
			c.wdata[off+k] = o;
			double nu = (Lp-Lm)/(2*h);
			double a = an[off+k]; if (q->type==CATS_CONV||q->type==CATS_CONV1D) a *= batch; /* conv averages over the batch */
			double err = fabs(nu-a)/(fabs(nu)+fabs(a)+1e-8);
			if (fabs(nu-a) < 1e-10) err = 0;	// below the finite-difference noise floor
			if (!(err==err)) err = 1;	// NaN is a failure
			if (err > maxerr) { maxerr = err; worst = i*100000+k; worst_an=a; worst_nu=nu; }
		}
	}
		printf("%-28s batch=%d maxrelerr=%.3g %s", name, batch, maxerr, maxerr<1e-4?"OK":"** FAIL **");
	if (maxerr>=1e-4) printf("  (layer %d idx %d: analytic %g numeric %g ratio %g)", worst/100000, worst%100000, worst_an, worst_nu, worst_an/worst_nu);
	printf("\n");
	return maxerr<1e-4;
}
static int check(const char *name, CatsEye_layer *u, int n, int batch, int classes)
{
	int ok = check_(name, u, n, batch, classes);
	if (!ok) g_fail++;
	return ok;
}

int main()
{
	xoroshiro128plus_init(getenv("SEED")?atoi(getenv("SEED")):1);
	for (int batch=1; batch<=3; batch+=2) {
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {4, CATS_LINEAR}, {0, CATS_ACT_SIGMOID}, {4, CATS_LOSS_MSE} };
	  check("mlp tanh/sigmoid mse", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_RELU}, {4, CATS_LINEAR}, {0, CATS_ACT_LEAKY_RELU}, {4, CATS_LOSS_MSE} };
	  check("mlp relu/leaky", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_ELU}, {4, CATS_LINEAR}, {0, CATS_ACT_LEAKY_RELU}, {4, CATS_LOSS_MSE} };
	  check("mlp elu", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {4, CATS_LINEAR}, {0, CATS_ACT_SOFTMAX}, {4, CATS_SOFTMAX_CE} };
	  check("mlp softmax_ce", u, 5, batch, 4); }
	{ CatsEye_layer u[] = { {36, CATS_CONV, .ksize=3, .stride=1, .ch=3}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  u[0].padding=0; check("conv3 s1 p0", u, 4, batch, 0); }
	{ CatsEye_layer u[] = { {36, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv3 s1 p1", u, 4, batch, 0); }
	{ CatsEye_layer u[] = { {64, CATS_CONV, .ksize=3, .stride=2, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv3 s2 p1", u, 4, batch, 0); }
	{ CatsEye_layer u[] = { {36, CATS_CONV, .ksize=1, .stride=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv1x1", u, 4, batch, 0); }
	{ CatsEye_layer u[] = { {36, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-conv", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {64, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_MAXPOOL, .ksize=2, .stride=2}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-maxpool-conv", u, 7, batch, 0); }
	{ CatsEye_layer u[] = { {64, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_AVGPOOL, .ksize=2, .stride=2}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-avgpool-conv", u, 7, batch, 0); }
	{ CatsEye_layer u[] = { {64, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=3}, {0, CATS_ACT_TANH}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=3}, {0, CATS_GAP}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-conv-gap-linear", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {16, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=4}, {0, CATS_ACT_TANH}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=4}, {0, CATS_PIXELSHUFFLER, .r=2, .ch=1}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-pixelshuffle", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {16, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_DECONV, .ksize=2, .stride=2, .padding=0, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-deconv k2s2", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {16, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_DECONV, .ksize=4, .stride=2, .padding=1, .ch=2}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-deconv k4s2p1", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {16, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=2}, {0, CATS_BATCHNORMAL}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("conv-bn-linear", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_BATCHNORMAL}, {0, CATS_ACT_TANH}, {4, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("linear-bn", u, 5, batch, 0); }

	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {4, CATS_LINEAR}, {0, CATS_ACT_SOFTMAX}, {4, CATS_LOSS_CE} };
	  check("softmax + ce loss", u, 5, batch, 4); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {4, CATS_LINEAR}, {0, CATS_ACT_SOFTMAX}, {4, CATS_LOSS_MSE} };
	  check("softmax + mse", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {4, CATS_LINEAR}, {0, CATS_ACT_SIGMOID}, {4, CATS_SIGMOID_BCE} };
	  check("sigmoid_bce", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_SIGMOID}, {4, CATS_LINEAR}, {0, CATS_ACT_SIGMOID}, {4, CATS_LOSS_BCE} };
	  check("bce loss", u, 5, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR}, {5, CATS_ACT_TANH}, {5, CATS_LINEAR}, {0, CATS_ACT_SIGMOID}, {0, CATS_ACT_LOG}, {5, CATS_LOSS_MSE} };
	  check("log act", u, 6, batch, 0); }
	{ CatsEye_layer u[] = { {6, CATS_LINEAR, .name="a"}, {5, CATS_ACT_TANH}, {5, CATS_LINEAR}, {0, CATS_ACT_TANH}, {5, CATS_LINEAR}, {0, CATS_SHORTCUT, .layer="a"}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("shortcut(linear)", u, 8, batch, 0); }
	{ CatsEye_layer u[] = { {36, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=1, .name="a"}, {0, CATS_ACT_TANH}, {0, CATS_CONV, .ksize=3, .stride=1, .padding=1, .ch=1}, {0, CATS_SHORTCUT, .layer="a"}, {0, CATS_ACT_TANH}, {0, CATS_LINEAR}, {4, CATS_LOSS_MSE} };
	  check("shortcut(conv)", u, 7, batch, 0); }
	}
	printf(g_fail ? "FAILED: %d\n" : "all gradient checks passed\n", g_fail);
	return g_fail != 0;
}

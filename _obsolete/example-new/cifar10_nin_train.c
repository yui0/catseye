//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2019 Yuichiro Nakada
//---------------------------------------------------------

// gcc cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops -fopenmp -lgomp
// clang cifar10_nin_train.c -o cifar10_nin_train -lm -Ofast -march=native -funroll-loops

#define CATS_USE_FLOAT
#include "../catseye.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#define NAME	"cifar10_nin_train"
#define SIZE	32	// 69.7%(10)
//#define SIZE	96	// 92.0%(10)

#define CATS_USE_MOMENTUM_SGD
//#define CATS_USE_RMSPROP
//#define ETA	0.0001
//#define ETA	0.001
#define ETA	0.01
//#define ETA	0.1 // -nan

int main()
{
#if SIZE == 32
	int k = 32;		// image size
	int size = 32*32*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 10000;
#else
	int k = 96;		// image size
	int size = 96*96*3;	// 入力層
	int label = 10;	// 出力層
	int sample = 946-1;
#endif

	// Network in Network
	CatsEye_layer u[] = {
		//{  size, CATS_CONV, ETA, .ksize=11, .stride=4, .ch=192, .ich=3, .sx=k, .sy=k },
		{  size, CATS_CONV, ETA, .ksize=11, .stride=2, .ch=192, .ich=3, .sx=k, .sy=k },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=160 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=5, .stride=1, .ch=192/*, .padding=2*/ },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
//		{     0, CATS_AVGPOOL, .ksize=2, .stride=2 }, // k3,s2,p1
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 }, // k3,s2,p1

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=label },
		{     0, _CATS_ACT_RELU },

//		{     0, CATS_AVGPOOL, .ksize=3, .stride=1 }, // global average pooling
		{     0, CATS_LINEAR, ETA, .outputs=label },
		{     0, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};
	/*CatsEye_layer u[] = {	// 52.4%(10), 95.8%(1000), 99.7%(2000)
		{  size, CATS_CONV, ETA, .ksize=6, .stride=1, .ch=96, .ich=3, .sx=k, .sy=k },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96 },
//		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_AVGPOOL, .ksize=2, .stride=2 },

		{     0, CATS_CONV, ETA, .ksize=4, .stride=1, .ch=192 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
//		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },
		{     0, CATS_AVGPOOL, .ksize=2, .stride=2 },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=192 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=label },

		{     0, CATS_AVGPOOL, .ksize=3, .stride=2 }, // global average pooling
//		{     0, _CATS_ACT_SIGMOID },
		{     0, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};*/
	// https://github.com/jiecaoyu/pytorch-nin-cifar10/blob/master/original.py
	// https://github.com/tflearn/tflearn/blob/master/examples/images/network_in_network.py
	// https://github.com/BIGBALLON/cifar-10-cnn/blob/master/2_Network_in_Network/Network_in_Network_keras.py
	/*CatsEye_layer u[] = {	// 52.4%(10), 95.8%(1000), 99.7%(2000)
		{  size, CATS_CONV, ETA, .ksize=5, .stride=1, .ch=192, .padding=2, .ich=3, .sx=32, .sy=32 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=160 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=96 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=3, .stride=2 },

		{     0, CATS_CONV, ETA, .ksize=5, .stride=1, .ch=192, .padding=2 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=3, .stride=2 },
//		{     0, CATS_AVGPOOL, .ksize=3, .stride=2 },

		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=192, .padding=1 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=192 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=label },
		{     0, _CATS_ACT_RELU },

		{     0, CATS_AVGPOOL, .ksize=8, .stride=1 }, // global average pooling
//		{     0, _CATS_ACT_SIGMOID },
		{ label, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};*/
#if 0
	CatsEye_layer u[] = {	// 52.4%(10), 95.8%(1000), 99.7%(2000)
		/*{  size, CATS_CONV, ETA, .ksize=5, .stride=1, .ch=192, .padding=2, .ich=3, .sx=32, .sy=32 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=160 },
		{     0, _CATS_ACT_RELU },*/
		{  size, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=100, .ich=3 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },

/*		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=100, },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=100, },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .padding=1, .ch=100, },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_MAXPOOL, .ksize=2, .stride=2 },*/
		{     0, CATS_CONV, ETA, .ksize=3, .stride=1, .ch=50, .padding=1 },
		{     0, _CATS_ACT_RELU },
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=50 },
		{     0, _CATS_ACT_RELU },
		/*{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=50 },
		{     0, _CATS_ACT_RELU },

		{     0, CATS_LINEAR, 0.01, .outputs=256 },
		{     0, _CATS_ACT_RELU },

		{     0, CATS_LINEAR, 0.01, .outputs=label },*/
		{     0, CATS_CONV, ETA, .ksize=1, .stride=1, .ch=label },
		{     0, CATS_AVGPOOL, .ksize=16, .stride=1 }, // global average pooling
		{     0, _CATS_ACT_SOFTMAX },
		{ label, CATS_LOSS_0_1 },
	};
#endif
	CatsEye cat;
	_CatsEye__construct(&cat, u);

	// 訓練データの読み込み
	printf("Training data: loading...");
	int16_t *t;
#if SIZE == 32
	real *x = _CatsEye_loadCifar("../example/data_batch_1.bin", 32*32*3, 1, sample, &t);
#else
	real *x = _CatsEye_loadCifar("./animeface.bin", k*k*3, sizeof(int16_t), sample, &t);
#endif
	printf("OK\n");

	// 訓練
	printf("Starting training using (stochastic) gradient descent\n");
	_CatsEye_train(&cat, x, t, sample, 10/*repeat*/, 1000/*random batch*/, sample/10/*verify*/);
	printf("Training complete\n");

	// 結果の表示
	static int result[10][10];
	unsigned char *pixels = calloc(1, size*100);
	int c = 0;
	int r = 0;
	for (int i=0; i<sample; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);
		result[t[i]][p]++;
		if (p==t[i]) r++;
		else {
			if (c<100) {
				_CatsEye_visualize(x+size*i, k*k, k, &pixels[(c/10)*size*10+(c%10)*k*3], k*10, 3);
			}
			c++;
		}
	}
	for (int i=0; i<10; i++) {
		for (int j=0; j<10; j++) {
			printf("%3d ", result[i][j]);
		}
		printf("\n");
	}
	printf("Prediction accuracy on training data = %f%%\n", (float)r/sample*100.0);
	stbi_write_png(NAME"_wrong.png", k*10, k*10, 3/*bpp*/, pixels, 0);

	int n[10]; // 10 classes
	memset(n, 0, sizeof(int)*10);
	memset(pixels, 0, size*100);
	for (int i=0; i<10*10; i++) {
		int p = _CatsEye_predict(&cat, x+size*i);

		_CatsEye_visualize(x+size*i, k*k, k, &pixels[(p*k*k*10+(n[p]%10)*k)*3], k*10, 3);
		n[p]++;
	}
	stbi_write_png(NAME"_classify.png", k*10, k*10, 3, pixels, 0);

	memset(pixels, 0, size*100);
	/*for (int i=0; i<10*10; i++)*/ {
		int p = _CatsEye_predict(&cat, x/*+size*i*/);

		int x = 0;
		for (int n=0; n<cat.layers; n++) {
			CatsEye_layer *l = &cat.layer[n];
			if (l->type == CATS_LINEAR) {
				continue;
			}

			int mch = l->ch > 10 ? 10 : l->ch;
			for (int ch=0; ch<mch; ch++) {
//				CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pixels[n*(k+2) +ch*(k+2)*k*10], k*10);
				CatsEye_visualize(&l->z[ch*l->ox*l->oy], l->ox*l->oy, l->ox, &pixels[x +ch*(l->oy+2)*k*10], k*10);
			}
			x += l->ox+2;
		}
	}
	stbi_write_png(NAME"_predict.png", k*10, k*10, 1, pixels, 0);
	free(pixels);

	free(t);
	free(x);
	CatsEye__destruct(&cat);

	return 0;
}

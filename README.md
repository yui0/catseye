# Cat's Eye

Neural network library written in C and Javascript

## Features

- Lightweight and minimalistic:
  - Header only
  - Just include catseye.h and write your model in c. There is nothing to install.
  - Small dependency & simple implementation
- Fast: [under construction]
  - OpenCL support (GPGPU)
  - OpenGL support (GPGPU)
  - ~~SSE, AVX support (But gcc and clang support SIMD...)~~
  - ~~OpenMP support~~
  - ~~Support half precision floats (16bit)~~
- Supported networks:
  - Activation functions
    - sigmoid
    - softmax
    - tanh, scaled tanh (1.7519 * tanh(2/3x))
    - ReLU, Leaky ReLU, ELU, RReLU
    - abs
    - identity
  - Loss functions
    - cross-entropy, mean-squared-error
  - Optimization algorithms
    - SGD (stochastic gradient descent) with/without L2 normalization
    - Momentum SGD
    - AdaGrad
    - RMSProp
    - Adam
  - Layer types
    - linear (mlp)
    - convolution
    - deconvolution
    - Sub-Pixel Convolution (Pixel Shuffler)
    - max pooling
    - average pooling
    - global average pooling (GAP)
    - batch normalization
    - concat
    - shortcut
- Loader formats:
  - PNG
  - cifar [https://www.cs.toronto.edu/~kriz/cifar.html]
  - MNIST

## Usage

Just include header files in your project.

for more information, see example/

```bash
$ dnf install ghostscript ocl-icd-devel
$ cd example
$ make
$ ./sin
```

## Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://raw.githubusercontent.com/yui0/catseye/master/catseye.ipynb)

- Recognizing handwritten digits by MNIST training ([example/mnist_train.c](example/mnist_train.c))
  - http://yui0.github.io/catseye/example/html/mnist.html

- Recognizing pictures ([example/cifar10_train.c](example/cifar10_train.c))
  - http://yui0.github.io/catseye/example/html/cifar10.html

- Neural Network 'paints' an image ([example/paint.c](example/paint.c))

  ![Sakura](example/paint_sakura.png)
  [![Sakura](example/paint_sakura0499.png)](http://www.youtube.com/watch?v=445ilzeKtto)

  [![CCSakura](example/paint_ccsakura0149.png)](http://www.youtube.com/watch?v=CnZ-z2C64_8)
  [![Aika](example/paint_aika0499.png)](http://www.youtube.com/watch?v=Q6ylERYqoWE)

  ![Nyanko](example/paint_cat.png)
  [![Nyanko](example/paint_cat0499.png)](http://www.youtube.com/watch?v=qy_R2gp5rx0)

- Function approximation ([example/sin.c](example/sin.c))

  ![sin](example/sin.png)
  ![quadratic function](example/quadratic.png)

- Convolution Autoencoder ([example/mnist_cnn_autoencoder.c](example/mnist_cnn_autoencoder.c),[example/cifar_autoencoder.c](example/cifar_autoencoder.c))

  ![autoencoder](example/mnist_cnn_autoencoder.png)
  ![autoencoder](example/mnist_cnn_autoencoder.svg)
  ![autoencoder](example/cifar_autoencoder.png)

- DCGAN ([example/mnist_lsgan.c](example/mnist_lsgan.c))
  - epoch 1900

  ![dcgan](example/mnist_lsgan.png)

- Autoencoder ([example/mnist_autoencoder.c](example/mnist_autoencoder.c),[example/mnist_autoencoder2.c](example/mnist_autoencoder2.c))

  ![Autoencoder](example/mnist_autoencoder.png "Autoencoder")
  ![Autoencoder](example/mnist_autoencoder2.png "Autoencoder")
  ![Autoencoder](example/mnist_autoencoder2.svg "Autoencoder")

- VanillaGAN

  ![epoch=41](example/mnist_vgan_00041.png "epoch=41")
  ![GIF](example/mnist_vgan.gif "GIF")

- Convolutional Neural Networks (example/mnist_cnn_train.c)
  - tanh, 7x7, 32ch, 99.2%

  ![99.2%](example/mnist_cnn_train_32ch_k7.png "Convolutional")


## Question

- Neural Network Always Produces Same/Similar Outputs for Any Input
  - Scale down the problem to manageable size.
  - Make sure you have enough hidden units.
  - Change the activation function and its parameters.
  - Change learning algorithm parameters.


## Refrences

- Documents
  - Neural Networks and Deep Learning [http://nnadl-ja.github.io/nnadl_site_ja/chap1.html]
  - Explain easy backpropagation in the universe [https://www.yukisako.xyz/entry/backpropagation]
  - Optimization algorithm with super easy explanation [https://qiita.com/omiita/items/1735c1d048fe5f611f80]
  - Basic parts of calculation graph used for backpropagation method, etc. [https://qiita.com/t-tkd3a/items/031c0a4dbf25fd2866a3]
  - Automatic differentiation [https://tech-lab.sios.jp/archives/21072]
  - Machine learning [http://hokuts.com/category/%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/]
  - tiny-cnn [https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88]

  - CS231n Convolutional Neural Networks for Visual Recognition [http://cs231n.github.io/neural-networks-3/#anneal]
  - SVM [http://d.hatena.ne.jp/echizen_tm/20110627/1309188711]
  - Autoencoder
    - Summary of research on VAE [https://www.hiro877.com/entry/vae-research]
    - Hello Autoencoder [https://kiyukuta.github.io/2013/08/20/hello_autoencoder.html]
    - Autoencoder [https://pc.atsuhiro-me.net/entry/2015/08/18/003402]
    - Autoencoder [https://www.slideshare.net/at_grandpa/chapter5-50042838]
  - Convolutional Neural Networks [http://blog.yusugomori.com/post/129688163130/%E6%95%B0%E5%BC%8F%E3%81%A7%E6%9B%B8%E3%81%8D%E4%B8%8B%E3%81%99-convolutional-neural-networks-cnn]
  - Backpropagation [http://postd.cc/2015-08-backprop/]
  - Perceptron [http://tkengo.github.io/blog/2015/08/21/visual-perceptron/]
- Programing
  - Multilayer perceptron [http://kivantium.hateblo.jp/entry/2014/12/22/004640]
  - Weather example [http://arakilab.media.eng.hokudai.ac.jp/~t_ogawa/wiki/index.php?LibSVM]
  - Recognizing handwritten digits [http://aidiary.hatenablog.com/entry/20140201/1391218771]
  - Recognizing handwritten digits on Web [http://d.hatena.ne.jp/sugyan/20151124/1448292129]
  - Image generator by Denoising Autoencoder [http://joisino.hatenablog.com/entry/2015/09/09/224157]
  - Neural Network 'paints' an image [http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html]

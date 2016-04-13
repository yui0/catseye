# Cat's Eye
Neural network library written in C and Javascript

## Features
- Lightweight and minimalistic
  - Header only
  - Just include catseye.h and write your model in c. There is nothing to install.
  - Small dependency & simple implementation
- Support Deep Learning
  - Multilayer perceptron (MLP)
  - Deep Neural Networks (DNN)
  - Convolutional Neural Networks (CNN)
  - Network in Network (NIN)
- Supported networks
  - Activation functions
    - sigmoid, tanh, scaled tanh, ReLU, Leaky ReLU, ELU, abs, identity function
  - Loss functions
    - cross-entropy, mean-squared-error
  - Optimization algorithms
    - stochastic gradient descent (with/without L2 normalization and momentum)
  - Layer types
    - linear layer
    - convolutional layer
    - CCCP, Cascaded Cross Channel Parametric Pooling layer
    - max pooling layer

## Usage
just include header files in your project

for more information, see example/

## Demo
- Recognizing handwritten digits by MNIST training (example/mnist_train.c)
  - http://yui0.github.io/catseye/example/html/mnist.html

- Neural Network 'paints' an image (example/paint.c)

  ![Sakura](example/paint_sakura.png)
  [![Sakura](http://img.youtube.com/vi/445ilzeKtto/0.jpg)](http://www.youtube.com/watch?v=445ilzeKtto)

- Function approximation (example/sin.c)

  ![sin](example/sin.png)
  ![quadratic function](example/quadratic.png)

- Autoencoder (example/mnist_autoencoder.c)
  - Unit 64 [tied weight]

  ![epoch=100](example/mnist_autoencoder_u64ae_s100.png "epoch=100")
  ![epoch=500](example/mnist_autoencoder_u64ae_s500.png "epoch=500")
  ![epoch=1500](example/mnist_autoencoder_u64ae_s1500.png "epoch=1500")

  ![epoch=100](example/mnist_autoencoder_weights_u64ae_s100.png "epoch=100")
  ![epoch=500](example/mnist_autoencoder_weights_u64ae_s500.png "epoch=500")
  ![epoch=1500](example/mnist_autoencoder_weights_u64ae_s1500.png "epoch=1500")

  - Unit 64

  ![epoch=100](example/mnist_autoencoder_u64_s100.png "epoch=100")
  ![epoch=500](example/mnist_autoencoder_u64_s500.png "epoch=500")
  ![epoch=1500](example/mnist_autoencoder_u64_s1500.png "epoch=1500")

  ![epoch=100](example/mnist_autoencoder_weights_u64_s100.png "epoch=100")
  ![epoch=500](example/mnist_autoencoder_weights_u64_s500.png "epoch=500")
  ![epoch=1500](example/mnist_autoencoder_weights_u64_s1500.png "epoch=1500")

  - Unit 16

  ![epoch=100](example/mnist_autoencoder_u16_s100.png "epoch=100")
  ![epoch=500](example/mnist_autoencoder_u16_s500.png "epoch=500")
  ![epoch=1500](example/mnist_autoencoder_u16_s1500.png "epoch=1500")

- Denoising Autoencoder
  - Unit 64

  ![epoch=100](example/mnist_autoencoder_u64da_s100.png "epoch=100")
  ![epoch=100](example/mnist_autoencoder_weights_u64da_s100.png "epoch=100")

- Convolutional Neural Networks (example/mnist_cnn_train.c)
  - tanh, 7x7, 32ch, 99.2%

  ![99.2%](example/mnist_cnn_train_32ch_k7.png "Convolutional")

## Refrence
- Documents
  - Neural Networks and Deep Learning [http://nnadl-ja.github.io/nnadl_site_ja/chap1.html]
  - Machine learning [http://hokuts.com/category/%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/]
  - CS231n Convolutional Neural Networks for Visual Recognition [http://cs231n.github.io/neural-networks-3/#anneal]
  - SVM [http://d.hatena.ne.jp/echizen_tm/20110627/1309188711]
  - Hello Autoencoder [http://kiyukuta.github.io/2013/08/20/hello_autoencoder.html]
  - Autoencoder [http://pc.atsuhiro-me.net/entry/2015/08/18/003402]
  - Autoencoder [http://www.slideshare.net/at_grandpa/chapter5-50042838]
  - Convolutional Neural Networks [http://blog.yusugomori.com/post/129688163130/%E6%95%B0%E5%BC%8F%E3%81%A7%E6%9B%B8%E3%81%8D%E4%B8%8B%E3%81%99-convolutional-neural-networks-cnn]
  - tiny-cnn [https://github.com/nyanp/tiny-cnn/wiki/%E5%AE%9F%E8%A3%85%E3%83%8E%E3%83%BC%E3%83%88]
  - Backpropagation [http://postd.cc/2015-08-backprop/]
  - Perceptron [http://tkengo.github.io/blog/2015/08/21/visual-perceptron/]
- Programing
  - Multilayer perceptron [http://kivantium.hateblo.jp/entry/2014/12/22/004640]
  - Weather example [http://arakilab.media.eng.hokudai.ac.jp/~t_ogawa/wiki/index.php?LibSVM]
  - Recognizing handwritten digits [http://aidiary.hatenablog.com/entry/20140201/1391218771]
  - Recognizing handwritten digits on Web [http://d.hatena.ne.jp/sugyan/20151124/1448292129]
  - Image generator by Denoising Autoencoder [http://joisino.hatenablog.com/entry/2015/09/09/224157]
  - Neural Network 'paints' an image [http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html]

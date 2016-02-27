# Cat's Eye
Neural network library written in C and Javascript

# Usage
just include header files in your project

for more information, see example/

# Demo
- Recognizing handwritten digits by MNIST training
  - http://yui0.github.io/catseye/example/html/index.html

- Function approximation

  ![sin](example/sin.png)
  ![quadratic function](example/quadratic.png)

- Autoencoder
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

# Refrence
- Neural Networks and Deep Learning [http://nnadl-ja.github.io/nnadl_site_ja/chap1.html]
- Machine learning [http://hokuts.com/category/%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/]
- CS231n Convolutional Neural Networks for Visual Recognition [http://cs231n.github.io/neural-networks-3/#anneal]

- Multilayer perceptron [http://kivantium.hateblo.jp/entry/2014/12/22/004640]
- Weather example [http://arakilab.media.eng.hokudai.ac.jp/~t_ogawa/wiki/index.php?LibSVM]
- Recognizing handwritten digits [http://aidiary.hatenablog.com/entry/20140201/1391218771]
- Recognizing handwritten digits on Web [http://d.hatena.ne.jp/sugyan/20151124/1448292129]
- Autoencoder [http://pc.atsuhiro-me.net/entry/2015/08/18/003402]

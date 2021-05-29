# Example

## Sample data
* example_01.csv
  * two dimensional three class classification
  * first column shows label, second and third column show data
  * borrowed from https://github.com/kivantium/libnn
* weather_sapporo.csv
  * the weather in sapporo from 2008-04-01 to 2014-03-31
  * borrowed from http://www.data.jma.go.jp/gmd/risk/obsdl/
* digits.csv
  * borrowed from https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/datasets/data
* train-images-idx3-ubyte and train-labels-idx1-ubyte
  * borrowed from http://yann.lecun.com/exdb/mnist/
  * ref. http://y-uti.hatenablog.jp/entry/2014/07/23/074845

## Sample Program
* digits_autoencoder.c
  * autoencoder sample
* digits_train.c and digits_test.c
  * classify numbers image by using multi layer perceptron
* example\_01.c
  * classify sample data by using multi layer perceptron
  * as code is heavily commented, you can understand how it works
* mnist_autoencoder.c
  * autoencoder sample
* mnist_train.c and mnist_test.c
  * classify numbers image by using multi layer perceptron
* quadratic.c
  * function approximation by using multi layer perceptron
* sin.c
  * function approximation by using multi layer perceptron
* weather.c
  * classify weather data by using multi layer perceptron

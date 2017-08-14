# 七、在TensorFlow中进行卷积
> 内容由Chip Huye编写，Danijar Hafner校对
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_07.pdf)

原文档尚未完成，故较短。

@ Last class, we had a wonderful guest lecture by Justin Johnson. I hope that by now, you’re more or less familiar with common layers of a convolutional neural networks. Today, we will make a (crude) attempt to understand convolution and what kind of support TensorFlow has for convolutions.
上节课，我们邀请Justin Johnson进行了一场精彩的讲座。我希望到目前为止，你或多或少熟悉卷积神经网络的普通层。今天，我们将做一个(粗略的)尝试去理解卷积，以及TensorFlow是怎样支持卷积的。

## 理解卷积
@ If you’ve taken maths or physics, you’re probably familiar with this term. Oxford dictionary define the mathematical term convolution as:
如果你学过数学或物理，你可能很熟悉这个术语。牛津词典将数学术语卷积定义为:

@ a function derived from two given functions by integration that expresses how the shape of one  is  modified  by  the other.
> 一个函数由两个给定的函数通过积分来表示一个函数的形状是如何被另一个函数所改变的。

@ And that’s pretty much what convolution means in the neural networks setting. Convolution is how the original input (in the first convolutional layer, it’s part of the original image) is modified by the kernel (or filter). To better understand convolutions, you can refer to this  wonderful blog post  by Chris Olah at Google Brain.
这就是卷积在神经网络中所表示的意思。卷积是经原始输入(在第一个卷积层，它是原始图像的一部分)由卷积核(或滤波器)进行卷积。为了更好地理解卷积，你可以参考Chris Olah在谷歌Brain中的这篇精彩的[博客文章](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/1.png)

[在维基百科上查看动图](https://en.wikipedia.org/wiki/Convolution)

@ TensorFlow has great support for convolutional layers. The most popular one is tf.nn.conv2d.
TensorFlow对卷积层有很好的支持。最有名的是`tf.nn.conv2d`.

```python
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu = None, data_format = None, name = None)
```
- Input: Batch size x Height x Width x Channels
- Filter: Height x Width x Input Channels x Output Channels (e.g. [5, 5, 3, 64])
- Strides: 4 element 1-D tensor, strides in each direction(often [1, 1, 1, 1] or [1, 2, 2, 1])
- Padding: 'SAME' or 'VALID' 
- Data_format: default to NHWC
(不是我忘了翻译，我真没看懂)

@ Generally, for strides, you don’t want to use any number other than 1 in the first and the fourth dimension,
一般来说，对于strides，不建议在第一个和第四个维度中使用任何数字，

@ Because of this property of convolution, we can do convolutions without training anything. We can simply choose a kernel and see how that kernel affects our image. For example, the kernel often used for blurring an image is as below. You can do the element-wise multiplication to see how this kernel helps blurring an image.
因为卷积的性质，我们可以对图像进行卷积而不需要任何训练。我们可以简单地选择一个卷积核，看看卷积核如何影响我们的图像。例如，通常用于模糊图像的卷积核如下所示。您可以使用元素的乘法来查看卷积核是如何帮助模糊图像的。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/2.png)

@ Just as a fun exercise, you can see several other popular kernels in the kernels.py file on the class GitHub repository, and see how to use them in 07_basic_filters.py
作为一个有趣的练习，你可以在本课程在GitHub上的资源中的kernels.py中看到其他几个流行的卷积核，并在07_basic_filters.py中学习使用它们。

@ let’s see what the Blurring kernel does to this super cute image of a Chihuahua.
让我们看看模糊核是对这张图片作出怎样的操作的。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/3.png)

@ There are also several other built-in convolutional operations. Please refer to t  he official documentation  for more information.
还有其他一些内置的卷积操作。更多有关信息，请参阅[官方文档](https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/convolution)。

@ conv2d :   Arbitrary  filters that can mix channels together.
@ depthwise_conv2d :   Filters  that operate on each channel independently. 
@ separable_conv2d :  A depthwise spatial filter followed  by  a pointwise filter.

> conv2d： 任意过滤器，可以将通道组合在一起。
> depthwise_conv2d： 在每个通道上独立运行的过滤器。
> separable_conv2d： 一个深度的空间过滤器，然后是一个pointwise过滤器。

@ In this case, we hard code our kernels. When training a convnet, we don’t know what the values for our kernels and therefore have to figure them out by learning them. We’ll go through the process of learning the kernels through a simple convnet with our old friend MNIST.
在这种情况下，我们硬编码了我们的内核。当我们训练一个卷积神经网络时，我们不知道我们的卷积核是什么值，因此我们必须通过学习来找出它们的价值。我们将通过与老朋友MNIST的一个简单的卷积来学习确定卷积核的过程。

## 对Mnist进行卷积
@ We’ve done logistic regression on MNIST and the result is abysmal. Since MNIST dataset contains of just images, let’s see how much better we can do with convnet on MNIST.
我们已经对MNIST进行了逻辑回归，结果非常糟糕。由于MNIST数据集包含的是图像，让我们看看在MNIST上，我们是否能做的更好。

@ For MNIST, we will be using two convolutional layers, each followed by a relu and a maxpool layers, and one fully connected layer. See the example on GitHub under the name 07_convnet_mnist.py
对于MNIST，我们将使用两个卷积层，分别是relu和maxpool层，以及一个完全连接的层。请参阅GitHub上的示例，名为07_convnet_mnist.py。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/4.png)

## 变量作用域
@ Since we’ll be dealing with multiple layers, it’s important to introduce variable scope. Think of a variable scope something similar to a namespace. A variable name ‘weights’ in variable scope ‘conv1’ will become ‘conv1-weights’. The common practice is to create a variable scope for each layer, so that if you have variable ‘weights’ in both convolution layer 1 and convolution layer 2, there won’t be any name clash.
由于我们将处理多个卷积层，所以引入变量范围是很重要的。考虑一个类似于名称空间的变量范围。在变量范围的“conv1”中，变量名“权重”将变成“卷积权重”。通常的做法是为每一层创建一个可变的范围，这样如果在卷积层1和卷积层2中都有可变的“权重”，就不会有任何名称冲突。

@ In variable scope, we don’t create variable using tf.Variable, but instead use tf.get_variable()
在变量范围中，我们不使用`tf.Variable`来创建变量，而是使用`tf.get_variable()`来创建变量。

```python
tf.get_variable(<name>, <shape>, <initializer>)
```

@ If a variable with that name already exists in that variable scope, we use that variable. If a variable with that name doesn’t already exists in that variable scope, TensorFlow creates a new variable. This setup makes it really easy to share variables across architecture. This will come in extremely handy when you build complex models and you need to share large sets of variables. Variable scopes help you initialize all of them in one place.
如果一个变量在那个变量范围中已经存在，那么TensorFlow就会使用那个变量。如果一个变量在变量范围内不存在，那么TensorFlow就会创建一个新的变量。这种设置使得跨架构共享变量变得非常容易。当您构建复杂的模型并需要共享大量的变量时，这将非常方便。变量作用域帮助您在一个地方初始化它们。

@ Nodes in the same variable scope will be grouped together, and therefore you don’t have to use name scope any more. To declare a variable scope, you do it the same way you do name scope:
相同变量作用域中的节点将被组合在一起，因此您不再需要使用名称范围。要声明一个变量范围，您可以用同样的方式来命名范围:
```python
with tf.variable_scope('conv1') as scope:
```
例如：
```python
with tf.variable_scope('conv1') as scope:
    w = tf.get_variable('weights', [5, 5, 1, 32])
    b = tf.get_variable('biases', [32], initializer = tf.random_normal_initializer()) 
    conv = tf.nn.conv2d(images, w, strides = [1, 1, 1, 1], padding = 'SAME')
    conv1 = tf.nn.relu(conv + b, name = scope.name)

with tf.variable_scope('conv2') as scope:
    w = tf.get_variable('weights', [5, 5, 32, 64])
    b = tf.get_variable('biases', [64], initializer = tf.random_normal_initializer()) 
    conv = tf.nn.conv2d(conv1, w, strides = [1, 1, 1, 1], padding = 'SAME')
    conv2 = tf.nn.relu(conv + b, name = scope.name)
```

@ You see that with variable scope, we now have neatly block of code that can be broken into smaller function for reusability.
您可以看到，在变量范围内，我们现在有了整洁的代码块，这些代码可以被分解为更小的函数，以实现可重用性。

@ For more information on variable scope, please refer to t  he official documentation. 
有关可变范围的更多信息，请参阅[官方文档](https://www.tensorflow.org/how_tos/variable_scope/)。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/5.png)

下面是训练结果：
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note7/markdown/img/note7/6.png)

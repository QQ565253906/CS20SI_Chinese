# 一、介绍tensorflow
> 内容由Chip Huye编写，Danijar Hafner, Jon Gautier, Minh-Thang Luong, Paul Warren校对
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_01.pdf)

## 概述

《TensorFlow for Machine Intelligence》的作者对Tensorflow进行了一个较为贴切的描述，我在此对其进行引用。

“尽管深度学习背后的数学理论已经存在了数十年，但是用于创建和训练深度学习模型的框架在近几年才得以实现。

不幸的是，大多数的深度学习框架无法做到灵活性和生产价值两全。灵活性的框架对于新的深度模型的构建的价值是不可估量的，但是这种框架不是太慢就是不能够用于实际生产。另一方面，高效、快速、可以进行分布式运算的深度学习框架往往只能处理特定的几种深度学习模型，不能满足研究和开发新模型的需求。这使得开发者面临着一个两难的困境，究竟应该使用灵活性高却难以实用的深度学习框架，还是应该使用一个完全不同的框架，用于生产？如果我们选择前者，我们有可能不能检验不同类型的神经网络模型；如果我们选择后者，我们不得不修改我们的代码以使其能够适应完全不同的API接口。**我们对于这件事情还有资源吗？**

TensorFlow致力于解决这一问题。“

## 什么是TensorFlow？

TensorFlow最初由Google公司编写，作为一个内部的机器学习工具。TensorFlow在2015年十一月份宣布开源，并遵守Apache 2.0开源协议。为了更深入的了解什么是TensorFlow，我们来看一看它的作者是如何描述它的。

在TensorFlow的官网上，我们可以看到：

口号：
&emsp &emsp 一个用于机器学习的开源代码库。
定义：
&emsp &emsp TensorFlow是一种基于数据流图的用于进行数据计算的一个开源的软件库。

我们可以看到TensorFlow是开源的。然而，需要注意的是，只有其发布在GitHub上的项目是开源的，Google内部还依旧独立维护着另一版本的TensorFlow。据说谷歌这样做是因为TensorFlow与其内部的其他工具有很复杂的关联，并不是因为谷歌公司故意不愿发布好的东西。我们希望这是真的。

另一个重点是，我们可以看到TensorFlow是一个实现“机器智能的代码库”。它是由大公司开发的十多个机器智能库中的一个，很有可能是最新的一个。了解当前的深度学习库，请访问[链接](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)。

## 为什么使用TensorFlow

考虑到有如此多的开源库，为什么我们选择Tensorflow来教授这门课呢?

我们的第一个原因是图书馆应较为受欢迎。根据我们在斯坦福大学的同行们的观察，在这些名单中，四大最受欢迎的机器学习框架是“Theano”、“Torch”和“TensorFlow”。

Torch框架是在Lua写的，这是一种很好的框架，但不幸的是，我们对它并不熟悉。在游戏开发社区和深度学习社区之外，它也不是很受欢迎。

对于Theano来说，《Fundamentals of Deep Learning》对Tensorflow和Theano做了一个很好的对比:

“首先，Theano有一个额外的“图表编译”步骤，在建立某种类型的深度学习体系时，花了大量的时间。虽然与培训时间相比，它的规模很小，但在编写和调试新代码时，这一编译阶段却令人沮丧。

其次，与Theano相比，TensorFlow有一个更简洁的界面。许多类型的模型可以用更少的行表达，而不牺牲框架的表达能力。

最后，TensorFlow是兼顾了生产使用，而Theano是由研究人员设计的，几乎纯粹是为了研究目的。

所以，TensorFlow有许多的开箱即用的特性和使它更好的选择适合实际系统(能够运行在移动环境中,轻松地构建模型,跨越多个gpu在单个机器,和使用分布式的方式训练大规模网络模型)。”

由于这些原因，TensorFlow成为了我们的选择。一言以蔽之，我们选择了TensorFlow，是因为
- Python API
- 可移植性: 在桌面、服务器或移动设备上都使用相同的API将计算部署到一个或多个cpu或gpu上
- 灵活性:从树莓派、Android、Windows、iOS、Linux到服务器机房
- 可视化(TensorBoard真的是棒极了)
- 检查点(用于管理实验)
- 自动求微分(不需要手动求导)
- 大型社区(一年内有超过一万次提交和超过3000次与tf相关的repos)
- 有很多很棒的项目已经使用了TensorFlow

**TensorFlow是新生事物，但它已经在工业和研究领域获得很大了发展。在293名报名参加课程CS 20SI课程的学生中，深度学习框架的使用情况出现了问题。这并不意味着学生们以前从未使用过任何深度学习图书馆。**

#### 使用TensorFlow的公司
- Google
- DeepMind
- OpenAI
- Snapchat
- Uber
- Airbus
- eBay
- Dropbox
- A bunch of startups

#### 一些使用Tensorflow的酷项目
- DeepMind’s WaveNet Text 的语音合成系统
- 谷歌大脑的品红项目利用机器学习创造出引人注目的艺术和音乐
- 神经风格翻译
- 谷歌翻译的重大改进

#### 2016年，来自[谷歌研究博客](https://opensource.googleblog.com/2016/11/celebrating-tensorflows-first-year.html)的更多TensorFlow项目的例子。
- 澳大利亚海洋生物学家利用TensorFlow在成千上万的高分辨率照片中寻找海牛，以更好地了解它们的种群，以及它们正面临灭绝的威胁。
- 一位有进取心的日本黄瓜农民训练了一种带有TensorFlow的模型，根据大小、形状和其他特征对黄瓜进行分类。
- 放射学家已经在医学扫描中调整了TensorFlow来识别帕金森病的迹象。
- 海湾地区的数据科学家们已经安装了TensorFlow和树莓派来跟踪Caltrain。

着难道不是很酷么？

## 开始使用TensorFlow

这部分很酷，但不那么重要。你可以跳过。

### 1. TF Learn（tf.contrib.learn）
TensorFlow有一个简化的接口，TF Learn(tensorflow.contrib.learn)，它提供了用户可以简单调用的现成的模型。这是为了模仿scikit-learn，“为了顺利从线性学习的机器学习过渡到可以构建不同类型的机器学习模型的更加开放的世界。”事实上，TF学习最初是一个叫做Scikit Flow(SKFlow)的独立项目。

TF学习允许你加载数据、构造模型、训练模型、评估精度，这些都有其自己的单独操作。在TF中直接可以调用一些模型，包括线性分类器、线性回归器、深度神经网络分类器等。Google也提供了关于如何使用TF学习构建定制模型的很好的教程。

TF Learn文档中对深度神经网络分类器的介绍:使用tensorflow的一个例子。

iris.py
```python
# Load dataset.
# 载入数据
iris = tf.contrib.learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = cross_validation.train_test_split( \
    iris.data, iris.target, test_size=0.2, random_state=42)

# Build 3 layer DNN with 10, 20, 10 units respectively.
# 建立一个三层的深度神经网络，分别每层有10，20，10个神经元
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier = tf.contrib.learn.DNNClassifier( \
    feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

# Fit and predict.
# 训练与预测
classifier.fit(x_train, y_train, steps=200)
predictions = list(classifier.predict(x_test, as_iterable=True)) 
score = metrics.accuracy_score(y_test, predictions)

print('Accuracy: {0:f}'.format(score))
```

你可以看到，通过使用TF，你只需要一行就可以装载数据，在另一行中分割数据，你可以调用内置的深层神经网络分类器并自己设定隐藏单元的数量。在上面的代码中，第一层有10个隐藏单元，第二个层有20个隐藏单元，第三层有10个隐藏单元。您还可以指定标签类的数量、优化的类型(例如，梯度下降)，或者初始权重。

你可以访问TF学习示例库，了解更多关于TF学习的例子。但是，请记住，大多数内置的模型都是使用弃用的函数实现的，所以如果你调用它们，你会看到许多警告。

### 2. TF-Slim (tf.contrib.slim)
另一个简单的API叫做tf-slim，它可以简化构建、训练和评估神经网络。

### 3.TensorFlow高级api
在TensorFlow上面构建了许多高级api。一些最流行的api包括Keras(keras@GitHub)、TFLearn(tflearn@GitHub)和pretty tensor(prettytensor@GitHub)。

注意:您不应该将高级API TFLearn（没有空格）与简化的接口TF Learn混淆。TFLearn支持大多数最近的深度学习模式，如:卷积、LSTM、BiRNN、ResNets、生成式对抗网络以及诸如BatchNorm、PReLU等。TFLearn是由Aymeric Damien开发的。

剧透:Aymeric将会在几周后做一场演讲。

然而，TensorFlow的主要目的不是提供开箱即用的机器学习解决方案。相反，TensorFlow提供了一套广泛的功能和类，它允许用户从头定义模型。虽然着相比之下要复杂得多，但是却也提供了更多的灵活性。您可以通过TensorFlow构建各种神经网络。

## 数据流图
下一个重点是数据流图。TF通过数据流图来完成它所有的计算。

请参阅第一堂课的幻灯片(第23页)，了解更多详情！
http://danijar.com/what-is-a-tensorflow-session/

会话还将分配内存以存储该变量的当前值。

正如您所看到的，变量的值仅在一个会话中有效。如果我们在第二次会话中尝试查询该值，则TensorFlow将会产生一个错误，因为该变量没有在那里初始化。

## 参考
https://www.tensorflow.org/
《Tensorflow for Machine Intelligence》
《Hands-On Machine Learning with Scikit-Learn and TensorFlow》. 第九章: Up and running with TensorFlow
《Fundamentals of Deep Learning》. 第三章: Implementing Neural Networks in TensorFlow.
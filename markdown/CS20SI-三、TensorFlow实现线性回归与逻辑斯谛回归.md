# 三、TensorFlow实现线性回归与逻辑斯谛回归
> 内容由Chip Huye编写，Danijar Hafner校对
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_03.pdf)

在前两节课中我们已经学到了很多关于TensorFlow的内容。这是前两节课的关键概念。如果你不清楚其中的任何一个，你应该回到之前的两节课重新落实。
- 图与会话
- 操作符：常量、变量、函数？？？
- TensorBoard
- 懒操作

我们已经讨论了TensorFlow的基本原理。是的,我们这快!让我们把它们放在一起看看我们能做些什么。

## TensorFlow实现线性回归
让我们从一个简单的线性回归例子开始。我希望你们都已经对线性回归很熟悉了。如果没有，你可以在[维基百科](https://en.wikipedia.org/wiki/Linear_regression)上读到。

问题:我们经常听到保险公司使用诸如火灾和盗窃之类的因素来计算社区的危险程度。我的问题是:这是多余的吗?在一个社区里，火灾和盗窃的数量是否有关系，如果有的话，我们能找到它吗?

换句话说，我们能否找到一个函数f如果X是火灾的数量而Y是偷窃的数量，那么Y=f(X)?

考虑到这段关系，如果我们有一个特定区域的发生火灾的次数，我们能预测那个地区的被盗窃的盗窃次数。

我们有由美国公民权利委员会收集的数据，这是由[Cengage Learning](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr05.html)提供的。

**数据集描述:**
- 名称:芝加哥的火灾和盗窃
- X = 每1000个单位住宅发生火灾的次数
- Y = 每1000人中有偷窃行为的人数
- 在芝加哥市区的同一区域内
- 邮政编码区域总数:42

**解决方案:**
首先，假设火灾和盗窃的数量之间的关系是线性的:Y=wX+b

我们需要找到标量参数w和b，用均值平方误差作为损失函数。

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE  =   "data/fire_theft.xls"

# Step 1: 从.xls文件中读入数据
book = xlrd.open_workbook(DATA_FILE, encoding_override = "utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)]) 
n_samples = sheet.nrows - 1

# Step 2: 分配给X(火灾数量)和Y(小偷数量)的占位符
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

# Step 3: 创建权重和偏差值，全部初始化为零
w = tf.Variable(0.0, name = "weights") 
b = tf.Variable(0.0, name = "bias")

# Step 4: 构建预测结果
Y_predicted = X * w + b

# Step 5: 使用平方误差作为损失函数
loss = tf.square(Y - Y_predicted, name = "loss")

# Step 6: 使用梯度下降法，学习速率0.01以最小化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
with tf.Session() as sess:

# Step 7: 初始化所有变量，在这里是w和b。
    sess.run(tf.global_variables_initializer())

       # Step 8: 训练模型
    for i in range(100):   # 训练100次 
        for x, y in data:
                    # 运行会话中optimizer以最小化损失函数
            sess.run(optimizer, feed_dict = {X: x, Y: y})

    # Step 9: 输出w和b的值。
    w_value, b_value = sess.run([w, b])
```

经过100次的培训后，我们的平均损失为1372.77701716，而w=1.62071，b=16.9162。损失相当大。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/1.png)

它并不满足。我们能更好地利用二次函数Y=wXX+uX+b吗?
让我们试一试。我们只需要添加另一个变量b，然后改变y预测值的公式。

```python
# Step 3: 创建变量weights_1, weights_2, bias。并全部置零。
w = tf.Variable(0.0, name = "weights_1") 
u = tf.Variable(0.0, name = "weights_2") 
b = tf.Variable(0.0, name = "bias")

# Step 4: 预测
Y_predicted = X * X * w + X * u + b 
# Step 5: Profit!
```

在10次迭代之后，我们的平均损失是797.335975976，有w, u, b = [0.071343 0.010234 0.00143057]

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/2.png)

这比线性函数收敛的时间要少，但由于右侧的几个异常值，仍然不能完全拟合原始数据。如果使用[Huber loss](https://en.wikipedia.org/wiki/Huber_loss)，而不是MSE或第三次多项式作为函数f，可能可以获得更好的效果，你可以在家里尝试。

使用Huber的二次模型，我得到了一些更好的模型，忽略了异常值:
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/3.png)

**怎么才能知道我的模型是正确的？**
如果你不知道r的平方是什么，Minitab有一个很好的[博客文章](http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)来解释它。下面是它的要点:
r-平方是一种统计数据，用来衡量数据与拟合回归线的距离。
它也被称为确定系数，或者多重回归的多重确定系数。
r平方的定义是相当直接的;它是由线性模型解释的响应变量变化的百分比。(it is the percentage of the response variable variation that is explained by a linear model.) ???
r平方=解释变异/总变异

在数据集上尝试运行一下。
---
我们在机器学习课上学到的是，这一切都归结于验证和测试。因此，第一个方法显然是在测试集上测试我们的模型。

为训练、验证和测试提供单独的数据集是非常棒的，但是这意味着我们将有更少的训练数据。有很多文献可以帮助我们绕过那些没有大量数据的问题，比如k-折叠交叉验证。

用虚拟数据测试我们的模型
---
我们可以测试模型的另一种方法是对虚拟数据进行测试。例如，在这种情况下，我们可以创建一些虚拟数据，这些数据的线性关系已经知道了，可以测试我们的模型。在这种情况下，让我们创建100个数据点(X，Y)这样的Y 3 X，看看我们的模型输出w=3 b=0。

产生虚拟数据
```python
# each value y is approximately linear but with some random noise
X_input = np.linspace(-1, 1, 100)
Y_input = X_input * 3 + np.random.randn(X_input.shape[0]) * 0.5
```

我们使用numpy array来产生x_input和y_input，以方便稍后的迭代(当我们输入占位符X和Y的输入时)。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/4.png)

非常完美！

这个故事的寓意:虚拟数据比真实世界数据要容易得多，因为虚拟数据是根据模型的假设生成的。现实世界是艰难的!

### 分析代码
我们的模型中的代码非常简单，除了两行:
```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01 ).minimize(loss)
sess.run(optimizer, feed_dict = {X: x, Y: y})
```

我记得我第一次遇到类似的代码时，我很困惑。两个问题:
1. 为什么训练操作要在会话循环中运行。
2. tensorFlow如何知道哪些变量需要更新。

实际上，我们可以在`tf.session.run()`中传入任何TensorFlow操作。TensorFlow将执行这些操作所依赖的图形的一部分。在这种情况下，我们看到训练的目的是最小化损失，而损失则取决于变量w和b。
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/5.png)

从图中,可以看到巨大的节点GrandientDescentOptimizer取决于三个节点:权重,偏差（bias）,和梯度(自动为我们计算)。

### 优化
梯度优化器意味着我们的更新规则是梯度下降。TensorFlow为我们进行自动微分，然后更新w和b的值以最小化损失。Autodiff是惊人的!
在默认情况下，优化器将训练所有可训练的变量，它们的目标函数依赖于这些变量。如果有一些你不想训练的变量，那么当你声明一个变量时，你可以将关键字训练设置为False。你不想训练的变量的一个例子是变量globalstep，这是一个常见的变量，你将在许多TensorFlow模型中看到，以跟踪你运行您的模型的次数。

```python
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)

increment_step = global_step.assign_add(1)
optimizer = tf.GradientDescentOptimizer(learning_rate) # learning rate can be a tensor
```

当我们讨论这个问题时，让我们来看看这个类 tf.Variable的完整定义:
```python
tf.Variable(initial_value = None, trainable = True, collections = None,  \
            validate_shape = True, caching_device = None, name = None, \ 
            variable_def = None, dtype = None, expected_shape = None, import_scope = None)
```

```python
# 创建优化器
optimizer = GradientDescentOptimizer(learning_rate = 0.1) 

# 通过一系列变量计算梯度
grads_and_vars = opt.compute_gradients(loss, < list of variables >)

# grads_and_vars 是一个数对表 (gradient, variable)。 自定义。 
# 例如，需要“梯度”部分，将每个部分都减去1。
subtracted_grads_and_vars = [(gv[0] - 1.0, gv[1]) for gv in grads_and_vars]

# 要求优化器按定义的方式进行梯度优化
optimizer.apply_gradients(subtracted_grads_and_vars)
```

更多关于梯度的操作
---

优化器类会自动在图上计算导数，但是自定义优化器或专家用户的可以调用底层函数。

```python
tf.gradients(ys, xs, grad_ys = None, name = 'gradients',  \
            colocate_gradients_with_ops = False, gate_gradients = False, \
            aggregation_method = None)
```

???技术细节：这在训练模型时特别有用。例如，我们可以使用tf.gradients()来将损失的G.r.t的微分G带到中间层。然后我们使用优化器来最小化中间层输出M和M+G之间的差异，这只会更新网络的下半部分。???
This is especially useful when training only parts of a model. For example, we can use tf.gradients() for to take the derivative G of the loss w.r.t. to the middle layer. Then we use an optimizer to minimize the difference between the middle layer output M and M + G. This only updates the lower half of the network.

### 优化器列表
梯度优化器不是TensorFlow支持的唯一更新规则。这是TensorFlow支持的优化器列表，截止到2017年1/8。名字都是不言而喻的。您可以访问[官方文档](https://www.tensorflow.org/api_docs/python/train/)了解更多细节:
- tf.train.GradientDescentOptimizer 
- tf.train.AdadeltaOptimizer 
- tf.train.AdagradOptimizer 
- tf.train.AdagradDAOptimizer 
- tf.train.MomentumOptimizer 
- tf.train.AdamOptimizer 
- tf.train.FtrlOptimizer 
- tf.train.ProximalGradientDescentOptimizer 
- tf.train.ProximalAdagradOptimizer 
- tf.train.RMSPropOptimizer

Sebastian Ruder是数据分析研究中心的博士候，[他在博客](http://sebastianruder.com/optimizing-gradient-descent/)中对这些优化器进行了较为详细的比较。如果你懒得阅读，以下是结论:

RMSprop是Adagrad的一个扩展，它可以快速降低学习速度率。它与Adadta是相同的，除了Adadta在分子更新规则中使用了参数更新的RMS。最后，Adam对RMSprop补充了“bias-correction”和“momentum”。在这种情况下，RMSprop、Adadta和Adam都是非常相似的算法。Kingma等显示，它的“修正”帮助Adam在优化结束时略微超越了RMSprop，因为梯度变得稀疏。在这一问题上，Adam可能是最好的选择。”

“RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. [15] show that its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice.”

简而言之，用`AdamOptimizer`。

### 思考：
我们可以用线性回归解决哪些现实世界的问题呢?你能写一个简单的程序吗?

## TensorFlow实现逻辑斯谛回归
如果没有逻辑回归，我们就不能讨论线性回归。让我们来解释一下TensorFlow中的逻辑回归，它是解决了MNIST问题的老分类器。

MNIST数据库(Mixed National Institute of Standards and Technology database)可能是用于培训各种图像处理系统的最流行的数据库之一，它是一个手写数字的数据库。图像是这样的:
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/6.png)

每个图像都是28x 28像素，拉伸为为1维张量，大小为784。每一个都有一个标签。例如，第一行上的图像被标记为0，第二个为1，以此类推。数据集是托管在网站[雅安·勒存](http://yann.lecun.com/exdb/mnist/)。

TF学习(TensorFlow的简化界面)有一个脚本，让你从Yann Lecun的网站上加载MNIST的数据集，并把它分成训练集、验证集和测试集。

```python
from tensorflow.examples.tutorials.mnist import input_data 
MNIST = input_data.read_data_sets("/data/mnist", one_hot = True)
```

MNIST是一个TensorFlow的数据集对象。它有55,000个数据点(mnist.train)，10,000点的测试数据(mnist.test)，以及5000点的验证数据(mnist.validation)。

逻辑回归模型的构建与线性回归模型非常相似。然而，现在我们有了更多的数据。我们在CS229中学到了，如果我们在每一个数据点之后计算梯度，它将会非常缓慢。解决这个问题的一种方法是将它们进行批量处理。幸运的是，TensorFlow对批处理数据有很好的支持。

为了进行批处理逻辑回归，我们只需要更改x占位符和y占位符的维度，以便能够容纳更多的数据点。
```python
X = tf.placeholder(tf.float32, [batch_size, 784], name = "image") 
Y = tf.placeholder(tf.float32, [batch_size, 10], name = "label")
```

当你将数据输入到占位符时，而不是为每个数据点提供数据，我们可以以数据点的数量来提供数据。

```python
X_batch, Y_batch = mnist.test.next_batch(batch_size) 
sess.run(train_op, feed_dict = { X : X_batch, Y : Y_batch })
```

这是全部的实现代码：
```python
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Step 1: 读入数据
# 使用TF Learn在/data/mnist文件夹中内置的数据
MNIST = input_data.read_data_sets("/data/mnist", one_hot = True)

# Step 2: 定义模型所需要参数
learning_rate = 0.01 
batch_size = 128 
n_epochs = 25

# Step 3: 为特征设定占位符
# 每幅MNIST图像大小为28x28，长度为28*28 = 784
# 因此每幅图片代表的是1x784的张量
# 图像共分为10十类，分别对应0 - 9. 
# 每个标签也是一个向量.
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

# Step 4: 设定权重和偏差量
# w 被设定为平均值为0的随机数
# b 初始化为0
# w的形状取决于X和Y，所以Y = tf.matmul(X, w)
# b的形状依赖于y
w = tf.Variable(tf.random_normal(shape = [784, 10], stddev = 0.01), name = "weights")
b = tf.Variable(tf.zeros([1, 10]), name = "bias")

# Step 5:  通过X、w和b预测Y
# 返回图像可能标签的概率分布的模型
# 通过softmax函数
# 一批 x 10 大小的矩阵代表了图像属于某一类的分类
logits = tf.matmul(X, w) + b

# Step 6: 定义损失函数
# 使用softmax交叉熵以及对数作为损失函数
# 计算平均交叉熵, softmax是一个内部函数
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy) 
# 计算每一批数据的softmax交叉熵均值

# Step 7: 定义训练过程
# 使用梯度下降法，学习速率0.01减小损失函数 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init)
    n_batches = int(MNIST.train.num_examples / batch_size)
    for i in range (n_epochs):   # train the model n_epochs times
        for _ in range (n_batches):
        X_batch, Y_batch = MNIST.train.next_batch(batch_size) 
        sess.run([optimizer, loss], feed_dict = { X : X_batch, Y : Y_batch})
# 在25轮迭代后，大概损失在0.35左右
```

在我的Mac上，批量大小为128的模型运行时间大概在0.5秒，而非批处理模型运行时间大概在24秒！但是，请注意，较高的批处理大小通常需要更多的时间，但它执行的更新步骤更少。在Bingios的实用技巧中，可以看到[迷你批量](https://arxiv.org/pdf/1206.5533v2.pdf)。

我们可以通过测试集对其进行测试，让我们看看如何在TensorFlow中完成它。

```python
# 测试模型
n_batches = int(MNIST.test.num_examples / batch_size) 
total_correct_preds = 0
for i in range(n_batches):
    X_batch, Y_batch = MNIST.test.next_batch(batch_size)
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict = {X : X_batch, Y : Y_batch})
    preds = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1)) 
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))   # 和numpy.count_nonzero(boolarray)相同 :(
    total_correct_preds += sess.run(accuracy)
    print "Accuracy {0}".format(total_correct_preds / MNIST.test.num_examples)
```

在10轮迭代之后，我们的准确率达到了90%。这就是我们从线性分类器中得到的。
注意:TensorFlow为MNIST提供了一个馈送器(数据集解析器)，但不要指望它为任何数据集提供一个反馈。你应该学习如何编写自己的数据解析器。

这是我们的图表在TensorBoard上的样子:

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note3/markdown/img/note3/7.png)

我们讲在下节课中讲解如何构建模型。

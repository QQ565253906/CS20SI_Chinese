# 二、TensorFlow的操作符
> 内容由Chip Huye编写，Danijar Hafner校对
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_01.pdf)

## 玩转TensorBoard

在TensorFlow中，我们将常量、变量、操作符统称为操作符。TensorFlow不仅是一个软件库，还包括一套软件，包括TensorFlow、TensorBoard和TensorServing。为了最大限度地利用TensorFlow，我们应该知道如何将以上所有的应用结合在一起。在这节课中，我们将首先介绍TensorBoard。

TensorBoard是TensorFlow库中的图形可视化软件。用[谷歌自己的话](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)来说:“TensorFlow可以进行的计算是和神经网络一样复杂的。为了更容易地理解、调试和优化TensorFlow程序，我们在TensorFlow的库中加入了一套叫做TensorBoard的可视化工具。

当完全配置时，TensorBoard将会是这样的。图像取自TensorBoard的网站。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/1.png)

当用户在tensorboard激活的TensorFlow程序中执行某些操作时，这些操作将被导出到一个事件文件中。TensorBoard能够将这些事件文件转换为图形，从而能够观察模型的行为。学会使用TensorBoard，会让TensorFlow的工作变得更加有趣和高效。

让我们写下第一个TensorFlow项目，并把它用TensorBoard表示出来。

```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(x))
```

要启动这个程序的TensorBoard，在你建立了你的模型之后，在运行训练的语句之前之前加上这行。

```python
writer = tf.summary.FileWriter(logs_dir, sess.graph)
```

上面的行是创建一个writer对象来将操作写到事件文件中，存储在文件夹logs_dir中。您可以将logs_dir命名为'./graphs'之类。

```python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
#关闭文件输出流
writer.close()
```

接下来，到终端运行程序。**确保您当前的工作目录与运行Python代码的位置相同。**

```bash
$ python  [ yourprogram . py ]
$ tensorboard  -- logdir = "./graphs"
```

打开您的浏览器，进入http://localhost:6006/ (或在运行tensorboard命令后返回的链接)。

进入标签页，你会看到这样的东西:
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/2.png)

转到Graph，你可以看到一张3个节点的图
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/3.png)

```python
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
```
“Const”和“Const_1”分别表示a和b，而节点“Add”表示x。我们给它们的名称(a、b和x)仅仅是供我们访问需要。它们在TensorFlow中并没有任何意义。要让TensorBoard显示你的操作的名称，你必须专门命名它们。

```python
a = tf.constant([2, 2], name = 'a')
b = tf.constant([3, 6], name = 'b')
x = tf.add(a, b, name = 'add')
```

现在，如果你再次运行TensorBoard，你会看到这张图:
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/4.png)

图本身定义了操作和依赖关系，但不显示值。它只运行会话时才会显示一些值。提醒一下，防止你忘了:

```python
tf.Session.run(fetches, feed_dict = None, options = None, run_metadata = None)
```

**注意:如果你已经多次运行您的代码,将会有多个事件文件存储在 "~/dev/cs20si/graph/ lecture01"中,TF将只显示最新的图并显示多个事件文件的警告。要删除警告，你需要删除你不再需要的所有事件文件。**

## 常数类型
原文档链接，(https://www.tensorflow.org/api_docs/python/constant_op/)

### 你可以通过以下语句创建标量或张量。
```python
tf.constant(value, dtype = None, shape = None, name = 'Const', verify_shape = False)
```

```python
# 创建一维张量（向量）
a  =  tf.constant([2, 2],  name = "vector")
# 创建一个2x2的张量 (矩阵)
b  =  tf.constant ([[0, 1], [2, 3]], name = "b")
```

### 你也可以创建赋有特定的值的张量
**注意与numpy的相似性**

```python
tf.zeros(shape, dtype = tf.float32, name = None)
# 创建一个shape型的全部元素都是0的张量

tf.zeros([2, 3],  tf.int32)   ==>   [[ 0, 0, 0], [0, 0, 0]]
```

```python
tf.zeros_like(input_tensor, dtype = None, name = None, optimize = True)
# 创建一个与输入张量大小相同的内部值全为零的张量。

# 输入的张量为[[0, 1], [2, 3], [4, 5]]
tf.zeros_like(input_tensor)   ==>   [[0, 0], [0, 0], [0, 0]]
```

```python
tf.ones(shape, dtype = tf.float32, name = None)
# 创建一个shape大小的内部值全为1的张量。

tf.ones([2, 3], tf.int32)   ==>   [[1, 1, 1], [1, 1, 1]]
```

```python
tf.ones_like(input_tensor, dtype = None, name = None, optimize = True)
# 创建一个与输入张量大小相同的内部值全为1的张量。

# 输入的张量为[[0, 1], [2, 3], [4, 5]]
tf.ones_like(input_tensor)   ==>  [[1, 1], [1, 1], [1, 1]]
```

```python
tf.fill(dims, value, name = None)
# 创建一个用value填充的张量。

tf.fill([2, 3], 8)   ==>   [[8, 8, 8], [8, 8, 8]]
```

```python
tf.linspace(start, stop, num, name = None)
# 创建一个num个数的以start为起点，stop为终点的等差数列。
# start, stop, num 必须是标量
# 和numpy.linspace稍稍有区别
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

tf.linspace(10.0, 13.0, 4, name = "linspace")   ==>   [ 10.0   11.0   12.0   13.0]
```

```python
tf.range(start, limit = None, delta = 1, dtype = None, name = 'range')
# 创建一个以start为开始，limit为结束（不包括），步长为delta的数列。
# 和python稍稍不同。 
# 'start' is 3, 'limit' is 18, 'delta' is 3

tf.range(start, limit, delta)   ==>   [3, 6, 9, 12, 15] # 'start' is 3, 'limit' is 1, 'delta' is -0.5
tf.range(start, limit, delta)   ==>   [3, 2.5, 2, 1.5]
# 'limit' is 5

tf.range(limit)   ==>   [0, 1, 2, 3, 4]
```
**注意：不像numpy或其他Python序列，tf.range是不可迭代访问的**

```python
for _ in np.linspace(0, 10, 4):   # OK
for _ in tf.linspace(0, 10, 4):   # TypeError("'Tensor' object is not iterable.")
for _ in range(4):   # OK
for _ in tf.range(4):   # TypeError("'Tensor' object is not iterable.")
```

你也可以通过指定的分布求得随机常数。

```python
tf.random_normal(shape, mean = 0.0, stddev = 1.0, dtype = tf.float32, seed = None, name = None) 
tf.truncated_normal(shape, mean = 0.0, stddev = 1.0, dtype = tf.float32, seed = None, name = None)
tf.random_uniform(shape, minval = 0, maxval = None, dtype = tf.float32, seed = None, name = None)
tf.random_shuffle(value, seed = None, name = None)
tf.random_crop(value, size, seed = None, name = None)
tf.multinomial(logits, num_samples, seed = None, name = None) 
tf.random_gamma(shape, alpha, beta = None, dtype = tf.float32, seed = None, name = None)
```

## 数学运算
TensorFlow的数学运算是相当类似于NumPy。访问(https://www.tensorflow.org/api_docs/python/math_ops/arithmetic_operators)获取更多，因为单独列举它们每一个运算是非常无聊的。

```python
a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b) # >> [5 8]
tf.add_n([a, b, b]) # >> [7 10]. 等价于 a + b + b
tf.mul(a, b) # >> [6 12] Hadamard积
tf.matmul(a, b) # >> ValueError
tf.matmul(tf.reshape(a, shape=[1, 2]), tf.reshape(b, shape=[2, 1])) # >> [[18]] tf.div(a, b) # >> [1 3]
tf.mod(a, b) # >> [1 0]
```

下面是Python的操作表，由“深度学习基础”的作者提供。


|类别|例子|
|--|--|
|通用型型计算符号|Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal...|
|数组操作|Concat, Slice, Split, Constant, Rank, Shape, Shuffle...|
|矩阵操作|MatMul, MatrixInverse, MatrixDeterminant...|
|状态操作符|Variable, Assign, AssignAdd...|
|建立神经网络用模块|SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool...|
|检查操作符|Save, Restore...|
|队列与同步操作|Enqueue, Dequeue, MutexAcquire, MutexRelease...|
|控制流操作|Merge, Switch, Enter, Leave, NextIteration...|

## 数学类型

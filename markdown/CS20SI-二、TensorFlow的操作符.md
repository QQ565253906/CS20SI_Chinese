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
### Python原生类
TensorFlow采用Python的原生类型，例如Python布尔值、数值(整数、浮点数)和字符串。单个值将被转换为0维张量(或标量)，值列表将被转换为1维张量(向量)，值列表的列表将被转换为2维张量。
张量(矩阵)，等等。下面的例子被改编并修改为“TensorFlow for Machine Intelligence”。

```python
t_0 = 19   # 被当作0维张量或者标量
tf.zeros_like (t_0)   # ==> 0
tf.ones_like(t_0)   # ==> 1
t_1 = [b"apple", b"peach", b"grape"]   # 被当作一维张量或向量
tf.zeros_like (t_1)   # ==> ['' '' '']
tf.ones_like(t_1)   # ==> TypeError: Expected string, got 1 of type 'int' instead.
t_2 = [[True, False, False], 
       [False, False, True],
       [False, True, False]]   # 当作二维张量或矩阵
tf.zeros_like(t_2)   # ==> 2x2矩阵 所有的元素都是False 
tf.ones_like(t_2)   # ==> 2x2矩阵 所有的元素都是True
```
### TensorFlow原生类型
和NumPy一样，TensorFlow也有自己的数据类型，就像你看到的tf.int32，tf.float32。下面是来自TensorFlow官方文档的当前TensorFlow数据类型的列表。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/5.png)

### NumPy数据类型
这NumPy已经成为数据科学的通用语言。现在，您可能已经注意到了NumPy和TensorFlow之间的相似之处。TensorFlow被设计为与Numpy无缝集成。

TensorFlow的数据类型是基于NumPy的;实际上，np.int32==tf.int32返回True。您可以将NumPy类型传递给TensorFlow操作。
例子：

```python
tf.ones([2, 2], np.float32)   ==>   [[1.0 1.0], [1.0 1.0]]
```

还记得我们的好朋友`tf.session.run(fetch)`吗?如果被请求的对象是一个张量，那么它的输出将是一个NumPy数组。

**大多数情况下，您可以交替使用TensorFlow类型和NumPy类型。**

**这里有一个字符串数据类型的捕获。对于数值型和布尔型类型，TensorFlow和NumPy dtypes都是匹配的。然而由于NumPy处理字符串的方式，字符串在NumPy中没有确切的匹配。TensorFlow仍然可以从NumPy中导入字符串数组，只是不要在NumPy中指定dtype！**

**TensorFlow和NumPy都支持n维数组。NumPy支持ndarray，但不提供创建张量函数和自动计算导数的方法，也不提供GPU支持。所以TensorFlow还是更有优势!**

**使用Python类型来指定TensorFlow对象是快速而简单的，并且它对于原型设计非常有用。然而，这样做有一个很明显的缺陷。Python类型缺乏显式地声明数据类型的能力，但是TensorFlow的数据类型更具体。例如，所有的整数都是相同的类型，但是TensorFlow有8位、16位、32位和64位的整数。因此，如果使用Python类型，TensorFlow必须推断出您的数据类型。**

当您将数据转换为TensorFlow时，可以将数据转换为适当的类型，但是某些数据类型仍然难以正确地声明，例如复数。因此，将手动定义的张量对象创建为NumPy数组是很常见的。但是，在可能的情况下，最好是使用TensorFlow类型，因为TensorFlow和NumPy都可以做到同样的效果，这样的兼容性就不再存在了。

## 变量
常数一直都很有用，但我认为你们现在已经足够理解变量了。常量和变量之间的区别:
1. 一个常数是恒定的。一个变量可以被赋值，它的值可以被改变。
2. 一个常量的值存储在图中，它的值随图的每次加载而生成。变量是单独存储的，并且可以在一个参数服务器上保存。

第2点说的是着常量存储在图定义中。当常量存储代价很大时，每次加载图形时都会很慢。要查看图的定义和存储在图的定义中的内容，只需打印出图的原始缓存。原始缓存表示协议缓冲区，？？？“Google的语言中立、平台无关、可扩展的用于序列化结构化数据的机制——考虑XML，但更小、更快、更简单。”？？？

```python
import tensorflow as tf

my_const = tf.constant([1.0, 2.0], name="my_const") 
print tf.get_default_graph().as_graph_def()
```
输出
```json
node {
    name: "my_const" op: "Const"
    attr {
        key: "dtype" value {
            type: DT_FLOAT 
        }
    } 
    attr {
        key: "value" value {
            tensor {
                dtype: DT_FLOAT tensor_shape {
                    dim { 
                        size: 2
                    }   
                }
            tensor_content: "\000\000\200?\000\000\000@" 
            }
        } 
    }
}
versions {
    producer: 17 
}
```

### 声明变量
？？？ To declare a variable, you create an instance of the class tf.Variable. Note that it’s tf.constant but tf.Variable and not tf.variable because tf.constant is an op, while tf.Variable is a class. ？？？

```python
#创建一个变标量
a = tf.Variable(2, name = "scalar")
#创建一个变向量
b = tf.Variable([2, 3], name = "vector")
#创建2x2变矩阵
c = tf.Variable([[0, 1], [2, 3]], name = "matrix")
# 创建一个784x10的变矩阵
W = tf.Variable(tf.zeros([784, 10]))
```

tf.Variable可以进行下面几个操作
```python
x = tf.Variable(...)
x.initializer # 初始化
x.value() # 读取值
x.assign(...) # 空值
x.assign_add(...)
# and more
```

**在使用变量之前，必须先初始化变量。**如果在初始化这些变量之前尝试调用这些变量，那么就会遇到失败:尝试使用未初始化的值张量。

最简单的方法是初始化所有变量：`tf.global_variables_initializer()`

```python
init = tf.global_variables_initializer()

with tf.Session() as sess:
    tf.run(tf)
```
注意，您使用`tf.run()`来运行初始化器，而不是获取任何值。

要初始化一个变量的子集，您需要使用`tf.variables_initializer()`，并使用您想要初始化的变量列表:
```python
init_ab = tf.variables_initializer([a, b], name = "init_ab") 
with tf.Session() as sess:
    tf.run(init_ab)
```

您还可以使用tf.variable.initializer分别初始化每个变量。
```python
# 创建一个全是0的784x10的矩阵
W = tf.Variable(tf.zeros([784, 10])) 
with tf.Session() as sess:
    tf.run(W.initializer)
```

初始化变量的另一种方法是从保存文件中恢复它。我们将在几周内讨论。

### 获取变量值
如果我们打印初始化的变量，我们只看到张量对象。

```python
# W是一个随机的700 x 100变量矩阵
W = tf.Variable(tf.truncated_normal([700, 10])) 
with tf.Session() as sess:
    sess.run(W.initializer) 
    print W

>> Tensor("Variable/read:0", shape =(700, 10), dtype = float32)
```

要获得一个变量的值，我们需要使用eval()。
```python
# W是一个随机的700 x 100变量矩阵
W = tf.Variable(tf.truncated_normal([700, 10])) 
with tf.Session() as sess:
    sess.run(W.initializer) 
    print W.eval()

>>   [[- 0.76781619   - 0.67020458   1.15333688   ...,   - 0.98434633   - 1.25692499  - 0.90904623]
 [- 0.36763489   - 0.65037876   - 1.52936983   ...,   0.19320194   - 0.38379928  0.44387451]
 [   0.12510735   - 0.82649058   0.4321366   ...,   - 0.3816964   0.70466036  1.33211911]
 ...,
 [   0.9203397   - 0.99590844   0.76853162   ...,   - 0.74290705   0.37568584
 0.64072722]
 [- 0.12753558   0.52571583   1.03265858   ...,   0.59978199   - 0.91293705
 - 0.02646019]
 [   0.19076447   - 0.62968266   - 1.97970271   ...,   - 1.48389161   0.68170643
 1.46369624 ]]
```

### 给随机变量赋值
我们可以使用`tf.Variable.assign()`为变量分配一个值。
```python
W = tf.Variable(10) 
W.assign(100)

with tf.Session() as sess:
    sess.run(W.initializer) 
    print W.eval() # >> 10
```
为什么是10而不是100? `W.assign(100)`不将值分配给W，而是创建一个赋值操作来执行该操作。为了使这个op生效，我们必须在会话中运行这个op。

注意，在这种情况下，我们没有初始化W，因为assign()为我们进行了赋值。实际上，初始化器op是分配变量初始值给变量本身的赋值op。
```python
# in the  (source code)[https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/variables.py] （上不去）
self._initializer_op = state_ops.assign(self._variable, self._initial_value, validate_shape = validate_shape).op
```

一个有趣的例子
```python
# 创建一个原值是2的变量
a = tf.Variable(2, name = "scalar")
# 创建一个给a赋值为a*2的操作
a_times_two = a.assign(a * 2)
init = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init)
    # 必须进行初始化，因为a_times_two操作依赖a的值
    sess.run(a_times_two)   # >> 4
    sess.run(a_times_two)   # >> 8
    sess.run(a_times_two)   # >> 16
```
TensorFlow在每次对a_times_two进行操作时，都会把a乘2。

对于变量的简单递增和分解，TensorFlow包含`tf.Variable.assign_add()`和`tf.Variable.assign_sub()`方法。与`tf.Variable.assign()`不同，`tf.Variable.assign_add()`和`tf.Variable.assign_sub()`不会初始化你的变量，因为这些操作取决于变量的初始值。

```python
W = tf.Variable(10)

with tf.Session() as sess: 
    sess.run(W.initializer)
    print sess.run(W.assign_add(10)) # >> 20 
    print sess.run(W.assign_sub(2)) # >> 18
```

因为TensorFlow会话单独维护变量的值，在每一次会话中变量的值不同。
```python
W = tf.Variable(10)

sess1 = tf.Session() 
sess2 = tf.Session()
sess1.run(W.initializer) 
sess2.run(W.initializer)
print sess1.run(W.assign_add(10)) # >> 20 
print sess2.run(W.assign_sub(2)) # >> 8
print sess1.run(W.assign_add(100)) # >> 120 
print sess2.run(W.assign_sub(50)) # >> -42
sess1.close() 
sess2.close()
```
**当然，你可以声明一个依赖于其他变量的变量。**
假设你要声明U = W * 2

```python
# W 是一个700 x 100矩阵
W = tf.Variable(tf.truncated_normal([700, 10])) 
U = tf.Variable(W * 2)
```
在本例中，你应该使用`initialized_value()`来确保在初始化W之前对W进行初始化。
```python
U = tf.Variable(W.intialized_value() * 2)
```

## InteractiveSession
你有时会看到InteractiveSession而不是会话。唯一的区别是，InteractiveSession使自己成为默认的会话，这样您就可以调用`run()`或`eval()`，而不需要显式地调用会话。这在交互式shell和IPython笔记本中是很方便的，因为它可以避免通过显式的会话对象来运行操作。但是，当您有多个会话要运行时，会使整个程序变得复杂。

```python
sess = tf.InteractiveSession() 
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# 我们只需要使用`eval()`函数就可以获得c的值。
print(c.eval()) 
sess.close()
```

`tf.InteractiveSession.close()`关闭一个InteractiveSession。
`tf.get_default_session()`返回当前线程的默认会话。返回的会话将是会话或会话的最内部会话。在上下文中使用`Session.as_default()`设定默认会话。

## 独立性控制
有时，我们将有两个独立的操作，但是你不想要指定应该先运行哪个op，你可以使用`tf.Graph.control_dependencies(control_inputs)`。
例子：
```python
# your graph g have 5 ops: a, b, c, d, e 
with g.control_dependencies([a, b, c]):
# `d`和`e` 将会在 `a`, `b`, 和 `c` 运行后再被执行. 
    d = ...
    e = ...
```

## Placeholders和feed_dict
记得在第一讲中，一个TensorFlow项目通常有两个阶段
1. 声明一个图。
2. 通过会话来对图进行操作。

因此，我们可以先组装数据流图，而不需要知道计算所需的值。这等价于定义x，y的函数，却不知道x的值，例如f(x，y)=x * 2 + y。
x，y是实际值的占位符。

在图形组装完成后，我们或客户端可以在需要执行计算时提供自己的数据。

定义一个Placeholder
```python
tf.placeholder(dtype, shape = None, name = None)
```
Dtype是指定占位符值数据类型的必需参数。

形状指定了可被接受为占位符的实际值的张量的形状。形状=不意味着任何形状的张量都将被接受。使用任意形状张量很容易构造图形，但用于调试的却是噩梦。你应该尽可能详细地定义你的占位符的形状。

你也可以给你的占位符一个名字，就像你在TensorFlow中的任何其他操作一样。

更多信息，参见(官方文档)[https://www.tensorflow.org/api_docs/python/io_ops/placeholders] （也上不去）

```python
# 创建一个float 32-bit类型, 三个元素长的向量占位符。
a = tf.placeholder(tf.float32, shape = [3])

# 创建一个float 32-bit类型, 三个元素长的常向量。
b = tf.constant([5, 5, 5], tf.float32)

# 像操作常量一样使用占位符
c = a + b  # tf.add(a, b)的简写

# 如果你尝试调用c，系统会报错。
with tf.Session() as sess: 
    print(sess.run(c))
>>   NameError
```

这会导致一个错误，因为要计算c，我们需要a的值，但是a只是一个没有实际值的占位符。我们必须先将实际值输入a。
```python
with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]} 
    # fetch value of c
    print(sess.run(c, {a: [1, 2, 3]}))

>> [6. 7. 8.]
```

让我们看一下它在TensorBoard中是怎么表示的。

```python
writer = tf.summary.FileWriter('./my_graph', sess.graph)
```

在命令行键加入以下内容：
```python
$ tensorboard -- logdir = 'my_graph'
```

正如您所看到的，占位符和运算符一样被处理，3是占位符的形状。
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/6.png)

在前面的例子中，我们为占位符提供一个单一的值。如果我们想要将多个数据提供给占位符，会怎么样呢?这是一个合理的假设，因为我们经常需要在训练或测试集中通过多个数据进行计算。

可以使用for循环迭代向会话中传入数据。
```python
with tf.Session() as sess:
    for a_value in list_of_a_values: 
        print(sess.run(c, {a: a_value}))
```

你可以给不是占位符的张量提供值。任何张量都可以被复制，来检查张量是否可以赋值，使用:
```python
tf.Graph.is_feedable(tensor)
```

```python
# 创建操作和张量，使用默认会话。
a = tf.add(2, 5)
b = tf.mul(a, 3)

# 开始一个会话，作为默认会话。
sess = tf.Session()

# 定义一个用于替代原张量的字典
replace_dict = {a: 15}

# 运行会话并加入replace_dict字典。
sess.run(b, feed_dict = replace_dict) # returns 45
```

feed_dict对您的模型进行测试是非常有用的。当您有一个大的图表，并且只想测试某些部分时，您可以提供虚拟的值，这样TensorFlow就不会浪费时间进行不必要的计算。

## 懒操作
我看到的最常见的TensorFlow的不是bug的bug(我曾经提交过)是我的朋友Danijar和我所称的“懒加载”。延迟加载是指你可以延迟声明/初始化一个对象直到它被加载时，它指的是一个编程模式。在TensorFlow的上下文中，这意味着您可以延迟创建op，直到您需要计算它时。例如，你在组装图表时才创建了操作z。

```python
x = tf.Variable(10, name='x') 
y = tf.Variable(20, name='y') 
z = tf.add(x, y)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
        for _ in range(10):
            sess.run(z) 
    writer.close()
```
这是某个聪明人使用延迟加载来保存一行代码时所发生的事情:

```python
x = tf.Variable(10, name='x') 
y = tf.Variable(20, name='y')
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
        for _ in range(10):
            sess.run(tf.add(x, y)) # 当你需要使用时再创建操作 
        writer.close()
```

让我们看一下它们在TensorBoard上的图形。正常的加载图看起来和我们期望的一样。
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/7.png)

懒操作
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note2/markdown/img/note2/8.png)

好吧，节点“Add”缺失了，这是可以理解的，因为我们在给FileWriter写了图之后添加了“Add”。这使得阅读图表变得更加困难，但它并不是一个错误。

让我们看一下图像的定义。记住，要打印出图形的定义，我们使用:
```python
print(tf.get_default_graph().as_graph_def())
```

在正常加载图中，图的原型只有1个节点“Add”:
```json
node {
    name: "Add" 
    op: "Add" 
    input: "x/read" 
    input: "y/read" 
    attr {
        key: "T" 
        value {
            type: DT_INT32 
        }
    } 
}
```

另一张图，在懒操作的图上的原始缓存中有10个节点的“Add”。每次您想要计算z时，它都会添加一个新的节点“添加”。

```json
node {
    name: "Add" 
    op: "Add"
    ...
}

node {
    name: "Add_9" 
    op: "Add"
    ...
}
```

你可能会想:“这太蠢了。为什么我要多次计算相同的值呢?很少有人认为这是一个错误。它发生的频率比你想象的要高。例如，你可能想要计算相同的损失函数，或者在一定数量的培训样本之后进行一些预测。在你了解它之前，你已经计算了数千次，并在图中添加了数千个不必要的节点。你的图形定义变得臃肿，加载速度慢，并且代价更高。

有两种方法可以避免这种bug。首先，在可以的时候，总是将操作的定义和它们的执行分开。如果你想将相关的操作组分组到类中，你可以使用Python属性来确保您的函数在第一次调用时只加载一次。这不是一门Python课程，所以我不会深入探讨如何去做。但是，如果你想知道，看看[Danijar Hafner的这篇精彩的博客文章](http://danijar.com/structuring-your-tensorflow-models/)。（别点了，这个也没有）
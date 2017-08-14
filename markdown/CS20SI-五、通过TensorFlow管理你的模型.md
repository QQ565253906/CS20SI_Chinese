# 五、通过TensorFlow管理你的模型
> 内容由Chip Huye编写，Danijar Hafner校对
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_05.pdf)

我们已经建立了word2vec模型，对于它所使用的小型数据集来说，它似乎运行得很好。但我们知道，对于一个更大的数据集来说，需要花费更长的时间，而且我们也知道，训练更复杂的模型可能会耗费大量的时间。例如，在CS 224N的未来作业通常需要4个小时。做抽象的总结可能需要几天的时间，即使是在非常强大的gpu上，也只能得到不好的效果。许多计算机视觉任务需要更长的时间来进行训练。

我们无法忍受让我们的模型运行数天，等着看它们如何运行，然后进行调整。或者，如果我们的电脑崩溃了，训练就会中断，我们将不得不重新运行我们的模型。因此在任何时候都能停止训练是至关重要的，因为任何原因，并且恢复训练，就好像什么都没有发生一样。它将特别有助于分析我们的模型，因为这使我们可以在任何训练步骤之后对我们的模型进行仔细的检查。

在这节课中，我们将讨论TensorFlow提供的一系列优秀的工具，帮助我们管理我们的实验。我们今天讨论的主题包括:tf.train.Saver()类、TensorFlow的随机种子和NumPy的随机状态，以及可视化我们的培训进度(也就是更多的TensorBoard)。

## tf.train.Saver()
一个好的做法是在一定数量的步骤之后定期地保存模型的参数。如果需要的话，我们可以从这一步恢复/再培训我们的模型。`tf.train.Saver()`类允许我们通过保存二进制文件中的图形变量来实现这一操作。

```python
tf.train.Saver.save(sess, save_path, global_step = None, latest_filename = None, \
                    meta_graph_suffix = 'meta', write_meta_graph = True, write_state = True)
```

例如，如果我们想在每1000个训练步骤之后保存图表的变量，我们就按如下做:

```python
# 定义模型
# 创建一个saver对象
saver = tf.train.Saver()
# launch a session to compute the graph
with  tf . Session ()   as  sess: # actual training loop
    for step in range(training_steps):
        sess.run([optimizer])
        if (step + 1) % 1000 == 0:
            saver.save(sess, 'checkpoint_directory/model_name', \
                        global_step=model.global_step)
```

在TensorFlow术语中，保存图形变量的步骤称为检查点。由于我们将创建许多检查点，因此在一个名为globalstep的变量中添加我们的模型所经过了培训步骤的数量是很有帮助的。在TensorFlow程序中，它是一个不同的通用变量。我们首先需要创建它，将它初始化为0，并将它设置为不可训练，因为我们不想让TensorFlow来优化它。

```python
self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step')
```
我们需要将globalstep作为一个参数传递给优化器，这样它就知道在每个训练步骤中增加一个全局步骤:

```python
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                            global_step = self.global_step)
```

为了在文件夹的“检查点”中保存会话的变量，使用名称模型-全局步骤，我们使用以下方法:

```python
saver.save(sess, 'checkpoints/skip-gram', global_step = model.global_step)
```

所以我们的word2vec的培训循环现在看起来是这样的:

```python
self.global_step = tf.Variable(0, dtype = tf.int32, trainable = False, name = 'global_step') 

self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,  global_step = self.global_step)

saver = tf.train.Saver()   # 定义存储所有变量  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0.0
    writer = tf.summary.FileWriter('./improved_graph', sess.graph)  
    for index in xrange(num_train_steps):
        batch = batch_gen.next()
        loss_batch, _ = sess.run([model.loss, model.optimizer], feed_dict = {model.center_words : batch[0], \
                                                                            model.target_words : batch[1]})
        average_loss  +=  loss_batch                            
        if (index + 1) % 1000 == 0:
            saver.save(sess, 'checkpoints/skip-gram', global_step = model.global_step)
```

如果你去文件夹'检查点'你会看到像belo这样的文件

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note5/markdown/img/note5/1.png)

为了恢复变量，我们使用`tf.train.Saver.restore(sess, save_path)`。例如，您希望在10,000次迭代的时候恢复检查点。

```python
saver.restore(sess, 'checkpoints/skip-gram-10000')
```

当然，如果有有效的检查点，我们只能加载保存的变量。你可能想要做的是，如果有一个检查点，就恢复它。如果没有，从一开始就训练。TensorFlow允许您从一个带有`tf.train.get_checkpoint_state(‘directory-name’)`的目录中获得检查点。检查的代码看起来是这样的:

```python
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
```

文件检查点自动更新最新检查点的路径。

```
model_checkpoint_path : "skip-gram-21999" 
all_model_checkpoint_paths : "skip-gram-13999" 
all_model_checkpoint_paths : "skip-gram-15999" 
all_model_checkpoint_paths : "skip-gram-17999" 
all_model_checkpoint_paths : "skip-gram-19999"
all_model_checkpoint_paths : "skip-gram-21999" 
```

默认情况下，saver.save()推荐存储图的所有变量。但是，你也可以设置一个列表，用来记录需要保存哪些变量，在我们创建保存对象时将它们作为一个参数。来自TensorFlow文档示例。

```python
v1 = tf.Variable(..., name = 'v1') 
v2 = tf.Variable(..., name = 'v2')  
# 讲需要保存的变量以字典形式存入     
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# 以列表形式存入
saver = tf.train.Saver([v1, v2])

# 传递一个列表相当于用变量op名传递一个字典 # 作为主键
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```
注意，存储器只保存变量，而不是整个图表，所以我们仍然需要自己创建图表，然后加载变量。检查点指定了从变量名映射到张量的方法。

人们通常不只是保存上次迭代中参数，还会保存出最佳结果的参数。以便你可以在目前最好的参数上对模型进行评估。

## tf.summary
我们一直在使用matplotlib来可视化我们的损失和准确性，这很有效。但对于TensorFlow来说，没有必要，因为TensorBoard为我们提供了一组很好的工具来可视化我们训练期间的汇总统计数据。一些常用的统计数据是损失，平均损失，准确性。你可以把它们想象成标量图，直方图，甚至图像。所以我们在图中有一个新的命名空间来容纳所有的总结操作。

```python
def _create_summaries(self):
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.histogram("histogram loss", self.loss) 
        # 因为你有多个反馈图表，你需要把它们合并到一起
        # 为便于操作，将其整合为一个操作符 
        self.summary_op = tf.summary.merage_all()
```

因为它是一个操作符，所以你需要通过`sess.run()`来执行它。

```python
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], \
                                feed_dict = feed_dict)
```

现在您已经获得了summary，您需要使用我们创建的相同的FileWriter对象来将summary文件写入文件。

```python
writer.add_summary(summary, global_step = step)
```

@ Now, if you go run tensorboard and go to h  ttp://localhost:6006/,  in the Scalars page, you will see the plot of your scalar summaries. This is the summary of your loss in scalar plot.

现在，如果您运行tensorboard并进入http://localhost:6006/，在标量页面中，您将看到标量summary的表现。这是你在标量图中损失的反映。
![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note5/markdown/img/note5/2.png)

@ And the loss in histogram plot.
还有通过矩形图表示的损失。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note5/markdown/img/note5/3.png)

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note5/markdown/img/note5/4.png)

@ If you save your summaries into different sub-folder in your graph folder, you can compare your progresses. For example,the first time we run our model with learning rate 1.0, we save it in ‘improved_graph/lr1.0’ and the second time we run our model, we save it in ‘improved_graph/lr0.5’, on the left corner of the Scalars page, we can toggle the plots of these two runs to compare them. This can be really helpful when you want to compare the progress made with different optimizers or different parameters.

如果将Summary文件保存到图表文件夹中的不同子文件夹中，则可以比较您的训练效果。例如，我们第一次使用学习率1.0来运行我们的模型，我们将它保存在“改进图/lr1”中，第二次运行我们的模型时，我们将它保存在“改进图/lr0.0.5”中，在标量页面的左角上，我们可以切换这两个运行的图来比较它们。当你想要比较不同的优化器或不同的参数所取得的进展时，这是非常有用的。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note5/markdown/img/note5/5.png)

@ You can write a Python script to automate the naming of folders where you store the graphs/plots of each experiment.

你可以编写一个Python脚本来管理命名文件夹的名称，在其中存储每个实验的Summary。

@ You can visualize the statistics as images using tf.summary.image.

你可以使用tf.summary.image将统计信息可视化。

```python
tf.summary.image(name, tensor, max_outputs = 3, collections = None)
```

## 随机控制化
@ I never realized what an oxymoron this sounds like until I’ve written it down, but the truth is that you often have to control the randomization process to get stable results for your experiments. You’re probably familiar with random seed and random state from NumPy. TensorFlow doesn’t allow to you to get random state the way numpy does (at least not that I know of -- I will double check), but it does allow you to get stable results in randomization through two ways:

在我写下来之前，我从来没有意识到这是一种矛盾的说法。但事实是，你必须控制随机化过程，才能得到稳定的实验结果。你可能很熟悉随机的种子和NumPy的随机状态。TensorFlow不允许你按照numpy的方式获得随机的状态(至少不是我所知道的，我一定会加倍检查)，但是它确实允许你通过两种方式获得稳定的结果:

@ 1. Set random seed at operation level. All random tensors allow you to pass in seed value in their initialization. For example:
1. 在操作层设置随机的种子。所有随机的张量允许您在初始化时传递种子值。例如:
```python
my_var = tf.Variable(tf.truncated_normal((-1.0, 1.0), stddev = 0.1, seed = 0))
```
@Note that, session is the thing that keeps track of random state, so each new session will start the random state all over again.
注意，会话是跟踪随机状态的东西，因此每个新会话将重新启动随机状态。
```python
c = tf.random_uniform([], -10, 10, seed = 2)
with tf.Session() as sess:
    print sess.run(c) # >> 3.57493 
    print sess.run(c) # >> -5.97319
```
```python
c = tf.random_uniform([], -10, 10, seed = 2) 
with tf.Session() as sess:
    print sess.run(c) # >> 3.57493

with tf.Session() as sess:
    print sess.run(c) # >> 3.57493
```
@  With operation level random seed, each op keeps its own seed.
使用操作级别的随机种子，每个操作符保持它的自己的种子。
```python
c = tf.random_uniform([], -10, 10, seed = 2) 
d = tf.random_uniform([], -10, 10, seed = 2)

with tf.Session() as sess:
    print sess.run(c) # >> 3.57493
    print sess.run(d) # >> 3.57493
```

@ 2. Set random seed at graph level with tf.Graph.seed
2. 用tf.Graph.seed在图层上设置随机种子
```python
tf.set_random_seed(seed)
```
@ If you don’t care about the randomization for each op inside the graph, but just want to be able to replicate result on another graph (so that other people can replicate your results on their own graph), you can use tf.set_random_seed instead. Setting the current TensorFlow random seed affects the current default graph only.
如果你不关心图中每个操作符的随机化，仅仅只是希望能够在另一个图上复制结果(以便其他人可以在自己的图上复制结果)，那么您可以使用tf.set_random_seed代替。当前设置的TensorFlow随机种子只会影响当前的默认图。

@ For example, you have two models a.py and b.py that have identical code:
```python
import  tensorflow  as  tf
tf.set_random_seed(2)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)

with tf.Session() as sess:
    print sess.run(c)
    print sess.run(d)
```
@  Without graph level seed, running python a.py and b.py will return 2 completely different results, but with tf.set_random_seed, you will get two identical results.

没有图级种子，运行python a.py和b.py将返回两个完全不同的结果，如果使用tf.set随机数种子，你将得到两个相同的结果。

```
$ python a.py
>> -4.00752
>> -2.98339

$ python b.py
>> -4.00752
>> -2.98339
```

## 从TensorFlow中读取数据
@  There are two main ways to load data into a TensorFlow graph: one is through feed_dict that we are familiar with, and another is through readers that allow us to read tensors directly from file. There is, of course, the third way which is to load in your data using constants, but you should only use this if you want your graph to be seriously bloated and un-runnable (I made up another word but you know what I mean).
将数据加载到TensorFlow图中有两种主要方式:一种是通过我们所熟悉的feed_dict，另一种是通过读取器使我们能够直接从文件中读取张量。当然，第三种方法是使用常量来加载数据，但是，如果您想让图变得非常臃肿和不可运行(我编了另一个词，但您知道我的意思)，那么您就应该使用这个方法。

@ To see why we need something more than feed_dict, we need to look into how feed_dict works under the hood. Feed_dict will first send data from the storage system to the client, and then from client to the worker process. This will cause the data to slow down, especially if the client is on a different machine from the worker process. TensorFlow has readers that allow us to load data directly into the worker process.
要想知道为什么feed_dict有什么不足，我们还需要研究一下，在这个系统的作用下，它是如何工作的。Feed_dict将首先从存储系统向客户端发送数据，然后从客户端发送到工作流程。这将导致运算慢下来，特别是当客户端与工人进程处于不同的机器上时。TensorFlow允许我们将数据直接加载到工作流程中。

@ The improvement will not be noticeable when we aren’t on a distributed system or when our dataset is small, but it’s still something worth looking into. TensorFlow has several built in readers to match your reading needs.
当我们不在一个分布式系统或者数据集很小的时候，这个改进是不足为道的，但是它仍然值得我们去研究。TensorFlow有几个内置在输入接口，以满足你的读取数据的需求。

```
tf.TextLineReader
输出由换行符分隔数据的文件
E.g. text files, CSV files

tf.FixedLengthRecordReader
当文件有相同的固定长度时，输出整个文件
E.g. each MNIST file has 28 x 28 pixels, CIFAR-10 32 x 32 x 3

tf.WholeFileReader
输出全部文件内容

tf.TFRecordReader
用TensorFlow自己的文件读取方式读取数据

tf.ReaderBase
允许你创建自己的输入接口
```

@ Data can be read in as individual data examples or in batches of examples.
数据可以作为单独的数据示例或成批示例读取。
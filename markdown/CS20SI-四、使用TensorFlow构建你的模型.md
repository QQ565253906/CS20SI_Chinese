# 四、使用TensorFlow构建你的模型
> 内容由Chip Huye编写
> 中文翻译：xinjiyuan97，校对：
> 原内容[下载地址](http://web.stanford.edu/class/cs20si/lectures/notes_04.pdf)

@ Up until this point, we’ve implemented two simple models in TensorFlow: linear regression on the number of fire and theft in the city of Chicago, and logistic regression to do an Optical Character Recognition task on the MNIST dataset. With the tools we have, we can definitely build more complicated models. However, complex models would require better planning, otherwise our models would be pretty messy and hard to debug. In the next two lectures, we will discuss a way to efficiently structure our models. And we will be doing that through an example: word2vec.
到目前为止，我们已经在TensorFlow中实现了两个简单的模型:在芝加哥城市的火灾和盗窃数量关系的线性回归，以及在MNIST数据集上进行逻辑回归。有了这些工具，我们就可以构建更复杂的模型。然而，复杂的模型需要更好的计划，否则我们的模型将会非常混乱并且难于调试。在接下来的两节课中，我们将讨论如何有效地构造模型。我们将通过一个例子来实现这一点:word2vec。

@ I expect that most of you are already familiar with word embedding and understand the importance of a model like word2vec. For those who aren’t familiar with this, you can read the CS 224N lecture slide about the motivation for and explanation of word2vec at  Simple Word Vector Representations .
我希望你们大多数人已经熟悉词嵌入和理解像word2vec这样的模型的重要性。对于那些不熟悉这一点的人，你可以回顾一下CS224N的关于单词向量表示的目的的[课程幻灯片](http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture2.pdf)。

@ The original papers by Mikolov et al.
论文原作者是Mikolov等人。
[Distributed Representations of Words and Phrases and their Compositionality Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

@ In short, we need a vector representation of words so that we can input them into our neural networks to do some magic tricks. Word vectors form the basis of many models studied in CS224N, as well as most language models in real life.
简而言之，我们需要将一个单词用矢量表示，这样我们就可以把它们输入到我们的神经网络中去做一些运算。词向量构成了CS224N研究的许多模型的基础，以及现实生活中的大多数语言模型。

@ Skip - gram vs CBOW   ( Continuous   Bag - of - Words)
@ Algorithmically ,  these models are similar ,   except  that CBOW predicts center words  from context words ,   while  the skip - gram does the inverse  and  predicts source context - words  from the center words .   For  example ,   if  we have the sentence :   "" The  quick brown fox jumps "" ,   then CBOW tries to predict  "" brown ""   from   "" the "" ,   "" quick "" ,   "" fox "" ,   and   "" jumps "" ,   while skip - gram tries to predict  "" the "" ,   "" quick "" ,   "" fox "" ,   and   "" jumps ""   from   "" brown "".
@ Statistically  it has the effect that CBOW smoothes over a lot of the distributional information  ( by  treating an entire context  as  one observation ).   For  the most part ,   this turns  out  to be a useful thing  for  smaller datasets .   However ,  skip - gram treats each context - target pair  as  a  new  observation ,   and   this  tends to  do  better  when  we have larger datasets.

@ ** 需要专业NLP人士 **

> **可以跳过 ———— gram模型 vs CBOW模型
> 从算法上来说，这些模型是相似的，除了CBOW可以预测来自上下文单词的中心词，而跳跃表则是相反的，从中心词来预测整个句子的内容。举例来说，如果有一个句子：""The  quick brown fox jumps""，CBOW模型会从""the"", ""quick"", ""fox"", 和""jumps""中预测""brown""，而skip-gram模型会从""brown""中预测""the"", ""quick"", ""fox"", 和""jumps""。
> 从统计学上来说，它的作用是，CBOW将许多分布信息进行平滑处理(通过将整个语境作为一种观察)。在大多数情况下，这对于较小的数据集来说是一件有用的事情。而skip-gram通过一个新的观察角度将每一个上下文-目标对作为一个，当我们有更大的数据集时，它会做得更好。

@ Vector representations of words projected on a 3D space.
在空间中单词向量的表示。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/1.png)

@ In this lecture, we will try to build word2vec, the skip-gram model. You can find an explanation/tutorial to the skip-gram model here.

在这节课中，我们将尝试构建word2vec，即skipgram模型。您可以在这里找到一个skipgram模型的教程。
[Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

@ In the skip-gram model, to get the vector representations of words, we train a simple neural network with a single hidden layer to perform a certain task, but then we don’t use that neural network for the task we trained it on. Instead, we care about the weights of the hidden layer. These weights are actually the “word vectors”, or “embedding matrix” that we’re trying to learn.

在skipgram模型中，我们通过训练一个简单的神经网络，得到单词的矢量表示，其中有一个隐藏层来执行某项任务，但是我们不使用这个神经网络来完成我们训练它的任务。相反，我们关心的是隐藏层的权重。这些权重实际上是我们要学习的“词向量”或“嵌入矩阵”。

@ The certain, fake task we’re going to train our model on is predicting the neighboring words given the center word. Given a specific word in a sentence (the center word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.
我们要训练我们的模型的特定的，假的任务是预测相邻词的中心词。给定一个句子中的一个特定单词(中心词)，看看附近的单词，然后随机选择一个单词。这个网络将告诉我们，我们所选择的“附近单词”中的每一个单词的概率。

### Softmax，负采样，和噪声对比估计
@ In CS 224N, we learned about the two training methods: hierarchical softmax and negative sampling. We ruled out softmax because the normalization factor is too computationally expensive, and the students in CS 224N implemented the skip-gram model with negative sampling.
在CS224N中，我们学习了两种训练方法:分层Softmax和负采样。我们排除了softmax，因为标准化因子的计算太昂贵，而CS 224N中学生用负采样实现了skip-gram模型。

@ Negative sampling, as the name suggests, belongs to the family of sampling-based approaches. This family also includes importance sampling and target sampling. Negative sampling is actually a simplified model of an approach called Noise Contrastive Estimation (NCE), e.g. negative sampling makes certain assumption about the number of noise samples to generate (k) and the distribution of noise samples (Q) (negative sampling assumes that kQ(w) = 1) to simplify computation (read Sebastian Rudder’s “O  n word embeddings - Part 2: Approximating the Softmax ” and Chris Dyer’s “ Notes on Noise Contrastive Estimation and Negative Sampling ”). Mikolov et al. have shown in their paper “D  istributed Representations of Words and Phrases and their Compositionality”  that training the Skip-gram model that results in faster training and better vector representations for frequent words, compared to more complex hierarchical softmax.
正如名字所写的，负采样属于基于采样的方法。这一类方法还包括重要的抽样和目标抽样。负采样实际上是噪声对比评估方法的一个简化模型,如负采样使得某些假设噪声样本的数量生成(k)和噪声样本的分布(Q)(负采样(w)= 1)假定kQ简化计算(Sebastian舵的“O n字嵌入-第2部分:近似将Softmax”和克里斯·戴尔的“笔记噪音对比估计和负采样”)。Mikolov等人在他们的论文中讨论了“对词语和短语的描述，以及他们的组合性”，这是训练Skip-gram模型的结果，这一模型对频繁的词汇进行了更快的训练和更好的矢量表达，而不是更复杂的层次性softmax回归。

@ While negative sampling is useful for the learning word embeddings, it doesn’t have the theoretical guarantee that its derivative tends towards the gradient of the softmax function, which makes it not so useful for language modelling.
虽然负采样对学习词嵌入很有用，但它并没有理论保证它的导数趋向于Softmax函数的梯度，这使得它对语言建模没有太大的用处。

@ NCE has this nice theoretical guarantees that negative sampling lacks as the number of noise samples increases.  Mnih and Teh (2012)  reported that 25 noise samples are sufficient to match the performance of the regular softmax, with an expected speed-up factor of about 45.
NCE有一个很好的理论保证，即随着噪声样本数量的增加，负采样的数量也会下降。Mnih和Teh在(2012)的报告说，25个噪音样本足以与常规的软件的性能匹配，预计将会增加约45个。

@ In this example, we will be using NCE because of its nice theoretical guarantee.
在本例中，我们将使用NCE，因为它有很好的理论保证。

@ Note that sampling-based approaches, whether it’s negative sampling or NCE, are only useful at training time -- during inference, the full softmax still needs to be computed to obtain a normalized probability.
请注意，基于采样的方法，无论是负的采样还是NCE，只在训练时有用——在推理过程中，仍然需要计算完整的softmax，以获得一个标准化的概率。

### 关于数据集
@ 100MB is not enough to train really good word embeddings, but enough to see some interesting relations. There are 17,005,207 tokens by simple splitting the text by blank space using split() function of python strings.
100MB不足以真正训练好词嵌入，但足以让我们看到一些有趣的关系。通过使用python字符串的split()函数简单地将文本分割为空白空间，就可以得到17,005,207个词。

@ For better results, you should use the dataset fil9 of the first 10^9 bytes of the Wikipedia dump, as described on  Matt Mahoney’s website. 
为了更好的结果，你应该使用维基百科上的前$ 10^9 $个字节的数据集，就像Matt Mahoney的网站上所描述的那样。

## 如何设计你的TensorFlow模型

@ We’ve done only 2 models in the past, and they more or less have the same structure:
我们过去只做过两种模型，它们或多或少具有相同的结构:

@ Phase 1: assemble your graph
@ 1. Define placeholders for input and output 
@ 2. Define the weights
@ 3. Define the inference model
@ 4. Define loss function
@ 5. Define optimizer

阶段1:组装你的图表
1. 定义输入和输出的占位符
2. 定义权重
3. 定义推理模型
4. 定义损失函数
5. 定义优化器

@ Phase 2: execute the computation
@ Which is basically training your model. There are a few steps:
@ 1. Initialize all model variables for the first time.
@ 2. Feed in the training data. Might involve randomizing the order of data samples.
@ 3. Execute the inference model on the training data, so it calculates for each training input example the output with the current model parameters.
@ 4. Compute the cost
@ 5. Adjust the model parameters to minimize/maximize the cost depending on the model.

阶段2:执行计算
这基本上就是训练你的模型。这里有几个步骤:
1. 第一次初始化所有的模型变量。
2. 提供训练数据。包括随机化数据样本的顺序。
3. 在训练数据上执行推理模型，以便为每个培训输入示例输出当前参数模型的输出。
4. 计算出成本
5. 根据模型调整模型参数以最小化/最大化成本。

@ Here is a visualization of training loop from the book “TensorFlow for Machine Intelligence”:
下面是《TensorFlow for Machine Intelligence》中训练的可视化图。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/2.png)

@ Let’s apply these steps to creating our word2vec, skip-gram model.
让我们应用这些步骤来创建我们的word2vec，skipgram模型。

## 阶段1:组装你的图表
### 定义输入和输出的占位符
Input is the center word and output is the target (context) word. Instead of using one-hot vectors, we input the index of those words directly. For example, if the center word is the 1001th word in the vocabulary, we input the number 1001.
输入是中心词，输出是目标(上下文)词。我们不使用一个？？热？？的向量，而是直接输入这些词的索引。例如，如果中心词是词汇表中的第1001个单词，我们输入数字1001。

@ Each sample input is a scalar, the placeholder for BATCH_SIZE sample inputs with have shape [BATCH_SIZE].
@ Similar, the placeholder for BATCH_SIZE sample outputs with have shape [BATCH_SIZE].

每个样例输入都是一个标量，对于具有每一批数据有一个宽度为BATCH_SIZE的占位符。
同样的，对于对于每一批数据，输出数据有一宽度为BATCH_SIZE的占位符。

```python
center_words = tf.placeholder(tf.int32, shape = [BATCH_SIZE]) 
target_words = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
```

@ Note that our center_words and target_words being fed in are both scalars -- we feed in their corresponding indices in our vocabulary.
请注意，我们的中心词和目标词都是标量——我们在词汇表中输入它们对应的索引。

### 定义权重 (在这种情况下，指嵌入矩阵)
@ Each row corresponds to the representation vector of one word. If one word is represented with a vector of size EMBED_SIZE, then the embedding matrix will have shape [VOCAB_SIZE, EMBED_SIZE]. We initialize the embedding matrix to value from a random distribution. In this case, let’s choose uniform distribution.
每一行对应一个词的表示向量。如果一个词用一个大小嵌入的向量表示，嵌入矩阵由vocabsize大小的向量构成。我们将嵌入矩阵从一个随机分布中初始化。在这种情况下，让我们选择均匀分布。

```python
embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0 ))
```

### 定义推理模型 (计算出图形的前向路径)
@ Our goal is to get the vector representations of words in our dictionary. Remember that the embed_matrix has dimension VOCAB_SIZE x EMBED_SIZE, with each row of the embedding matrix corresponds to the vector representation of the word at that index. So to get the representation of all the center words in the batch, we get the slice of all corresponding rows in the embedding matrix. TensorFlow provides a convenient method to do so called tf.nn.embedding_lookup().
我们的目标是在字典中获取单词的向量表示。记住，嵌入矩阵的大小是VOCAB_SIZE x EMBED_SIZE，而嵌入矩阵的每一行对应于该索引中单词的向量表示。为了得到批处理中的所有中心词的向量表示，我们就需要得到嵌入矩阵中所有相应行的切片。TensorFlow提供了一种方便的方法，称为 `tf.nn.embedding_lookup()`。

```python
tf.nn.embedding_lookup(params, ids, partition_strategy = 'mod', name = None, validate_indices = True, max_norm = None)
```

@ This method is really useful when it comes to matrix multiplication with one-hot vectors because it saves us from doing a bunch of unnecessary computation that will return 0 anyway. An illustration from  Chris McCormick  for multiplication of a one-hot vector with a matrix.
这个方法在使用??单热??向量的矩阵乘法时非常有用，因为它可以避免我们做很多不必要的计算，而这些计算将返回0。Chris McCormick在演示中用了一个矩阵的一个热向量做乘法的例子。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/3.png)

@ So, to get the embedding (or vector representation) of the input center words, we use this:
因此，为了得到输入中心词的嵌入(或矢量表示)，我们使用如下:

```python
embed = tf.nn.embedding_lookup(embed_matrix, center_words)
```

### 定义损失函数
@ While NCE is cumbersome to implement in pure Python, TensorFlow already implemented it for us.
虽然NCE在纯Python中实现起来很麻烦，但TensorFlow已经为我们实现了它。

```python
tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true = 1, sampled_values = None, remove_accidental_hits = False, partition_strategy = 'mod', name = 'nce_loss')
```

@ Note that by the way the function is implemented, the third argument is actually inputs, and the fourth is labels. This ambiguity can be quite troubling sometimes, but keep in mind that TensorFlow is still new and growing and therefore might not be perfect. Nce_loss source code can be found  here .
注意，通过实现函数的方式，第三个参数实际上是输入，而第四个参数是标签。这种模棱两可有时会让人感到不安，但请记住，TensorFlow仍然是新兴的、正在发展的，因此可能并不完美。可以在这里找到[nce历史代码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py)。

@ For nce_loss, we need weights and biases for the hidden layer to calculate NCE loss.
对于nce的损失函数，我们需要对隐藏层的权重和偏差来计算NCE损失。

```python
nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE,  EMBED_SIZE], \
                        nce_bias = tf.Variable(tf.zeros([ VOCAB_SIZE ])), \
                        stddev = 1.0 / EMBED_SIZE ** 0.5))
```

@ Then we define loss:
```python
loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, \
                                       biases = nce_bias, \
                                    labels = target_words, \
                                    inputs = embed, \
                                    num_sampled = NUM_SAMPLED, \
                                    um_classes = VOCAB_SIZE ))
```

### 定义优化器
@ We will use the good old gradient descent.
我们使用传统的梯度下降优化器。

```python
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```

## 阶段2:执行计算
@ We will create a session then within the session, use the good old feed_dict to feed inputs and outputs into the placeholders, run the optimizer to minimize the loss, and fetch the loss value to report back to us.
我们将创建一个会话，使用feed_dict来替换输入和输出输入占位符，运行优化器以最小化损失，并获取损失值报告回给我们。

```python
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    average_loss = 0.0
    for index in xrange(NUM_TRAIN_STEPS): 
        batch = batch_gen.next() 
        loss_batch, _ = sess.run([loss, optimizer], feed_dict = {center_words: batch[0], target_words: batch[1]}) 
        average_loss += loss_batch
    if (index + 1) % 2000 == 0: 
        print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / (index + 1)))
```

@ You can see the full basic model on the class’s GitHub repo under the name word2vec_no_frills.py
您可以在GitHub上看到完整的基本模型，名称是word2vec_no_frills.py。

@ As you can see, the whole model takes less than 20 lines of code. If you’ve implemented word2vec without TensorFlow (as for the assignment 1 for CS224N), we know that this is really short. We’ve pretty much dumped everything into one giant function.
如您所见，整个模型只需要不到20行代码。如果您已经实现了没有TensorFlow的word2vec(对于CS224N的任务1)，我们知道这是非常短的。我们几乎把所有东西都扔进了一个巨大的函数里。

### 命名空间
@ Let’s give the tensors name and see how our model looks like in TensorBoard.
让我们给出张量的名字，看看我们的模型在TensorBoard上是怎样的。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/4.png)

@ This doesn’t look very readable, as you can see in the graph, the nodes are scattering all over. TensorBoard doesn’t know which nodes are similar to which nodes and should be grouped together. This setback can grow to be extremely daunting when you build complex models with hundreds of ops.
这看起来不太容易理解，正如你在图中所看到的，节点是分散的。TensorBoard不知道哪些节点与哪些节点相似，应该将哪些节点组合在一起。当您用数百个操作来构建复杂的模型时，这个过程就会变得非常可怕。

@ Then, how can we tell TensorBoard to know which nodes should be grouped together? For example, we would like to group all ops related to input/output together, and all ops related to NCE loss together. Thankfully, TensorFlow lets us do that with name scope. You can just put all the ops that you want to group together under the block:
那么，我们如何让TensorBoard知道应该将哪些节点组合在一起呢?例如，我们希望将所有与输入/输出相关的操作，以及所有与NCE损失有关的操作放在一起。幸运的是，TensorFlow允许我们使用名称范围来实现这一点。你可以把你想要的所有的操作放在一起:

```python
with tf.name_scope(name_of_that_scope):
    # declare op_1
    # declare op_2
    # ...
```
@ For example, our graph can have 3 op blocks: “Data”, “embed”, and “NCE_LOSS” like this:
例如，我们的图可以有3个op块:“数据”、“嵌入”和“nce损失”:

```python
with tf.name_scope('data'):
    center_words = tf.placeholder(tf.int32, shape = [BATCH_SIZE], name = 'center_words') 
    target_words = tf.placeholder(tf.int32, shape = [BATCH_SIZE, 1], name = 'target_words')

with tf.name_scope('embed'): 
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name = 'embed_matrix') 

with tf.name_scope('loss'): 
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name = 'embed') 
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev = 1.0 / math.sqrt(EMBED_SIZE)), name = 'nce_weight') 
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name = 'nce_bias') 
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, \
                        biases = nce_bias, labels = target_words, inputs = embed, \
                        num_sampled = NUM_SAMPLED, num_classes = VOCAB_SIZE), name = 'loss') 
```
@ It seems like the namescope ‘embed’ has only one node and therefore it is useless to put it in a separate namescope. It, in fact, has two nodes: one for the tf.Variable and one for tf.random_uniform.
似乎“名称空间”的“embed”只有一个节点，因此把它放在一个单独的名称空间中是没有用的。实际上，它有两个节点:`tf.Variable`和`tf.random_uniform`。

@ When you visualize that on TensorBoard, you will see your nodes are grouped into neat blocks:
当你在TensorBoard上看到它时，你会看到你的节点被分组成整齐的块:

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/5.png)

@ You can click on the plus sign on top of each name scope block to see all the ops inside that block. I love graphs so I find this visualization fascinating. Take your time to play around with it.
您可以单击每个name范围块顶部的加号，以查看该块内的所有操作。我喜欢图表，所以我觉得这个可视化很吸引人。请多花一点时间研究它。

@ You’ve probably noticed that TensorBoard has two kinds of edges: the solid lines and the dotted lines. The solid lines represent data flow edges. For example, the value of op tf.add(x + y) depends on the value of x and y. The dotted arrows represent control dependence edges. For example, a variable can only be used after being initialized, as you see variable embed_matrix depends on the op init). Control dependencies can also be declared using tf.Graph.control_dependencies(control_inputs) we talked about in lecture 2.
你可能已经注意到，TensorBoard有两种边缘:实线和虚线。实线表示数据流的边。例如，操作符tf.add(x + y)依赖于x和y的值，虚线箭头表示控制依赖边。例如，一个变量只能在初始化后使用，因为您看到变量嵌入矩阵依赖于操作符init()。控制依赖也可以使用tf.Graph.control_dependencies(control_inputs)来声明，我们在第2讲中已经讨论过了。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/6.png)

@ Here is the full legend of nodes in TensorBoard:
以下是TensorBoard中的节点的完整说明:

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/7.png)

@ So now, our whole word2vec program looks more or less like this:
现在，我们的word2vec程序看起来是这样的:

```python
# Step 1: 定义输入和输出的占位符
with tf.name_scope("data"):
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name = 'center_words') 
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name = 'target_words')

# 在CPU上组装图的这一部分。如果你有GPU，你可以把它改到GPU上
with tf.device('/cpu:0' ):
    with tf.name_scope("embed"):
        # Step 2: 定义权重. 在word2vec中, 我们其实只关心权重
        embed_matrix = tf.Vaniable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name = 'embed_matrix')
    # Step 3+4: 定义模型和损失函数
    with tf.name_scope("loss"):
        
        # Step 3: 定义模型
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        
        # Step 4: 定义NCE的损失函数
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], \
                                stddev=1.0 / math.sqnt(EMBED_SIZE)), name='nce_weight')
        nce_bias = tf.Vaniable(tf.zenos([VOCAB_SIZE]), name='nce_bias')

        # 定义NCE的损失函数
        loss = tf.neduce_mean(tf.nn.nce_loss(weights = nce_weight, 
                                biases = nce_bias, labels = target_words, inputs = embed,
                                num_sampled = NUM_SAMPLED, num_classes = VOCAB_SIZE), name = 'loss')
    # Step 5: 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```

@ If you’ve taken any CS106 class you’ll know that this program will get check minus on styles because “whatever happened to decomposition?” You can’t just dump everything into a giant function. Also, after we’ve spent an ungodly amount of time building a model, we’d like to use it more than once.
如果你学过CS106课程，你会知道这个程序应该如何被检查，因为“不管怎么分解?”“你不能把所有的东西都变成一个巨大的功能。而且，在我们花费了大量的时间建立一个模型之后，我们想要不止一次地使用它。

@ Question: how do we make our model most easy to reuse?
@ Hint: take advantage of Python’s object-oriented-ness. 
@ Answer: build our model as a class!

问题:如何使我们的模型易于重用?
提示:利用Python的对象-定向性。
回答:将我们的模型作为一个类构建。


@ Our class should follow the interface. We combined step 3 and 4 because we want to put embed under the name scope of “NCE loss”.
我们的类应该遵循这个接口。我们将第3步和第4步组合在一起，因为我们希望将词嵌入置于“NCE损失函数”的名称范围内。

```python
class SkipGramModel:          
 def  _create_optimizer ( self ):
    """ 创建图 """
    def __init__(self, params):
        pass

    def _create_placeholders(self):  
        """ Step 1: 定义输入和输出的占位符 """
        pass

    def _create_embedding(self):
    """ Step 2: 定义权重 """
        pass
    def _create_loss(self):
    """ Step 3 + 4: 定义推理模型 + 定义损失函数 """
        pass

    def _create_optimizer(self):
    """ Step 5: 定义优化器 """  
        pass
```

@ 在100000轮迭代后，损失下降到10.0

@ Now let’s see what our model finds after training it for 100,000 epochs.
现在让我们看看我们的模型在训练了10万次迭代之后发现了什么。

@ If we visualize our embedding with t-SNE we will see something like below. It’s hard to visualize in 2D, but we’ll see in class in 3D that all the number (one, two, ..., zero) are grouped in a line on the bottom right, next to all the alphabet (a, b, ..., z) and names (john, james, david, and such). All the months are grouped together. “Do”, “does”, “did” are also grouped together and so on.
如果我们把词嵌入模型t-sne的想象出来，我们会看到下面的东西。在二维中很难想象，但是我们会在3D中看到所有的数字(一，二，……0)被分组在右下方的一条线上，在所有字母表的旁边(a, b, ..., z)和名字(john, james, david, and such)。所有的月份都被分组在一起。“Do”, “does”, “did”也被分组在一起，等等。

!()[https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/8.png]

@ If you print out the closest words to ‘american’, you will find its closest cosine neighbors are ‘british’ and ‘english’. Fair enough.
如果你把最接近‘american’的单词打印出来，你会发现它最接近的词是‘british’和‘english’。很好。

!()[https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/9.png]
@ How about words closest to ‘government’?
那么和‘government’最接近的词呢？

!()[https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/10.png]

@ t-SNE   ( from   Wikipedia)

@ t-distributed stochastic neighbor embedding (t-SNE) is a machine learning algorithm for dimensionality reduction developed by Geoffrey Hinton and Laurens van der Maaten. It is a nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions^ which can then be visualized in a scatter plot. Specifically^ it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points.

@ The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects have a high probability of being picked, whilst dissimilar points have an extremely small probability of being picked. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback-Leibler divergence between the two distributions with respect to the locations of the points in the map. Note that whilst the original algorithm uses the  Euclidean  distance between objects  as  the  base of its similarity metric ,   this  should be changed  as  appropriate.

> t-SNE(引维基百科)
> t-分布式随机邻居词嵌入(t-sne)是Geoffrey Hinton和Laurens开发的一种机器学习算法。它是一种非线性的维数缩减技术，特别适合于将高维数据嵌入二维或三维空间中，然后在散点图中显示出来。具体地说，它通过一个二维或三维的点来模拟每个高维度的物体，使相似的物体被邻近的点所建模，而不同的物体则被远处的点所建模。
> t-sne算法由两个主要阶段组成。首先，t-sne构造了一种概率分布，通过对高维度物体的概率分布，类似的物体被选中的概率很高，而差异较大的点被选中的概率非常小。其次，t-sne在低维度地图上定义了一个类似的概率分布，它将两种分布在地图上点的位置上的Kullback-Leibler的散度最小化。注意，虽然原始算法使用了对象之间的欧氏距离作为其相似性度量的基础，但这应该是适当的更改。


@ If you haven’t used t-SNE, you should start using it! It’s super cool. Have you read Chris Olah’s blog post about  visualizing MNIST ? t-SNE made MNIST cool! Image below is from Olah’s blog. You should head to his blog for the interactive version.
如果你没有使用过t-sne，你应该开始使用它！这是超级酷。你读过Chris Olah的关于MNIST的[可视化文章](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)?t-SNE使MNIST酷!下面的图片来自Olah的博客。你应该到他的博客去学习一下。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/11.png)

@ We can also visualize our embeddings using PCA too.
我们也可以用PCA来可视化我们的嵌入。

![](https://github.com/xinjiyuan97/CS20SI_Chinese/blob/note4/markdown/img/note4/12.png)

@ And I did all that visualization with less than 10 lines of code. TensorBoard provided a wonderful tool for doing so. Warning: the TensorFlow official guide is a bit ambiguous so you should follow this guide.
我用不到10行代码做了所有的可视化工作。TensorBoard提供了一个非常棒的工具。警告:TensorFlow的官方指南有点模糊，所以你应该遵循这个指南。

@ There are several steps.
有几个步骤

```python
from tensorflow.contrib.tensorboard.plugins import projector
# 在训练后获得嵌入矩阵
final_embed_matrix  =  sess . run ( model . embed_matrix)
# 创建一个变量用于存储嵌入矩阵。必须是一个变量，而不能是一个常量。
# 而且不能是你之前定义的嵌入矩阵。
# 我也不知道为什么是这个样子。我获得了500个最火的单词。
sess.run(embedding_var.initializer)
embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter(LOGDIR)

# 添加嵌入的配置
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# 将嵌入矩阵链接到他们的元数据文件中。
# 在本例中，包含500个最流行单词的文件在我们的词汇表中
embedding.metadata_path = LOGDIR + '/vocab_500.tsv'
# 保存成一个TensorBoard在启动时将读取的配置文件
projector.visualize_embeddings(summary_writer, config)
# 保存嵌入矩阵
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, LOGDIR + '/skip-gram.ckpt', 1)
```

@ Now we run our model again, then again run tensorboard. If you go to http://localhost:6006, click on the Embeddings tab, you’ll see all the visualization.
现在我们再次运行我们的模型，然后再次运行tensorboard。如果您访问http://localhost:6006，单击嵌入式选项卡，您将看到所有的可视化。

@ Cool, huh?

@ You can visualize more than word embeddings, aka, you can visualize any embeddings.
你可以想象更多的信息，也就是说，你可以想象任何的嵌入。

## 我们为什么还需要学习梯度下降法？

@ You’ve probably noticed that in all the models we’ve built so far, we haven’t taken a single gradient. All we need to do is to build a forward pass and TensorFlow takes care of the backward path for us. So, the question is: why should we still learn to take gradient? Why are Chris Manning and Richard Socher making us take gradients of cross entropy and softmax? Shouldn’t taking gradients by hands one day be as obsolete as trying to take square root by hands since the invention of calculator?
您可能已经注意到，在我们构建的所有模型中，我们都没有采用单一的梯度下降。我们所需要做的就是建立一个向前的模型，而TensorFlow则负责为我们提供反向的路径。所以，问题是:我们为什么还要学习梯度法呢?为什么Chris Manning和Richard Socher让我们采用了交叉熵和softmax的梯度?在计算器发明后，真的还有必要花费一天的时间去计算梯度吗？

@ Well, maybe. But for now, TensorFlow can take gradients for us, but it can’t give us intuition about what functions to use. It doesn’t tell us if a function will suffer from exploding or vanishing gradients. We still need to know about gradients to get an understanding of why a model works while another doesn’t.
嗯,也许吧。但现在，TensorFlow可以为我们取梯度，但它不能让我们直观地知道要使用什么函数。它没有告诉我们一个函数是否会受到爆炸或消失的梯度的影响。我们仍然需要了解梯度，以了解为什么模型可以工作，而另一个不能。

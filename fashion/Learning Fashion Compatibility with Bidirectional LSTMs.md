

论文：《**Learning Fashion Compatibility with Bidirectional LSTMs**》

论文地址：https://arxiv.org/abs/1707.05691

代码地址：https://github.com/xthan/polyvore





## 论文笔记

### 1. 简介

时尚搭配推荐的需求越来越大，本文是基于两个方面的时尚推荐：

1. 给定已有的服饰，推荐一件空缺的衣服，从而形成一套搭配，即在已经有上衣、裤子的情况下推荐一双鞋子；
2. 根据多种形式的输入，比如文本或者一件衣服图片，生成一套搭配；

目前存在的难点在于：**如何对不同时尚类别的单品通过简单计算视觉相似性来建模和推理其匹配性关系**。

目前相关的工作，大部分主要集中在对服饰解析、服饰识别和服饰搜索这三个方向的工作，而对于少量做服饰推荐的工作，也存在这些问题：

1. 没有考虑做一套搭配的推荐；
2. 只能支持上述两个方向的其中一种，即要不只是推荐一套搭配或者对已有的搭配推荐缺失的一件衣服；
3. 目前还没有工作可以支持多种形式的输入，比如可以输入关键词，或者输入图片，或者图片+关键词的输入形式；

对于一套合适的搭配，如下图所示，本文认为应该满足这两个关键属性：

1. 这套搭配中的任意一件服饰应该是视觉上匹配并且是形似风格的；
2. 搭配不能存在重复类型的服饰，比如包含两双鞋子或者两条裤子；

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig3.png" style="zoom:80%;" />

目前相关在做搭配推荐的工作中，主要尝试的方法有：

1. 利用语义属性，即规定了哪些衣服之间是匹配的，但是这种需要对数据进行标记，代价很大而且没办法大量使用；
2. 利用度量学习来学习一对时尚单品之间的距离，但是这只能学习两两之间的匹配性，而非一套搭配；
3. 对于 2 的改进是采用投票策略，但是同样是计算代价非常的大，而且也没办法利用好集合中所有单品的一致性；



为了解决上述的问题，本文提出了通过一个端到端的框架来联合学习视觉语义向量(visual-semantic embedding) 和服饰物品之间的匹配性关系，下图就是本文的整体框架。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig2.png)

首先是利用 Inception-V3 模型作为特征提取器，将输入图片转成特征向量，然后采用一层共 512 个隐藏单元的双向 LSTM(Bi-LSTM)。之所以采用双向 LSTM，是因为作者认为可以将一套搭配当做是一个特定顺序的序列，搭配中的每件衣服就是一个时间点(time step)。在每个时间点，Bi-LSTM 模型将根据之前的图片来预测下一张图片。

另外，本文的方法还通过将图片特征映射到一个语义表示来学习一个视觉语义向量，它不仅提供了语义属性和类型信息作为训练 LSTM 的输入和正则化方法，还可以实现对用户的多种形式输入来生成一套搭配。

训练好模型后，本文通过三个任务来评估模型，如下图所示，分别是：

1. Fill in the blank：给定一套缺失某件衣服的搭配，然后给定四个选择，让模型选择最匹配当前搭配的服饰单品；
2. 搭配生成：根据多种输入来生成一套搭配，比如文本输入或者一张服饰图片；
3. 匹配性的预测：给定一套搭配，给出其匹配性得分。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig1.png)

### 2. Polyvore数据集

本文实验采用的是一个叫做 Polyvore 的数据集。Polyvore 是一个有名的流行时尚网站，用户可以在网站里创建和上次搭配数据，这些搭配包含了很丰富的多种形式的信息，比如图片和对单品的描述、对该搭配的喜欢数、搭配的一些哈希标签。

Polyvore 数据集总共有 21889 套搭配，本文将其划分成训练集、验证集和测试集，分别是 17316、1497 和 3076 套。

这里参考了论文《 Mining Fashion Outfit Composition Using An End-to-End Deep Learning Approach》，使用一个中图分割算法保证训练集、验证集和测试集不存在重复的衣服，另外对于包含太多单件的搭配，为了方便，仅保留前 8 件衣服，因此，数据集总共包含了 164,379个样本，每个样本是包含了图片以及其文本描述。

对于文本描述的清理，这里删除了出现次数少于 30 次的单词，并构建了一个 2757 个单词的词典。

另外 Polyvore 数据集的衣服是有一个固定的顺序的——一般是tops、bottoms、shoes 以及 accessories，对于 tops 的顺序也是固定的，一般是 shirts 或者 t-shirts，然后是外套，而 accessories 的顺序一般是手提包、帽子、眼镜、手表、项链、耳环等；

因此，这种固定顺序可以让 LSTM 模型学习到时间的信息。



### 3. 方法

#### 3.1 **Fashion Compatibility Learning with** Bi-LSTM

第一部分介绍的是基于双向 LSTM 的服饰匹配性学习。这主要是利用了 LSTM 模型的特性，**它们可以学习到两个时间点之间的关系，同时使用由不同细胞调节的记忆单元，有助于利用长期的时间依赖性**。

基于这个特性，本文将一套搭配看作一个序列，搭配中的每个图片就是一个独立的时间点，然后利用 LSTM 来对搭配的视觉搭配关系进行建模。

给定一套搭配，$F={x_1, x_2, \cdots, x_N}$，其中 $x_t$ 是搭配中第 t 个衣服经过 CNN 后，提取到的特征。在每个时间点，先采用前向的 LSTM 对给定的图片预测其下一张图片，这种做法学习两个时间点的关系相当于在学习两件衣服之间的匹配关系。

这里使用的 loss 函数如下所示：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig4.png)

其中 $\theta_f $表示前向预测模型中的模型参数，而 $P_r(\cdot)$ 是 LSTM 模型计算的，表示基于之前的输入来预测得到 $x_{t+1}$ 的概率。

更详细点说，LSTM 将输入映射到输出也是通过以下一系列的隐藏状态，计算公式如下：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig5.png)

其中 $x_t, h_t$ 分别表示输入和输出向量，其他的 $i_t, f_t, c_t, o_t$ 分别表示输入门、遗忘门、记忆单元和输出门的激活向量。

参考论文《 Recurrent neural network based language model》中使用 softmax 输出来预测一个句子中下一个单词，本文也是在 $h_t$ 之后增加了一个 softmax 层来计算下一件衣服出现的概率：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig6.png)

其中$X$ 表示的是当前 batch 的所有图片，这种做法可以让模型在看到多种样本后学习到具有区分度的风格和匹配性信息，另外，实际上可以让 $X$ 表示整个数据集，但是本文没有考虑这种做法的原因是数据集的数量太多以及图片特征的维度太大。只是限制在一个 batch 内可以提高训练的速度。

除了可以正向预测衣服，实际上也可以反向预测衣服，比如对于一条裤子，其下一件衣服可以是上衣或者是鞋子，所以可以有一个反向 LSTM ：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig7.png)

这里的 $P_r(\cdot)$ 计算方式如下：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig8.png)

注意，这里在 F 中会添加两个零向量 $x_0, x_{N+1}$ ，作用是让双向 LSTM 知道何时停止预测下一件衣服。

一般来说，一套搭配通常都是同种风格衣服的集合，即有相似的风格，比如颜色或者纹理，而本文的做法，将搭配当做一种固定顺序的序列来学习，即可以学到搭配的匹配性，也能学习到搭配的整体风格（主要是通过记忆单元来学习）。

#### 3.2 Visual-semantic Embedding

第二部分是视觉-语义向量的学习。

通常对于服饰推荐，都会有多种形式的输入，比如图片或者文本描述类型，所以很有必要学习一个文本和图片的多模态向量空间。

本文没有选择比较耗时耗力的人工标注图片的标签属性方式，而是采用弱标注的网络数据(weakly-labeled web data)，即数据集自带的每张图片的文本描述信息。根据这个信息，本文根据论文《Unifying visual semantic embeddings with multimodal neural language models》里常用的对图片-文本对构建模型的方法，将图片和其文本描述映射到一个联合空间，训练得到一个视觉-语义向量。

这里给出一些定义，令 $S={w_1,w_2,\cdots, w_M}$ 表示文本描述，其中 $w_i$ 表示单词，然后将其用一个 one-hot 向量 $e_i$ 进行表示，然后映射到向量空间 $v_i = W_T \cdot e_i$，其中 $W_T$ 表示词向量矩阵，所以最终这个文本描述的表示就是
$$
v=\frac{1}{M}\sum_i v_i
$$
 而图片向量表示为：
$$
f=W_I\cdot x
$$
在视觉-语言向量空间，本文会采用余弦相似性来评估图片和其文本描述的相似性：
$$
d(f,v) = f\cdot v
$$
其中 f 和 v 都会进行归一化，因此对于视觉-语义向量学习会用一个对比损失(contrastive loss) 进行优化：
$$
E_e(\theta_e)=\sum_f \sum_k max(0, m-d(f,v)+d(f,v_k)) + \sum_v\sum_k max(0, m-d(v,f)+d(v,f_k))
$$
其中 $\theta_e={W_I, W_T}$ 表示模型参数，$v_k$ 表示和 f 不匹配的文本描述，而$f_k$ 表示和 v 不匹配的图片。所以最小化上述 loss 来达到让图片向量 f 和其文本描述 v 的距离，比 f 到不匹配的文本描述 $v_k$ 的距离要更近一个间隔 m ，对 v 也是相同的实现效果。在训练过程中，同个 batch 中这种不匹配的样本例子是用于优化上述 loss，这种做法使得具有相似语义属性和风格的服饰在学习到的向量空间中距离更近。

#### 3.3 Joint Modeling

这里是将上述两种操作都联合起来，即同时学习搭配的匹配性和视觉-语义向量，因此整体的目标函数如下所示：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig9.png)

其中前两个就是双向 LSTM 的目标函数，第三个则是计算视觉语义向量 loss。整个模型的训练可以通过 Back-Propagation through time(BPTT) 来实现。对比标准的双向 LSTM，唯一的区别就是 CNN 模型的梯度是两个来源的平均值（即 LSTM 部分和视觉语义向量学习），这让 CNN 同时也可以学习到有用的语义信息。



### 4. 实验

#### 4.1 实现细节

- **双向 LSTM**：采用 InceptionV3 模型，输出 2048 维的 CNN 特征，然后经过一个全连接层，输出 512 维的特征，然后输入到 LSTM，LSTM 的隐藏单元数量是 512，设置 dropout 概率是 0.7；
- **视觉-语义向量**：联合向量空间的维度是 512 维，所以 $W_I$ 的维度是 $2048\times 512$ ，$W_T$ 的维度是 $2757\times 512$ ，2757 是字典的数量，然后令间隔 m=0.2
- **联合训练**：初始学习率是 0.2，然后衰减因子是 2，并且每 2 个 epoch 更新一次学习率；batch 大小是 10，即每个 batch 包含 10 套搭配序列，大概 65 张图片以及相应的文本描述，然后会微调网络的所有层，当验证集 loss 比较稳定的时候会停止训练。



#### 4.2 几种不同实验的结果

对于 fill in the blank，实验结果如下：

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig10.png)

一些好的例子和不好的例子：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig11.png" style="zoom:80%;" />

对于搭配匹配性预测，一些预测的例子如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig12.png" style="zoom:80%;" />

对于搭配生成，第一种实现是输入的只是衣服图片，如下所示：

- 给定单张图片，那么就如 a 图所示，同时执行两个方向的 LSTM，得到一套完整的搭配；
- 如果是给定多张图片，如 c 图所示，两张图片，那么会先进行一个类似 fill in the blank 的操作，先预测得到两张图片之间的衣服，然后如 d 所示，再进行预测其他位置的衣服，得到完整的搭配；

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig13.png)

如果是输入衣服和文本描述，如下所示：

这种输入的实现，首先会基于给定的衣服图片生成一套初始的搭配，然后对于给定的输入文本描述 $v_q$ ，在初始搭配中的非查询衣服 $f_i$ 会进行更新，更新方式为 $argmin_f d(f, f_i+v_q)$ ，所以更新后的衣服图片将不仅和原始衣服相似，还会在视觉语义向量空间里和输入的查询文本距离很接近。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig14.png)

或者是只输入文本描述，如下所示：

- 第一种场景，即前两行的图片例子，输入的文本描述是衣服的一种属性或者风格，首先最接近文本描述的衣服图片会被当做是查询图片，然后Bi-LSTM 将通过这张图片来生成一套搭配，接着是会基于给定的图片和文本输入更新搭配；
- 第二种场景，即后面两行图片例子，给定的文本描述是指向某种衣服类别，所以会根据文本描述检索相应的衣服图片，然后作为查询图片来生成搭配。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/Fashion_Compatibility_BiLstm_fig15.png)


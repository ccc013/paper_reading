# RRCF

论文：《Robust Random Cut Forest Based Anomaly Detection On Streams》

论文地址：https://d1.awsstatic.com/whitepapers/kinesis-anomaly-detection-on-streaming-data.pdf

Github 项目：

- python：https://github.com/kLabUM/rrcf
- java：https://github.com/aws/random-cut-forest-by-aws



## 介绍

本文对于动态数据流的异常检测问题提出了一种随机裁剪森林的算法，它是一种鲁棒性的随机裁剪数据结构，可以作为输入流的草图或者概要（sketch or synopsis）。

异常检测是数据挖掘中的一个核心问题，尽管过去几十年得到了很好的研究和发展，但随着互联网产生了大量的数据，这个问题需要重新考虑和更新解决方法。在目前的阶段，有两个核心问题：

1. 如何定义异常点？
2. 哪种数据结构可以高效的用于检测动态数据流中的异常点？

对于第一个问题，作者是这么看待的：

> 我们从**模型复杂性**的角度来看待这个问题，也就是说，**如果一个点的加入使模型的复杂性大大增加**，那么这个点就是一个异常点。

对于第二个问题，作者认为随机化的方法是很有用的，并且在有监督学习中非常有价值，这点和 Dropout 思想有点类似；

但它也存在一些问题，比如对于不相关的维度，会缺失重要的异常点，另外目前还不清楚如何将这个方法拓展到流式数据中，当然也有一些工作在研究这个，但是不够高效。

所以，本文为了处理上述的问题，提出了 RRCF （Robust Random Cut Forest）方法。



## 算法介绍

### 核心操作

首先是定义了如何构造 RRCT，而 RRCF 就是 RRCT 的集合：

给定数据点集 S 随机割树 **RRCT** 的定义如下：

1. 随机选择一个特征维度，概率正比于$\frac{\ell_{i}}{\sum_{j} \ell_{j}}$，其中$\ell_{i}=max⁡_{x∈S}x_i−min_{x∈S}x_i$.
2. 选择样本点 $X_i∼Uniform[min⁡_{x∈S}x_i,max⁡_{x∈S}x_i]$符合均匀分布。
3. 划分集合 $S_1=\{x∣x∈S,x_i≤X_i\}，S_2=S−S_1$。
4. 在 S1 和 S2 中重复以上步骤。

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/construce_rrcf.png)



- 从一个Tree中删除某个样本

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/delete_point.png" style="zoom:50%;" />

- 插入一个新的样本到树结构中

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/insert_point.png" style="zoom:50%;" />

- 计算异常得分

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/compute_score.png" style="zoom:50%;" />



### 如何衡量样本的异常值

- 引用论文中的一段话

> Let 0 be a left decision and 1 be a right decision. Thus, for x ∈ S, we can write its path as a binary string in the same manner of the isolation forest, m(x). We can repeat this for all x ∈ S and arrive at a complete model description: M(S) = ⊕x∈Sm(x), the appending of all our m(x) together. We will consider the anomaly score of a point x to be the degree to which including it causes our model description to change if we include it or not,

- 形象化的描述如下所示：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rrcf_fig2.png" style="zoom:50%;" />



上图左侧表示构造出来的树的结构，其中x是我们待处理的样本点，有图表示将该样本点删除后，动态调整树结构的形态。其中 $q_0,...,q_r$ 表示从树的根节点编码到  a 节点的描述串。

- 每个样本的异常分数的含义：**将点x的异常得分视为包含或不包含该点，而导致模型的描述发生改变的程度**

$$
E_T[|M(T|]-E_T[|M(Delete(x,T|]
$$

论文中通过对上式的变换，得到对应的公式：

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/rrcf_fig3.png" style="zoom:50%;" />

- 利用上述公式的描述，可以得到具体的衡量分数，但是如果将上述分数直接转换为异常值，还需要算法同学根据自己的场景进行合理的转换





### 流式改造 - 蓄水池采样算法

算法大致描述：给定一个数据流，数据流长度N很大，且N直到处理完所有数据之前都不可知，请问如何在只遍历一遍数据（O(N)）的情况下，能够随机选取出m个不重复的数据。

具体的伪代码描述：

```cpp
int[] reservoir = new int[m];

// init
for (int i = 0; i < reservoir.length; i++)
{
    reservoir[i] = dataStream[i];
}

for (int i = m; i < dataStream.length; i++)
{
    // 随机获得一个[0, i]内的随机整数
    int d = rand.nextInt(i + 1);
    // 如果随机整数落在[0, m-1]范围内，则替换蓄水池中的元素
    if (d < m)
    {
        reservoir[d] = dataStream[i];
    }
}
```

通过对数据流进行采样，可以较好的从数据流中等概率的进行采样，通过RRCF中提供的DELETE方法，可以将置换出模型的数据动态的删除掉，将新选择的样本数据动态的加入到已经有的树中去，进而得到对应的CODISP值。



### 并行调用的改造

该算法同Isolation Forest算法一样，非常适合并行构建，在此不做太多的赘述，推荐读者使用Python一个并行的软件包Joblib，能非常方便的帮助用户开发。

传送门：[Joblib: running Python functions as pipeline jobs](https://link.zhihu.com/?target=https%3A//joblib.readthedocs.io/en/latest/index.html)






















# disentangling-jacobian 论文阅读

论文地址: [https://openreview.net/pdf?id=Hkg4W2AcFm](https://openreview.net/pdf?id=Hkg4W2AcFm)

代码：[https://github.com/jlezama/disentangling-jacobian](https://github.com/jlezama/disentangling-jacobian)

![20190527204542.png](https://cdn.nlark.com/yuque/0/2019/png/308996/1558961183315-9082bacf-45a7-4c99-9160-9863c8dfd531.png#align=left&display=inline&height=302&name=20190527204542.png&originHeight=302&originWidth=1140&size=54970&status=done&width=1140)

## 概述

这篇论文主要是解决属性分解和重构的平衡问题，应用于人脸属性的修改问题上，一方面能实现修改任意一个人脸的属性，比如头发颜色、年龄、性别、表情、是否戴眼镜等，另一方面保证图片其余内容没有失真，即除了对应修改属性的区域外，保证其他区域不变。

论文是用到一个雅可比监督(jacobian supervision)方法，并且网络结构是基于 autoencoder，也就是 encoder-decoder 的结构，然后在训练策略上是先训练一个 Teacher 模型，然后采用雅可比监督训练 Student 模型，这两个模型还各自分两个阶段训练。



## 背景


## 方法

整体网络结构如下图所示：

![20190527211358.png](https://cdn.nlark.com/yuque/0/2019/png/308996/1558962850889-894dbac3-c812-472e-990b-2dbdfc4d4219.png#align=left&display=inline&height=312&name=20190527211358.png&originHeight=312&originWidth=742&size=117817&status=done&width=742)

其中图中所示的 z 和 y 都属于 latent code，即隐变量，也是非常关键的内容，它们各自负责两方面内容，其中 y 负责修改属性，而 z 负责重构图像，也可以说就是图像内容，之所以要训练 Teacher-Student 两个模型，目的也是要做到修改属性和保证图片其他内容不变的平衡。


## 实验结果

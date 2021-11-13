# CPN--Cascaded Pyramid Network论文笔记

**关键词**：关键点检测，多人姿势估计

论文地址：[https://arxiv.org/abs/1711.07319](https://arxiv.org/abs/1711.07319)

Tensorflow 代码：[https://github.com/chenyilun95/tf-cpn](https://github.com/chenyilun95/tf-cpn)

![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559094995131-857ccaa4-d4db-499b-b7b3-94a86a0f8c84.png#align=left&display=inline&height=160&originHeight=160&originWidth=722&size=0&status=done&width=722)
# 概述
随着 CNN 的发展，多人姿态估计问题得到了极大的提升。然而，依然存在很多挑战，比如被遮挡的关键点、看不见的关键点以及复杂的背景，这些目前都不能很好得到处理。本文提出了一个新的网络结构--级联金字塔网络(Cascaded Pyramid Network, CPN)，目标是解决这些难以解决的关键点问题。它分为两个阶段，GlobalNet 和 RefineNet。GlobalNet 是一个特征金字塔网络，它可以很好定位到比如眼睛和手这种比较简单的关键点，但很难精确识别到遮挡或者看不见的关键点。而 RefineNet 可以在整合来自 GlobalNet 的不同层级特征表征和采用一个在线困难点挖掘损失(online hard keypoint mining loss)来显示处理这些困难的关键点。一般来讲，处理多人姿态估计都是采用一个自顶向下的方法，我们的 CPN 也是采用这样的策略，先采用人体检测器生成一系列的人体框。我们的算法在 COCO 关键点 benchmark 上取得了目前最好的结果，在 COCO test-dev 上取得 73.0 的平均精度，在 COCO test-challenge 数据集取得 72.1 的平均精度，对比 COCO 2016的关键点比赛，提高了 19%。
# 主要贡献

1. 提出了一个新的网络--CPN，结合了一个全局金字塔网络（GlobalNet）和基于在线难点挖掘的金字塔改进网络（RefineNet），这个模型能够同时兼顾人体关节点的局部信息以及全局信息，结果取得了不错的效果;
1. 探索了基于自顶向下的方法中，多种不同因素对于多人姿态估计的影响；
1. 在线困难点挖掘(online hard keypoint mining)方法有助于定位预测困难的关键点，比如遮挡或者看不见的；
1. 测试阶段考量了soft-NMS和传统的hard-NMS（非极大值抑制）在human detection阶段产生的影响，结论是soft-NMS对于最后的结果是有所帮助的。 
1. 论文的算法在COCO2017多人姿态估计中取得state-of-art水平。
# 背景
深度学习在多人姿态估计还是具有很大挑战性，比如遮挡的关键点、复杂的背景信息等。深度学习之难以解决姿态估计，主要原因有两点：

1. 这些难的关键点不能简单地根据其外貌特征来识别。
1. 这些难的关键点在训练过程中没有被明确地解决。
# 
# 方法
## pipeline
通常有两种方法处理多人姿态估计：

- 自底向上：先预测所有的关键点，然后再组合到所有人的所有姿势中；
- 自顶向下：两个步骤，第一步先定位并裁剪出图片中的所有人，第二步就是对裁剪出来的图片当作单人姿态估计问题处理。

本文就是采用第二种，自顶向下，先检测图片的人，并裁剪出来，然后对每个人进行姿态估计，具体如下图所示：
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559094995872-91788a21-1999-41a4-b2c9-f197f0c2a473.png#align=left&display=inline&height=299&originHeight=299&originWidth=985&size=0&status=done&width=985)
## 
## motivation
下面这张图表面本文的网络设计思路。即一些比较容易识别出来的人体关键点，直接利用一个 CNN 模型就可以回归得到；而对于一些遮挡比较严重的关节点，则需要增大局部区域感受野以及结合上下文信息才能够进一步 refine 得到。
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559094995156-db1e38af-443a-49fc-b49d-31cae88c72fe.png#align=left&display=inline&height=489&originHeight=489&originWidth=1066&size=0&status=done&width=1066)
## 
## 网络结构
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559094995105-35ea2fd4-12c8-4031-afb5-d9343658d07b.png#align=left&display=inline&height=244&originHeight=244&originWidth=595&size=0&status=done&width=595)
多人姿态估计包括两部分，人体检测和姿态估计。上图只是 CPN 网络模型结构。

### **Human Detector **
人体检测算法主要是是基于 FPN[1] ，然后采用 Mask RCNN[2] 的 ROIAlign 代替 FPN 的 ROIPooling。COCO 的 80 个类别都会用来训练检测器，但是只有人体检测框用于多人骨架提取。

### Cascaded Pyramid Network(CPN)

GlobalNet 的结构如下所示：

        ![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559094995089-0f0c9ae2-68d8-4fcc-927c-0898a17daf32.png#align=left&display=inline&height=212&originHeight=212&originWidth=622&size=0&status=done&width=622)
# 
# 实验

# 参考
主要的参考论文：

1. Feature pyramid networks for object detection-- FPN，采用该模型检测人体；
1. Mask RCNN
1. Stacked hourglass networks for human pose estimation
1. Towards Accurate Multiperson Pose Estimation in the Wild

参考笔记：

- [《Cascaded Pyramid Network for Multi-Person Pose Estimation》--旷世2017COCO keypoints冠军论文解读](https://blog.csdn.net/zhangboshen/article/details/78836704)
- [CVPR 2018 | 旷视科技人体姿态估计冠军论文——级联金字塔网络CPN](https://zhuanlan.zhihu.com/p/37582402)
- [Cascaded Pyramid Network for Multi-Person Pose Estimation阅读笔记](https://www.paperweekly.site/papers/notes/225)

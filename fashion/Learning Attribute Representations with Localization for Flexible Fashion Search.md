# Learning Attribute Representations with Localization for Flexible Fashion Search

论文地址：[http://openaccess.thecvf.com/content_cvpr_2018/html/Ak_Learning_Attribute_Representations_CVPR_2018_paper.html](http://openaccess.thecvf.com/content_cvpr_2018/html/Ak_Learning_Attribute_Representations_CVPR_2018_paper.html)
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559098346999-f4675e02-7a4e-499a-bcf3-42d68e742e3a.png#align=left&display=inline&height=433&originHeight=433&originWidth=1443&size=0&status=done&width=1443)
# 摘要
本文主要是研究通过给定图片和属性信息实现一个详细的时尚搜索方法。一个可靠的时尚搜索平台应该能够实现：（1） 搜索到和查询的图片相似的图片，即包含相同的属性；（2）允许用户修改具体的属性，比如将衣领从圆领变为 v 领；（3）可以处理特定区域的属性，比如仅仅修改衣袖部分的颜色，其他区域颜色保持不变。要实现上述操作，一个关键挑战就是时尚产品都包含多个属性，然后让每种属性都有各自的表示特征非常重要。本文通过提出一个采用弱监督定位方法来提取区域特征的 **FashionSearchNet** 来解决这个关键问题。本文的方法，一是可以忽视不相关的特征从而提高相似性学习的效果，二是可以加入一个新的利用区域意识的方法来处理特定区域的请求。最后，**FashionSearchNet **的性能优于近期大部分的时尚搜索方法，并且证明了可以适用于不同动态请求下的不同应用场景。

具体应用例子如下所示：
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559098346725-44c6007d-2bea-44fb-a681-db6be7925d22.png#align=left&display=inline&height=418&originHeight=418&originWidth=496&size=0&status=done&width=496)
# 主要贡献

- 介绍了一个新的称作 FashionSearchNet 的网络结构，它可以通过对属性定位来实现属性表示学习，并得到更好的属性表示特征来进行属性操作；
- 属性激活图(attribute activation maps) 可以实现对特定区域的属性操作，并且也可以学习和发现区域和特点区域的属性表示；
- 实验结果表明本文方法要比基准的方法提高了 16% 的性能。
# 方法概览
本文方法的整体概览：
![](https://cdn.nlark.com/yuque/0/2019/png/308996/1559098346993-ee29ada6-5c99-4110-9da9-d15851546fd2.png#align=left&display=inline&height=582&originHeight=582&originWidth=1037&size=0&status=done&width=1037)
整体网络结构是基于 AlexNet，保留卷积层部分，即前面 5 个卷积层，修改的部分有：

- 去掉后面的全连接层，添加两个卷积层，conv6 和 conv7，消除去掉全连接层的影响；
- conv7 后加入 GAP（global average pooling)，得到每种属性的 AAMs（attribute activation maps)，并提取 ROIs(Regions of Interest)，表示每个属性特定的区域，并且在 AAMs 后加入一个分类损失函数；
- conv5 的特征图通过上一步得到的 ROIs 来进行 pooling 操作后连接几个全连接层(每个属性都是 3 个全连接层)，其输出就是属性表示(attribute representations)，然后采用 classification loss 和 ranking loss 来训练；
- 最后将上一步的 attribute representations 都连接为一个全局表示(F)，然后采用一个全局的 Ranking loss
- 此外，还采用了 triplet 的设置，体现在计算 ranking loss 

其中 GAP 可以采用 tf.reduce_mean() 实现，而对于利用 ROIs 进行 pooling 的操作，可以采用 tf.image.crop_and_resize() 实现。

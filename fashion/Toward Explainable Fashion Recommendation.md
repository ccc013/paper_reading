# Toward Explainable Fashion Recommendation

论文地址：[https://www.researchgate.net/publication/330409966_Toward_Explainable_Fashion_Recommendation](https://www.researchgate.net/publication/330409966_Toward_Explainable_Fashion_Recommendation)

![image.png](https://cdn.nlark.com/yuque/0/2019/png/308996/1561343762937-8ff3dac9-6035-4ab4-b6e5-393ca9d7c865.png#align=left&display=inline&height=217&name=image.png&originHeight=217&originWidth=669&size=36402&status=done&width=669)
# 概述

论文基于可解释的角度，提出一个搭配评分系统，它可以实现：

1. 对一套搭配给出评分，判断是好，还是不好；
1. 给出评分的原因，即判断哪套衣服或者是哪个衣服的哪种特征，导致搭配比较差；
1. 采用的特征是 **human-interpretable**，即人类可解释角度的特征集合；
1. 设计的评价实验是从搭配中替换一件会降低评分的衣服，然后看系统是否可以检测到被替换进来的衣服。

检测例子如下图所示，第一行是检测糟糕的衣服，第二、三行是特征角度上检测，分别对形状-纹理、颜色两种特征。

检测的原理主要是提出一个**Item-Feature Influence Value(IFIV)，衣服-特征影响数值，**其计算方法是用**特征和其对应输出分数的梯度进行相乘**得到，这和实现 CNNs 的可视化推断类似（输入图片和sensitivity map 相乘得到），以及 Grad-CAM 也相似。




![image.png](https://cdn.nlark.com/yuque/0/2019/png/308996/1561343827388-c7f5dc1c-4971-448c-bcf5-a6982303e38f.png#align=left&display=inline&height=504&name=image.png&originHeight=504&originWidth=499&size=162991&status=done&width=499)

# 背景
## 1. Measuring Goodness of Outfits
目前有不少方法都可以很好评估一套搭配的好坏，即给出分数，但问题是对预测的分数缺乏解释，给不出为什么可以得到这个分数的让人信服的理由。
## 2. Explaining Inference of Models
随着深度学习模型效果越来越好，也开始有更多人开始研究深度学习的可解释问题，即模型给出预测或者判断的原因和理由。

这方面的研究成果有：

1. **Class Activation Map(CAM)**[1]：每个类别训练一个模型，其输出一个激活图显示了输入图片哪个区域对模型预测有重大影响；
1. **Grad-CAM**[2]：对 CAM 的改进，可以应用到更多的 CNN 模型，包括 image captioning，Visual Question Answering(VQA）
## 3. Explaining Models for Fashion
同样也开始有研究时尚方面的可解释模型。

# 方法


# 实验


# 引用论文

1. B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2921–2929, 2016
1. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam,  D. Parikh, D. Batra, et al., “Grad-cam: Visual explanations from deep networks via gradient-based localization.,” in Proceedings of the IEEE International  Conference on Computer Vision (ICCV), pp. 618–626,
2017. 2




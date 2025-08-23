# 开发一个系统（一般的图像的系统或者遥感图像的系统）
主要用书：《Geoprocessing with Python》  
数字图像处理教程：  
https://www.bilibili.com/video/BV1j7411i78H?t=19.3&p=4  
https://www.bilibili.com/video/BV1YA411K7pp?t=49.5&p=8  
https://www.bilibili.com/video/BV1Kh411X7Qv?t=25.9&p=2  
https://www.bilibili.com/video/av1306278684?t=1877.8  

遥感数字图像处理教程：https://www.bilibili.com/video/BV1WJ411D7AG?t=195.6

对应教材：
https://etcnew.sdut.edu.cn/meol/common/script/preview/download_preview.jsp?fileid=3654454&resid=787291&lid=48026&preview=preview


遥感图像处理的基本操作包括一系列预处理、增强、分析和分类等步骤，具体如下：

## 1. 数据预处理 √
这是遥感图像处理的第一步，确保数据质量和一致性。包括：  

### 影像拼接（镶嵌）：将多幅图像拼接成完整的覆盖区域。  
教程：
《Geoprocessing with Python》 第213页  
https://pygis.io/docs/f_rs_mosaic.html  问题：用到的geowombat比较小众，github上只有188个star。  
https://blog.csdn.net/m0_51301348/article/details/124460560（GDAL,raster）  
https://www.cnblogs.com/RSran/p/17776513.html(GDAL)  
https://juejin.cn/post/6892002048214138894（rasterio和GDAL）  
https://zhuanlan.zhihu.com/p/139383690（GDAL）

### 裁剪与重采样：根据研究区域裁剪图像或对分辨率进行调整。
问题：如何用代码实现不规则裁剪（任意多边形）？

裁剪教程（用的技术：GDAL）：
    https://zhuanlan.zhihu.com/p/397149374  
    https://www.cnblogs.com/RSran/p/17631881.html  

重采样教程（用的技术：GDAL，Pyresample【Pyresample使用了Xarray和Dask】）：  
    《Geoprocessing with Python》 196页  
    https://blog.csdn.net/zhanglingfeng1/article/details/129227390  
    https://www.cnblogs.com/tangjielin/p/16599414.html  
    https://zhuanlan.zhihu.com/p/587583231

### 技术选择：GDAL。  
原因：有基金会和社区支持，开发了25年的时间。github上已有5.4k star。

## 2. 图像增强（可不做）
为了突出图像的某些特征，常用以下方法：  
直方图拉伸：调整亮度值范围以增强对比度。  
对比度增强：通过线性或非线性方法提升图像对比度。  
滤波处理：  
低通滤波：用于去噪。  
高通滤波：用于边缘增强。  
伪彩色合成：通过为不同波段分配颜色，增强地物区分。  

## 3. 特征提取（可不做）
提取图像中具有研究价值的信息，如：  
纹理分析：识别地物的粗糙度、平滑度等。  
边缘检测：提取地物边界（Sobel、Canny 算法等）。  
形状分析：分析地物的几何特征。  

## 4. 波段运算（可不做）
遥感影像通常包含多个光谱波段，波段运算是遥感数据处理的核心，包括：  
波段比值：如归一化植被指数（NDVI）。  
主成分分析（PCA）：压缩数据并突出主要信息。  
光谱混合分析：分解像元的混合光谱成纯像元分量。

## 5. 分类和地物识别

根据图像的光谱特征和其他属性，将像元归类：  

监督分类：基于训练样本的分类方法（如最大似然法、支持向量机）。  

非监督分类：无需训练样本的聚类方法（如K均值）。  

对象导向分类：结合光谱信息和空间特征（形状、纹理）进行分类。  

深度学习：利用卷积神经网络（CNN）等模型进行自动分类。（后面放我们自己的算法）  


### 资料
obia的过去，现在，未来：https://www.mdpi.com/2072-4292/12/12/2012  
An Introduction to Convolutional Neural Networks: https://arxiv.org/pdf/1511.08458  
卷积神经网络是如何工作的？: https://e2eml.school/how_convolutional_neural_networks_work.html  
斯坦福CNN：https://cs231n.github.io/convolutional-networks/  
构建第一个ANN：https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/  
Building a Multiclass Classification Model in PyTorch：https://machinelearningmastery.com/building-a-multiclass-classification-model-in-pytorch/  
构建一个卷积神经网络：https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/
### 问题
#### 根据图像的颜色不就可以分类吗？有那么多属性需要考虑吗？
答：足球场上的假草皮，和绿化带中的草丛都是绿色，你能根据颜色就把它们分为同一个类吗？

#### “利用卷积神经网络（CNN）等模型进行自动分类”中自动啥意思？
答：即不需要手工选取特征，而是CNN自动提取特征。


## 6. 变化检测
通过比较不同时间的遥感图像，分析地表变化：  

差值法：计算波段差值或光谱指数差值。  

比值法：计算两个时间点的波段比值。  

分类后比较法：对两期影像分别分类后进行对比。  

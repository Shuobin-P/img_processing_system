系统架构：B/S or C/S ?

B/S：
    Browser展示数据，Server: 提供数据
C/S:
    没做过C/S架构的软件。


数据预处理：
    影像拼接：
        上传tif图像 -> 拼接图像 -> 返回拼接后的图像

    裁剪与重采样：
        裁剪：
            矩形裁剪：
                上传想要裁剪的tif图像 -> 输入裁剪开始的起点，裁剪的宽度和高度 -> 返回裁剪结果
        重采样：
            选择RESAMPLING TO SMALLER PIXELS还是RESAMPLING TO LARGER PIXELS -> 上传tif图像 -> 返回结果

分类和地物识别：
    
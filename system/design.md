# 架构选择
系统架构：B/S or C/S ?

B/S：
    Browser展示数据，Server: 处理与提供数据

C/S:
    没做过C/S架构的软件。

注：现在只是要你把系统做出来，不要求这个系统做得多好，B/S架构能做，就用B/S架构。

技术栈选择：
    前端：
        Streamlit: A faster way to build and share data apps. 
                    Turn your data scripts into shareable web apps in minutes. All in pure Python. No front‑end experience required.
                    案例：https://streamlit.io/playground?example=computer_vision
                Github star: 41.2k
                特点：
                    开发快
                    纯python，不需要前端经验

        Gradio:
                Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!

                Github star: 39.7k
                特点：
                    快速实现通过web界面展示你的机器学习模型

        Dash: 
                Dash is the most downloaded, trusted Python framework for building ML & data science web apps.
                
                Github star: 24k

# 操作逻辑
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

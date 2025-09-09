import streamlit as st

st.title("分类与地物识别") 

# 思路：直接调用你训练好的模型即可，无需重复训练。

st.markdown("# 监督分类")

uploaded_img = st.file_uploader(
    "请上传你要分类的图像", accept_multiple_files=False, type=["tif", "tiff"]
)










st.markdown("# 非监督分类")
st.markdown("# 对象导向分类")
st.markdown("# 深度学习")

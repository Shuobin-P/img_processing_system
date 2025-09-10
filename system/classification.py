import streamlit as st
import tempfile
import numpy as np
from osgeo import gdal
import joblib

custom_temp_dir = r"D:\Projects\VsCode\Python\img_processing_system\tmp"

st.title("分类与地物识别") 

# 思路：直接调用你训练好的模型即可，无需重复训练。

st.markdown("# 监督分类")
st.markdown("## 随机森林模型")

uploaded_img = st.file_uploader(
    "请上传你要分类的图像", accept_multiple_files=False, type=["tif", "tiff"]
)

source_file_path = None
if uploaded_img is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
        tmp1.write(uploaded_img.getbuffer())
        source_file_path = tmp1.name

    img_ds = gdal.Open(source_file_path)
    rows = img_ds.RasterYSize
    cols = img_ds.RasterXSize
    if img_ds.RasterCount < 3:
        raise ValueError("训练影像波段数少于3，请检查 img_ds")

    # 读取前三个波段并堆叠为 (rows, cols, 3)
    b1 = img_ds.GetRasterBand(1).ReadAsArray()
    b2 = img_ds.GetRasterBand(2).ReadAsArray()
    b3 = img_ds.GetRasterBand(3).ReadAsArray()
    stack = np.dstack((b1, b2, b3))  # shape = (rows, cols, 3)

    # 重塑为 (n, 3)
    new_img_X = stack.reshape((rows * cols, 3))
    loaded_model = joblib.load(r"classification\supervised_classification\pkl\random_forest_model.pkl")
    pred_result = loaded_model.predict(new_img_X)
    prediction_matrix = np.reshape(pred_result, (rows, cols))
    driver = gdal.GetDriverByName('gtiff')

    result_path = r'tmp\supervised_classification_prediction_result.tif'
    predict_ds = driver.Create(result_path, cols, rows)
    predict_ds.SetProjection(img_ds.GetProjection()) 
    predict_ds.SetGeoTransform(img_ds.GetGeoTransform())
    band = predict_ds.GetRasterBand(1)
    band.WriteArray(prediction_matrix)
    band.FlushCache() 
    # 构建 colortable，索引对应标签 0-5
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0, 0, 0, 0))         # 0: No information -> 透明
    ct.SetColorEntry(1, (255, 0, 0, 255))     # 1: Artificial surfaces -> 红
    ct.SetColorEntry(2, (255, 255, 0, 255))   # 2: Agricultural areas -> 黄
    ct.SetColorEntry(3, (0, 128, 0, 255))     # 3: Forests -> 绿
    ct.SetColorEntry(4, (0, 160, 160, 255))   # 4: Wetlands -> 青/绿蓝
    ct.SetColorEntry(5, (0, 0, 255, 255))     # 5: Water -> 蓝

    predict_ds.GetRasterBand(1).SetRasterColorTable(ct) 
    del img_ds,predict_ds

    # 提供用户下载按钮
    with open(result_path, "rb") as f:
        st.download_button(
            label="下载分类结果",
            data=f,
            file_name="class_" + uploaded_img.name,
            mime="image/tiff"
        )


st.markdown("# 非监督分类")
st.markdown("## K均值模型")








st.markdown("# 对象导向分类")
st.markdown("# 深度学习")

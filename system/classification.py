import streamlit as st
import tempfile
import numpy as np
from osgeo import gdal
import joblib
import concurrent.futures
import pickle
from utils import process_segment
from skimage.segmentation import slic
from skimage import exposure
from multiprocessing import shared_memory


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
        raise ValueError("影像波段数少于3，请检查 img_ds")

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
    # ct.SetColorEntry(0, (0, 0, 0, 0))         # 0: No information -> 透明
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
uploaded_k_means_img = st.file_uploader(
    "请上传图像", accept_multiple_files=False, type=["tif", "tiff"]
)

source_file_path = None
if uploaded_k_means_img is not None:
    # 保存用户上传的图像
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
        tmp1.write(uploaded_k_means_img.getbuffer())
        source_file_path = tmp1.name

    img_ds = gdal.Open(source_file_path)
    rows = img_ds.RasterYSize
    cols = img_ds.RasterXSize
    if img_ds.RasterCount < 3:
        raise ValueError("影像波段数少于3，请检查 img_ds")

    # 读取前三个波段并堆叠为 (rows, cols, 3)
    b1 = img_ds.GetRasterBand(1).ReadAsArray()
    b2 = img_ds.GetRasterBand(2).ReadAsArray()
    b3 = img_ds.GetRasterBand(3).ReadAsArray()
    stack = np.dstack((b1, b2, b3))  # shape = (rows, cols, 3)

    # 重塑为 (n, 3)
    new_img_X = stack.reshape((rows * cols, 3))
    kmeans_clf = joblib.load(r"classification\unsupervised_classification\pkl\k_means_model.pkl")
    
    # 分类
    pred_result = kmeans_clf.predict(new_img_X)

    prediction_matrix = np.reshape(pred_result, (rows, cols)) + 1 # +1 为了与原分类保持一致，即1: Artificial surfaces -> 红

    driver = gdal.GetDriverByName('gtiff') 
    result_path = r"D:\Projects\VsCode\Python\img_processing_system\tmp\system_k_means_prediction_result.tif"
    predict_ds = driver.Create(result_path, cols, rows) 
    predict_ds.SetProjection(img_ds.GetProjection()) 
    predict_ds.SetGeoTransform(img_ds.GetGeoTransform())
    band = predict_ds.GetRasterBand(1)
    band.WriteArray(prediction_matrix)
    band.FlushCache() 

    # 添加colortable
    ct = gdal.ColorTable()

    ct.SetColorEntry(1, (255, 0, 0, 255))     # 1: Artificial surfaces -> 红
    ct.SetColorEntry(2, (255, 255, 0, 255))   # 2: Agricultural areas -> 黄
    ct.SetColorEntry(3, (0, 128, 0, 255))     # 3: Forests -> 绿
    ct.SetColorEntry(4, (0, 160, 160, 255))   # 4: Wetlands -> 青/绿蓝
    ct.SetColorEntry(5, (0, 0, 255, 255))     # 5: Water -> 蓝

    predict_ds.GetRasterBand(1).SetRasterColorTable(ct) 
    del img_ds, predict_ds

    # 提供用户下载按钮
    with open(result_path, "rb") as f:
        st.download_button(
            label="下载分类结果",
            data=f,
            file_name="class_" + uploaded_k_means_img.name,
            mime="image/tiff"
        )
    st.write("注：该k-means模型表现很差")

st.markdown("# 对象导向分类")
# 思路：将图片分成很多小块，然后根据每个小块的特征进行分类，
# 新图片可以从qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624.tif取某个块，模型训练和新图来自同一张图
# 模型从而能够对该新图做分类。
st.text("提示：建议用500x256分辨率的图片，大概1分钟能得到结果。5834x2000分辨率的图像跑了18min都没得到结果。")
uploaded_obia_img = st.file_uploader(
    "请上传要进行obia的图像", accept_multiple_files=False, type=["tif", "tiff"]
)
source_file_path = None
if uploaded_obia_img is not None:
    # 保存用户上传的图像
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
        tmp1.write(uploaded_obia_img.getbuffer())
        source_file_path = tmp1.name
    naip_fn = source_file_path
    naip_ds = gdal.Open(naip_fn)
    nbands = naip_ds.RasterCount
    band_data = []
    for i in range(1, nbands+1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    band_data = np.dstack(band_data)
    img = exposure.rescale_intensity(band_data)
    segments = slic(img, n_segments=50000, compactness=0.1) # segments是一个和img相同shape的矩阵，每个值表示img中每个像素点的segment ID
    
    shm_img = shared_memory.SharedMemory(create=True, size=img.nbytes)
    shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_img.buf) 
    shared_img[:] = img[:]

    shm_segments = shared_memory.SharedMemory(create=True, size=segments.nbytes)
    shared_segments = np.ndarray(segments.shape, dtype=segments.dtype, buffer=shm_segments.buf)
    shared_segments[:] = segments[:]

    # Pass metadata so workers can access shared memory
    shared_names = {
        'img': shm_img.name,
        'segments': shm_segments.name
    }
    shared_shapes = {
        'img': img.shape,
        'segments': segments.shape
    }
    shared_dtypes = {
        'img': img.dtype,
        'segments': segments.dtype
    }

    segment_ids = np.unique(segments)
    args_list = [(segment_id, shared_names, shared_shapes, shared_dtypes) for segment_id in segment_ids]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_segment, args_list, timeout=3600))
    print("完成了segment特征计算")
    object_ids, objects = zip(*results) # object_ids的元素是segment_id，objects的元素是一个segment的光谱特征
    shm_img.close()
    shm_img.unlink()
    shm_segments.close()
    shm_segments.unlink()

    # 加载模型
    with open(r"classification\supervised_classification\pkl\obia\random_forest_model_v2.pkl", "rb") as f:
        classifier = pickle.load(f)
    print("模型分类中")
    # 预测分类
    predicted = classifier.predict(objects)
    print("分类完成")
    clf = np.copy(segments)
    for segment_id, klass in zip(segment_ids, predicted): # segment_ids=[1, 2, 3,..., 40150]，
        # predicted与segment_ids一一对应
        clf[clf == segment_id] = klass # clf：即每个像素的分类（land cover）结果
        mask = np.sum(img, axis=2) # 对每个像素的所有波段求和，结果是一个二维矩阵。mask.shape = (2000, 5834) 2000代表y轴方向，5834代表x轴方向
    mask[mask > 0.0] = 1.0 # mask > 0.0表示该像素点有数据
    mask[mask == 0.0] = -1.0 # mask == 0，表示该像素点没有数据
    clf = np.multiply(clf, mask) # 不是线代中的矩阵乘法，而是两个矩阵相同位置的两个元素相乘
    clf[clf < 0] = -9999.0

    # Saving classificaiton to raster with gdal
    driverTiff = gdal.GetDriverByName('GTiff')

    result_path = r'tmp\obia_classified_result.tif'
    clfds = driverTiff.Create(result_path, naip_ds.RasterXSize, naip_ds.RasterYSize,
                            1, gdal.GDT_Float32)
    
    clfds.SetGeoTransform(naip_ds.GetGeoTransform())
    clfds.SetProjection(naip_ds.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clfds.GetRasterBand(1).WriteArray(clf)
    del naip_ds, clfds

    # 提供用户下载按钮
    with open(result_path, "rb") as f:
        st.download_button(
            label="下载obia分类结果",
            data=f,
            file_name="obia_classification" + uploaded_obia_img.name,
            mime="image/tiff"
        )









st.markdown("# 深度学习")

"""
测试数据：
    D:\Projects\VsCode\Python\img_processing_system\osgeopy-data\Massachusetts 中以O开头的所有tif文件
"""
import streamlit as st
from osgeo import gdal
import tempfile
import math
custom_temp_dir = r"D:\Projects\VsCode\Python\img_processing_system\tmp"

def get_extent(fn): 
    '''Returns min_x, max_y, max_x, min_y''' 
    ds = gdal.Open(fn)
    gt = ds.GetGeoTransform()
    # ds.RasterXSize 指照片的长有多少像素点，ds.RasterYSize是照片的宽有多少像素点
    ds_raster_x_size = ds.RasterXSize
    ds_raster_y_size = ds.RasterYSize
    del ds
    return (gt[0], gt[3], gt[0] + gt[1] * ds_raster_x_size, gt[3] + gt[5] * ds_raster_y_size)

st.title("遥感影像拼接") 

# 1. 上传文件
uploaded_files = st.file_uploader(
    "请上传遥感影像 (.tif)", accept_multiple_files=True, type=["tif", "tiff"]
)

def mosaic_tifs(uploaded_files):
    # 将上传的文件保存
    file_names = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
                tmp1.write(file.getbuffer())
                file_names.append(tmp1.name) 
    print("All files :", file_names) 
    min_x, max_y, max_x, min_y = get_extent(file_names[0]) 
    for fn in file_names[1:]: 
        minx, maxy, maxx, miny = get_extent(fn) 
        min_x = min(min_x, minx) 
        max_y = max(max_y, maxy) 
        max_x = max(max_x, maxx) 
        min_y = min(min_y, miny)

    # Calculate dimensions
    in_ds = gdal.Open(file_names[0]) 
    gt = in_ds.GetGeoTransform() 
    rows = math.ceil((max_y - min_y) / -gt[5]) 
    columns = math.ceil((max_x - min_x) / gt[1])

    # Create output
    driver = gdal.GetDriverByName('gtiff') 
    result_tif_path = custom_temp_dir + '\mosaic_result.tif'
    out_ds = driver.Create(result_tif_path, columns, rows) 
    print("columns = ", columns)
    print("rows = ", rows)
    out_ds.SetProjection(in_ds.GetProjection()) 
    out_band = out_ds.GetRasterBand(1)

    # Calculate new geotransform
    gt = list(in_ds.GetGeoTransform()) 
    gt[0], gt[3] = min_x, max_y 
    out_ds.SetGeoTransform(gt)

    for fn in file_names: 
        in_ds = gdal.Open(fn) 
        # Get output offsets
        trans = gdal.Transformer(in_ds, out_ds, []) 
        success, xyz = trans.TransformPoint(False, 0, 0) 
        x, y, z = map(int, xyz) 
        # Copy data
        data = in_ds.GetRasterBand(1).ReadAsArray() 
        out_band.WriteArray(data, x, y)

    del in_ds, out_ds

    return result_tif_path

# 2. 点击按钮触发拼接
if uploaded_files:
    if st.button("开始拼接"):
        st.write("正在拼接，请稍候...")
        result_path = mosaic_tifs(uploaded_files)
        st.success("拼接完成！")
        # 3. 提供下载
        with open(result_path, "rb") as f:
            st.download_button(
                label="下载拼接结果",
                data=f,
                file_name="mosaic_result.tif",
                mime="image/tiff"
            )

st.markdown("## 裁剪")
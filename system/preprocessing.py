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

uploaded_to_subset_file = st.file_uploader("请上传你要裁剪的图像",  type=["tif", "tiff"], accept_multiple_files=False)
# 让用户输入裁剪的起始位置，以及窗口大小，即x_offset, y_offset, window_width, window_height
if uploaded_to_subset_file is not None:
    # 用户输入裁剪参数
    x_offset = st.number_input("X 偏移量", min_value=0, value=0)
    y_offset = st.number_input("Y 偏移量", min_value=0, value=0)
    window_width = st.number_input("裁剪宽度", min_value=1, value=256)
    window_height = st.number_input("裁剪高度", min_value=1, value=256)

    source_file_path = None
    dst_file_path = custom_temp_dir + "\subset_" + uploaded_to_subset_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
                tmp1.write(uploaded_to_subset_file.getbuffer())
                source_file_path = tmp1.name
    print("source_file_path= ", source_file_path)
    print("dst_file_path= ", dst_file_path)
    ds = gdal.Translate(
        dst_file_path,
        source_file_path,
        srcWin=[x_offset, y_offset, window_width, window_height]  # x_offset, y_offset, window_width, window_height
    )
    del ds
    # 3. 提供下载
    with open(dst_file_path, "rb") as f:
        st.download_button(
            label="下载裁剪结果",
            data=f,
            file_name="subset_" + uploaded_to_subset_file.name,
            mime="image/tiff"
        )


st.markdown("## 重采样")
st.markdown("### 重采样到更高分辨率")

resampling_to_smaller_pixels_source_img= st.file_uploader("请上传你要重采样的图像",  type=["tif", "tiff"], accept_multiple_files=False)

if resampling_to_smaller_pixels_source_img is not None:
    source_img_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
            tmp1.write(resampling_to_smaller_pixels_source_img.getbuffer())
            source_img_path = tmp1.name
    in_ds = gdal.Open(source_img_path) 
    in_band = in_ds.GetRasterBand(1) # TODO 只处理了一个波段

    # Get number of output rows and columns
    out_rows = in_band.YSize * 2 
    out_columns = in_band.XSize * 2
    gtiff_driver = gdal.GetDriverByName('GTiff') 
    res_img_path = custom_temp_dir + "\\band1_resampled_smaller_pixels.tif"
    print("res_img_path= ", res_img_path)
    # Create output dataset
    out_ds = gtiff_driver.Create(res_img_path, out_columns, out_rows)
    out_ds.SetProjection(in_ds.GetProjection()) 

    # Edit the geotransform so pixels are one-quarter previous size
    geotransform = list(in_ds.GetGeoTransform()) 
    geotransform [1] /= 2 
    geotransform [5] /= 2 
    out_ds.SetGeoTransform(geotransform)

    # Specify a larger buffer size when reading data
    data = in_band.ReadAsArray(buf_xsize=out_columns, buf_ysize=out_rows)
    out_band = out_ds.GetRasterBand(1) 
    out_band.WriteArray(data)

    # Build appropriate number of overviews for larger image
    out_band.FlushCache() 
    out_band.ComputeStatistics(False) 
    out_ds.BuildOverviews('average', [2, 4, 8, 16, 32, 64])
    del out_ds

    # 提供下载
    with open(res_img_path, "rb") as f:
        st.download_button(
            label="下载重采样结果",
            data=f,
            file_name="resampling_to_smaller_pixel_" + resampling_to_smaller_pixels_source_img.name,
            mime="image/tiff"
        )

    st.markdown(
        """
        🔎 检查是否重采样成功的办法：查看 *原图* 和 *新图* 的分辨率
        """
    )






st.markdown("### 重采样到更低分辨率")


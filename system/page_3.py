# FIXME 新图原图的颜色不对!!! 甚至同一张图片直接原封不动写过去，颜色都变了。见Page4
import streamlit as st
from osgeo import gdal
import tempfile

custom_temp_dir = r"D:\Projects\VsCode\Python\img_processing_system\tmp"

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")
st.markdown("## 重采样")
st.markdown("### 重采样到更高分辨率")

resampling_to_smaller_pixels_source_img= st.file_uploader("请上传你要重采样的图像",  type=["tif", "tiff"], accept_multiple_files=False)

if resampling_to_smaller_pixels_source_img is not None:
    source_img_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
            tmp1.write(resampling_to_smaller_pixels_source_img.getbuffer())
            source_img_path = tmp1.name
    in_ds = gdal.Open(source_img_path) 
    res_img_path = custom_temp_dir + "\\resampled_smaller_pixels.tif" # 存储在服务端的文件名
    print("res_img_path= ", res_img_path)

    # Get number of output rows and columns
    out_rows = in_ds.GetRasterBand(1).YSize * 2 # 新图的行数是原图的两倍
    out_columns = in_ds.GetRasterBand(1).XSize * 2
    gtiff_driver = gdal.GetDriverByName('GTiff') 
    
    # Create output dataset
    out_ds = gtiff_driver.Create(res_img_path, out_columns, out_rows, bands=in_ds.RasterCount)
    out_ds.SetProjection(in_ds.GetProjection()) 

    # Edit the geotransform so pixels are one-quarter previous size
    geotransform = list(in_ds.GetGeoTransform()) 
    geotransform [1] /= 2 
    geotransform [5] /= 2 
    out_ds.SetGeoTransform(geotransform)

    for i in range(1, in_ds.RasterCount + 1):
        in_band = in_ds.GetRasterBand(i) 

        # Specify a larger buffer size when reading data
        print("Before= \n", in_band.ReadAsArray())
        data = in_band.ReadAsArray(buf_xsize=out_columns, buf_ysize=out_rows)
        print("After= \n", data)
        out_band = out_ds.GetRasterBand(i) 
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

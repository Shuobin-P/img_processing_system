import streamlit as st
from osgeo import gdal
import tempfile


custom_temp_dir = r"D:\Projects\VsCode\Python\img_processing_system\tmp"

st.markdown("# Page 4 🎉")
st.sidebar.markdown("# Page 4 🎉")
st.markdown("## 测试颜色的变化是否正常")
st.markdown("### 将一张图写到另一张图")

resampling_to_smaller_pixels_source_img= st.file_uploader("请上传你要测试的图像",  type=["tif", "tiff"], accept_multiple_files=False)

if resampling_to_smaller_pixels_source_img is not None:
    source_img_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=custom_temp_dir) as tmp1:
            tmp1.write(resampling_to_smaller_pixels_source_img.getbuffer())
            source_img_path = tmp1.name
    
    in_ds = gdal.Open(source_img_path) 
    res_img_path = custom_temp_dir + "\\test_pixels.tif" # 存储在服务端的文件名
    print("res_img_path= ", res_img_path)

    # Get number of output rows and columns
    out_rows = in_ds.GetRasterBand(1).YSize
    out_columns = in_ds.GetRasterBand(1).XSize
    gtiff_driver = gdal.GetDriverByName('GTiff') 
    
    # Create output dataset
    out_ds = gtiff_driver.Create(res_img_path, out_columns, out_rows, bands=in_ds.RasterCount, eType=in_ds.GetRasterBand(1).DataType, options=["ALPHA=NO"])
    out_ds.SetProjection(in_ds.GetProjection()) 

    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    for i in range(1, in_ds.RasterCount + 1):
        in_band = in_ds.GetRasterBand(i) 

        # Specify a larger buffer size when reading data
        print("Before= \n", in_band.ReadAsArray())
        print(in_band.ReadAsArray().dtype)
        data = in_band.ReadAsArray()
        print("After= \n", data)
        print(data.dtype)
        out_band = out_ds.GetRasterBand(i) 
        out_band.WriteArray(data)

        # Build appropriate number of overviews for larger image
        out_band.FlushCache() 
        stats = in_band.GetStatistics(0, 1)  # min, max, mean, std
        out_band.SetStatistics(*stats)

        out_band.ComputeStatistics(False) 

    #out_ds.BuildOverviews('average', [2, 4, 8, 16, 32, 64])
    del in_ds, out_ds

    # 提供下载
    with open(res_img_path, "rb") as f:
        st.download_button(
            label="下载测试结果",
            data=f,
            file_name="test_" + resampling_to_smaller_pixels_source_img.name,
            mime="image/tiff"
        )

    def inspect(path, tag):
        ds = gdal.Open(path)
        print("===", tag, "===")
        print("RasterCount:", ds.RasterCount)
        print("Projection:", ds.GetProjection()[:80])
        print("GeoTransform:", ds.GetGeoTransform())
        for i in range(1, ds.RasterCount+1):
            b = ds.GetRasterBand(i)
            print("Band", i, "dtype:", gdal.GetDataTypeName(b.DataType),
                "NoData:", b.GetNoDataValue(),
                "Scale:", b.GetScale(), "Offset:", b.GetOffset(),
                "HasColorTable:", bool(b.GetColorTable()))
            stats = b.GetStatistics(False, True)  # compute if needed
            print("  stats min,max,mean,std:", stats)
        ds = None

    inspect(source_img_path, "原图")
    inspect(res_img_path, "新图")
    source_ds = gdal.Open(source_img_path)
    target_ds = gdal.Open(res_img_path)
    import numpy as np
    for i in range(1, source_ds.RasterCount+1):
         print("band ", i, np.unique(source_ds.GetRasterBand(i).ReadAsArray() == target_ds.GetRasterBand(i).ReadAsArray()))
    
    
    del source_ds,target_ds
from osgeo import gdal
import os

# 通过指定矩形框来裁剪图像
def get_subset_tif(work_dir, src_tif_fn, offset_x, offset_y, win_xsize, win_ysize, res_tif_fn) -> None:
    """
    work_dir: 工作目录
    src_tif_fn: 待裁剪的图像的文件名
    offset_x: res_tif_fn左上角顶点的x方向偏移量。即左上角顶点的在x轴方向的像素值位置
    offset_y: res_tif_fn左上角顶点的y方向偏移量。即左上角顶点的在y轴方向的像素值位置
    win_xsize: 裁剪区域的x方向长度
    win_ysize: 裁剪区域的y方向长度
    res_tif_fn: 结果图像名
    """
    os.chdir(work_dir)
    src_file = src_tif_fn
    src_file_ds = gdal.Open(src_file)
    gtiff_driver = gdal.GetDriverByName('GTiff') 
    res_ds = gtiff_driver.Create(res_tif_fn, win_xsize, win_ysize, bands = 1, eType=src_file_ds.GetRasterBand(1).DataType)
    res_ds.SetProjection(src_file_ds.GetProjection()) 
    gt = src_file_ds.GetGeoTransform()
    new_gt = ( # 计算得到res_tif_fn左上角的坐标
        gt[0] + offset_x * gt[1],
        gt[1],
        gt[2],
        gt[3] + offset_y * gt[5],
        gt[4],
        gt[5]
    )
    res_ds.SetGeoTransform(new_gt)
    for i in range(1,src_file_ds.RasterCount+1): 
        in_band = src_file_ds.GetRasterBand(i)
        in_band_arr = in_band.ReadAsArray(offset_x, offset_y, win_xsize=win_xsize, win_ysize=win_ysize)
        res_ds.GetRasterBand(i).WriteArray(in_band_arr)
    
    res_ds.FlushCache() 

    for i in range(1, res_ds.RasterCount + 1):
        src_band = src_file_ds.GetRasterBand(i)
        min_val, max_val = src_band.ComputeRasterMinMax(False)
        dst_band = res_ds.GetRasterBand(i)
        dst_band.SetStatistics(min_val, max_val, 0, 0)
    
    del res_ds, src_file_ds
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
    res_ds = gtiff_driver.Create(res_tif_fn, win_xsize, win_ysize, bands = src_file_ds.RasterCount, eType=src_file_ds.GetRasterBand(1).DataType)
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
    """
    for i in range(1, res_ds.RasterCount + 1):
        src_band = src_file_ds.GetRasterBand(i)
        min_val, max_val = src_band.ComputeRasterMinMax(False)
        dst_band = res_ds.GetRasterBand(i)
        dst_band.SetStatistics(min_val, max_val, 0, 0)
    """
    for i in range(1, res_ds.RasterCount + 1):
        dst_band = res_ds.GetRasterBand(i)
        min_val, max_val = dst_band.ComputeRasterMinMax(False)  # 用裁剪后的数据计算
        dst_band.SetStatistics(min_val, max_val, 0, 0)

    del res_ds, src_file_ds



def get_subset_tif(work_dir, src_tif_fn, offset_x, offset_y, win_xsize, win_ysize, res_tif_fn) -> None:
    """
    修复版：裁剪图像并正确保存统计信息，确保显示颜色正常
    """
    os.chdir(work_dir)
    
    # 打开源文件
    src_ds = gdal.Open(src_tif_fn, gdal.GA_ReadOnly)
    if src_ds is None:
        raise Exception(f"无法打开源文件: {src_tif_fn}")
    
    # 获取源数据信息
    band_count = src_ds.RasterCount
    data_type = src_ds.GetRasterBand(1).DataType
    projection = src_ds.GetProjection()
    geotransform = src_ds.GetGeoTransform()
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        res_tif_fn, 
        win_xsize, 
        win_ysize, 
        band_count, 
        data_type,
        options=['COMPRESS=LZW', 'TILED=YES']  # 添加压缩选项以减少文件大小
    )
    
    if dst_ds is None:
        raise Exception(f"无法创建输出文件: {res_tif_fn}")
    
    # 设置地理参考信息
    new_geotransform = (
        geotransform[0] + offset_x * geotransform[1] + offset_y * geotransform[2],
        geotransform[1],
        geotransform[2],
        geotransform[3] + offset_x * geotransform[4] + offset_y * geotransform[5],
        geotransform[4],
        geotransform[5]
    )
    dst_ds.SetGeoTransform(new_geotransform)
    dst_ds.SetProjection(projection)
    
    # 复制每个波段的数据
    for band_idx in range(1, band_count + 1):
        src_band = src_ds.GetRasterBand(band_idx)
        dst_band = dst_ds.GetRasterBand(band_idx)
        
        # 读取裁剪区域的数据
        data = src_band.ReadAsArray(offset_x, offset_y, win_xsize, win_ysize)
        
        # 写入目标波段
        dst_band.WriteArray(data)
        
        # 复制无数据值（如果有）
        nodata = src_band.GetNoDataValue()
        if nodata is not None:
            dst_band.SetNoDataValue(nodata)
        
        # 强制计算统计信息
        dst_band.FlushCache()
        dst_band.ComputeStatistics(False)  # 近似统计即可
    
    # 构建金字塔（非常重要！影响显示效果）
    dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    
    # 确保所有数据写入磁盘
    dst_ds.FlushCache()
    
    # 关闭数据集
    dst_ds = None
    src_ds = None
    
    print(f"裁剪完成: {res_tif_fn}")
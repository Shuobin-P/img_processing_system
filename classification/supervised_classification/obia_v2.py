"""
设计思路：通过考虑形状, 纹理, 从而优化只考虑光谱特征的obia_v1.py
模型运行结果：
    混淆矩阵预测准确率：
        考虑光谱,纹理模型:（contrast, dissimilarity, homogeneity, energy），形状（长宽比）：[1.  0.92857143  0.4   0.5    0.7   0.72727273  0.875 ]
        考虑光谱，形状（长宽比）模型： [1.  0.92857143  0.44444444  0.66666667  0.63636364  0.8  0.85714286]，该v2模型预测的准确率低于v1模型。
    随机森林模型AUC得分:
        考虑光谱,纹理(contrast, dissimilarity, homogeneity, energy)，形状(长宽比)得分: 0.951
        考虑光谱,形状(长宽比)得分: 0.959        
    
特征选取分析：
     形状：
        长宽比: 可以用来区分公路的segment与其他segment

介绍形状，纹理的文章：
    https://blog.csdn.net/qq_31988139/article/details/133910317  
    https://patentimages.storage.googleapis.com/7a/86/7e/32c755861ca900/CN101930547A.pdf  
    http://ch.whu.edu.cn/cn/article/pdf/preview/2432.pdf  

    特征提取：
    https://medium.com/@abishaajith95/image-feature-extraction-using-python-part-i-5b17099ca2f6

"""


import numpy as np
from osgeo import gdal,ogr
from skimage import exposure
from skimage.segmentation import slic
from skimage.feature import graycomatrix, graycoprops
import scipy
import time
from multiprocessing import shared_memory
import concurrent.futures
import pickle
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import cv2


# ------------------ Compute segment features ------------------
def segment_features(segment_pixels): # 计算一个segment的像素的所有band的光谱特征，纹理特征
    features = []
    npixels, nbands = segment_pixels.shape # segment_pixels.shape的值是 (x, 4)，x即像素点的个数
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b]) # segment_pixels[:, b]即取b列的所有行
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        # 计算每个波段的纹理特征
        """
        glcm = graycomatrix([segment_pixels[:,b]], distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, prop='contrast')
        dissimilarity = graycoprops(glcm, prop='dissimilarity')
        homogeneity = graycoprops(glcm, prop='homogeneity')
        energy = graycoprops(glcm, prop='energy')
        texture_stats = [contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0]]
        features += band_stats + texture_stats # band_stats包含最小值，最大值，平均值，variance，skewness，kurtosis; texture_stats 包含contrast, dissimilarity, homogeneity, energy
        """
        features += band_stats
    return features 


# ------------------ Worker function ------------------
def process_segment(args):
    segment_id, shared_names, shared_shapes, shared_dtypes = args
    # Access shared memory
    existing_img = shared_memory.SharedMemory(name=shared_names['img'])
    img_np = np.ndarray(shared_shapes['img'], dtype=shared_dtypes['img'], buffer=existing_img.buf)

    existing_segments = shared_memory.SharedMemory(name=shared_names['segments'])
    segments_np = np.ndarray(shared_shapes['segments'], dtype=shared_dtypes['segments'], buffer=existing_segments.buf)

    # Select pixels for this segment and compute stats
    # (segments_np == segment_id).shape = (2000, 5834)，segments_np == segment_id的值为True或False，即表示某个位置处的像素是否属于指定segment。
    # segment_pixels.shape = (x,4), x为像素点的个数
    mask = (segments_np == segment_id)
    segment_pixels = img_np[mask] # 得到图片中都属于同一个segment_id的像素。img_np.shape = (2000, 5834, 4) 2000是y轴方向的长度，5834是x轴
    object_features = segment_features(segment_pixels)
    # 考虑segment的形状：segment最小外接矩形的长宽比
    binary_img = np.array(mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    ratio = rect[1][1] / rect[1][0] # 长：宽
    object_features.append(ratio)
    # Cleanup (close but don't unlink)
    existing_img.close()
    existing_segments.close()

    return (segment_id, object_features)

# 从标点(已分类)的.shp文件中获得对应segment的特征和分类
def get_shapefile_objects_features_and_labels(base_img_ds, shp_fn, all_obj_features, segment_ids, segments):
    ogr_ds = ogr.Open(shp_fn)
    lyr = ogr_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', base_img_ds.RasterXSize, base_img_ds.RasterYSize, 1, gdal.GDT_UInt16)
    raster_ds.SetGeoTransform(base_img_ds.GetGeoTransform())
    raster_ds.SetProjection(base_img_ds.GetProjection())
    # rasterize the test points
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(raster_ds, [1], lyr, options=options)
    # retrieve the rasterized data and print basic stats
    data = raster_ds.GetRasterBand(1).ReadAsArray() # test_ds 即 test.shp中的数据
    print(np.unique(data)) # [0 1 2 3 4 5 6 7] 0代表未分类
    print('min', data.min(), 'max', data.max(), 'mean', data.mean())
    truth = raster_ds.GetRasterBand(1).ReadAsArray() # truth的唯一值：[0 1 2 3 4 5 6 7], truth.shape: (2000, 5834)

    # Get segments representing each land cover classification type and ensure no segment represents more than one class.
    classes = np.unique(truth)[1:] # [1 2 3 4 5 6 7]
    print('class values', classes)
    segments_per_class = {}

    for klass in classes: # classes中没有0
        segments_of_class = segments[truth == klass] # 找到属于klass的segments
        segments_per_class[klass] = set(segments_of_class)
        print("segments for class", klass, ":", len(segments_of_class))

    intersection = set()
    accum = set() # 记录所有segment ID

    for class_segments in segments_per_class.values():
        intersection |= accum.intersection(class_segments)
        accum |= class_segments # 将class_segments加入到accum
    assert len(intersection) == 0, "Segment(s) represent multiple classes"

    # ------------------Classify the image-----------------#
    obj_features = []
    labels = []
    for klass in classes:
        # all_obj_features中的一个object有40个数：4个band，每个band有6个光谱特征，以及4个纹理特征
        # segment_ids：[1  2  3 ... 40148 40149 40150]，all_obj_features与segment_ids等长
        # i从0开始,到40149结束。v表示一个segment的光谱特征，光谱特征
        # i=0, segment_ids[0] = 1
        # object_id = [1, 2, 3, 4, ...,40150]

        # 遍历所有segment ID，如果segment ID属于klass，则把它的光谱特征和纹理特征保存。
        class_object = [v for i, v in enumerate(all_obj_features) if segment_ids[i] in segments_per_class[klass]]
        labels += [klass] * len(class_object) # [klass] * len(class_train_object) 会得到一个长度为len(class_train_object)的列表，值为klass
        obj_features += class_object # obj_features[0]代表klass=1的光谱数据
    return obj_features, labels
    
# ------------------ Main ------------------#
if __name__ == '__main__':
    # Load image
    # naip_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624.tif'
    naip_fn = r"D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624_subset.tif"
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn)
    assert naip_ds, "naip_ds is None"

    nbands = naip_ds.RasterCount
    band_data = []
    for i in range(1, nbands+1):
        band = naip_ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    band_data = np.dstack(band_data)

    # Scale image values to 0.0 - 1.0
    img = exposure.rescale_intensity(band_data)
    # Segmentation
    # segments = slic(img, n_segments=500000, compactness=0.1)
    """
    segments = slic(img, n_segments=50000, compactness=0.1)
    print(segments.shape)

    # 保存segments，以便后续调试程序
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\segments_v2.pkl", "wb") as f:
        pickle.dump(segments, f)
    """
    # 取出 segments
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\segments_v2.pkl", "rb") as f:
        segments = pickle.load(f)
    # Save segments raster (optional)
    segments_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\segments_final_v2.tif'

    segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,
                                    1, gdal.GDT_Float32)
    segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    segments_ds.SetProjection(naip_ds.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    del segments_ds
    del naip_ds
    
    segment_ids = np.unique(segments)
    """
    # 如果已经计算过，可以注释掉！！！
    # ------------------ Step 4: Setup shared memory ------------------
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

   
    # ------------------ Step 5: Parallel execution ------------------
    segment_ids = np.unique(segments)
    print(f"Processing {len(segment_ids)} segments in parallel...")
    start_time = time.time()
    args_list = [(segment_id, shared_names, shared_shapes, shared_dtypes) for segment_id in segment_ids]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_segment, args_list, timeout=3600))
    print(f"Feature extraction done in {time.time() - start_time:.2f} seconds")
    print(results)
    # ------------------ Step 6: Cleanup ------------------
    object_ids, objects = zip(*results) # object_ids的元素是segment_id，objects的元素是一个segment的光谱特征
    print(objects)

    # 保存一下object_ids, objects，否则每次运行，都要等很久
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\segment_features_v2.pkl", "wb") as f:
        pickle.dump((object_ids, objects), f)
    
    shm_img.close()
    shm_img.unlink()
    shm_segments.close()
    shm_segments.unlink()
    print("Parallization Finished.")
    """
    # 加载已有的object_ids, objects
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\segment_features_v2.pkl", "rb") as f:
        object_ids, objects = pickle.load(f)
    # read shapefile to geopandas geodataframe
    gdf = gpd.read_file(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\truth_data_subset_utm12\truth_data_subset_utm12.shp')
    # get names of land cover classes/labels
    class_names = gdf['lctype'].unique()
    print('class names', class_names)

    # create a unique id (integer) for each land cover class/label
    class_ids = np.arange(class_names.size) + 1
    print('class ids', class_ids)

    # create a pandas data frame of the labels and ids and save to csv
    df = pd.DataFrame({'label': class_names, 'id': class_ids})
    df.to_csv(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\class_lookup.csv')

    print('gdf without ids\n', gdf.head())
    # add a new column to geodatafame with the id for each class/label
    gdf['id'] = gdf['lctype'].map(dict(zip(class_names, class_ids)))
    print('gdf with ids\n', gdf.head())
    
    # split the truth data into training and test data sets and save each to a new shapefile
    gdf_train = gdf.sample(frac=0.7, random_state=0)
    gdf_test = gdf.drop(gdf_train.index)
    print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
    gdf_train.to_file(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\train.shp')
    gdf_test.to_file(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\test.shp')

    #----------------- Rasterize Training Data---------------#
    # open NAIP image as a gdal raster dataset
    naip_ds = gdal.Open(naip_fn)
    """
    # 若用之前已经训练过的模型，直接加载即可
    # open the points file to use for training data
    training_objects, training_labels = get_shapefile_objects_features_and_labels(naip_ds, r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\train.shp',objects, segment_ids, segments)

    classifier = RandomForestClassifier(n_jobs=-1, random_state=0)
    classifier.fit(training_objects, training_labels) # training_objects是从objects得到的。
    print('Fitting Random Forest Classifier')
    predicted = classifier.predict(objects) # objects是从所有segments计算得到。前面用objects部分数据进行训练，这里用该模型预测所有的objects。这种做法如果只是为了生成最终分类图，则没有问题；但后续如果评估模型，就必须去除训练集，只预测模型没见过的测试集
    print('Predicting Classifications')

    # 保存模型
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\random_forest_model_v2.pkl", 'wb') as f:  # 二进制写入模式
        pickle.dump(classifier, f)
    """
    # 加载模型
    print("load model ...")
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\random_forest_model_v2.pkl", "rb") as f:
        classifier = pickle.load(f)
    print("load model successfully")
    """
    # 保存预测结果
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\predicted_v2.pkl", "wb") as f:
        pickle.dump(predicted, f)
    """
    # 加载预测结果
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\predicted_v2.pkl", "rb") as f:
        predicted = pickle.load(f)
    print("load predicted successfully")
    clf = np.copy(segments) # clf.shape = (2000, 5834)
    """
    # 下面这个循环需要运行5min
    for segment_id, klass in zip(segment_ids, predicted): # segment_ids=[1, 2, 3,..., 40150]，
        # predicted与segment_ids一一对应
        clf[clf == segment_id] = klass # clf：即每个像素的分类（land cover）结果
    # 保存clf，便于后续调试
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\clf_v2.pkl", "wb") as f:
        pickle.dump(clf, f)
    """

    # 加载clf
    with open(r"D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\pkl\obia\clf_v2.pkl", "rb") as f:
        clf = pickle.load(f)
    print('Prediction applied to numpy array')
    mask = np.sum(img, axis=2) # 对每个像素的所有波段求和，结果是一个二维矩阵。mask.shape = (2000, 5834) 2000代表y轴方向，5834代表x轴方向
    mask[mask > 0.0] = 1.0 # mask > 0.0表示该像素点有数据
    mask[mask == 0.0] = -1.0 # mask==0，表示该像素点没有数据
    clf = np.multiply(clf, mask) # 不是线代中的矩阵乘法，而是两个矩阵相同位置的两个元素相乘
    clf[clf < 0] = -9999.0

    print('Saving classificaiton to raster with gdal')
    clfds = driverTiff.Create(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\classified_v2.tif', naip_ds.RasterXSize, naip_ds.RasterYSize,
                            1, gdal.GDT_Float32)
    clfds.SetGeoTransform(naip_ds.GetGeoTransform())
    clfds.SetProjection(naip_ds.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clfds.GetRasterBand(1).WriteArray(clf)
    del clfds
    print('Done!')

    # -----------------预测准确度计算----------------
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn)

    test_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\test.shp'
    test_ds = ogr.Open(test_fn)
    lyr = test_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    target_ds.SetProjection(naip_ds.GetProjection())
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

    # test.shp中的数据，test.shp来自你在qgis取的所有点
    truth = target_ds.GetRasterBand(1).ReadAsArray()
    print("Test truth Data Shape=", truth.shape) # (2000, 5834)
    print("Unique Test truth Data= ", np.unique(truth)) # [0 1 2 3 4 5 6 7]
    
    pred_ds = gdal.Open(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\classified_v2.tif')
    pred = pred_ds.GetRasterBand(1).ReadAsArray()

    idx = np.nonzero(truth)
    cm = metrics.confusion_matrix(truth[idx], pred[idx]) # test.shp中数据标签vs模型预测的结果
    # pixel accuracy
    print("Confusion Matrix: ")
    print(cm)
    print(cm.diagonal())
    print(cm.sum(axis=0))
    accuracy = cm.diagonal() / cm.sum(axis=0) # 预测的准确率
    print("accuracy: \n", accuracy)

    from sklearn.metrics import classification_report
    print("======classification_report starts=====")
    print(classification_report(truth[idx], pred[idx]))
    print("======classification_report end======")

    # ----------------用AUC的方法来评价模型-------------
    # 对于train.shp中的数据，获得对应segment的光谱特征和纹理特征，以及对应的landcover分类，训练模型
    # 对于test.shp中的数据，获得对应segment的光谱特征和纹理特征，以及对应的landcover分类
    
    test_objects, test_labels = get_shapefile_objects_features_and_labels(naip_ds, r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\test.shp', objects, segment_ids, segments)

    y_scores = classifier.predict_proba(test_objects)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_labels, y_scores, multi_class="ovo") 
    print("auc: ",auc) 

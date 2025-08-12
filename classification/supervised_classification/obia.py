# TODO 问题：没有用到纹理和形状，只用到了光谱信息
import numpy as np
from osgeo import gdal,ogr
from skimage import exposure
from skimage.segmentation import slic
import scipy
import time
from multiprocessing import shared_memory
import concurrent.futures
import pickle
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# ------------------ Step 1: Compute segment features ------------------
def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape # segment_pixels.shape的值是 (某个数字, 4)，某个数字即像素点的个数
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b]) # segment_pixels[:, b]即取b列的所有行
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features # 包含最小值，最大值，平均值，variance，skewness，kurtosis

# ------------------ Step 2: Worker function ------------------
def process_segment(args):
    segment_id, shared_names, shared_shapes, shared_dtypes = args
    # Access shared memory
    existing_img = shared_memory.SharedMemory(name=shared_names['img'])
    img_np = np.ndarray(shared_shapes['img'], dtype=shared_dtypes['img'], buffer=existing_img.buf)

    existing_segments = shared_memory.SharedMemory(name=shared_names['segments'])
    segments_np = np.ndarray(shared_shapes['segments'], dtype=shared_dtypes['segments'], buffer=existing_segments.buf)

    # Select pixels for this segment and compute stats
    segment_pixels = img_np[segments_np == segment_id] # 得到图片中都属于同一个segment_id的像素。img_np.shape = (2000, 5834, 4) 2000是y轴方向的长度，5834是x轴
    object_features = segment_features(segment_pixels)

    # Cleanup (close but don't unlink)
    existing_img.close()
    existing_segments.close()

    return (segment_id, object_features)

# ------------------ Step 3: Main ------------------
if __name__ == '__main__':
    # Load image
    # naip_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624.tif'
    naip_fn = r"D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624_subset.tif"
    driverTiff = gdal.GetDriverByName('GTiff')
    naip_ds = gdal.Open(naip_fn)
    if naip_ds is None:
        print("naip_ds is None")
        exit()
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
    segments = slic(img, n_segments=50000, compactness=0.1)
    """
    # 保存segments，以便后续调试程序
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\segments.pkl", "wb") as f:
        pickle.dump((segments), f)
    """
    # 取出 segments
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\segments.pkl", "rb") as f:
        segments = pickle.load(f)
    # Save segments raster (optional)
    segments_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\segments_final.tif'

    segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,
                                    1, gdal.GDT_Float32)
    segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    segments_ds.SetProjection(naip_ds.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    del segments_ds
    del naip_ds

    segment_ids = np.unique(segments)
    # 暂时注释掉！！！
    """
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

    # ------------------ Step 6: Cleanup ------------------
    object_ids, objects = zip(*results) # objects的元素是一个segment的若干特征，比如：光谱

    # 保存一下object_ids, objects，否则每次运行，都要等很久
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\segment_features.pkl", "wb") as f:
        pickle.dump((object_ids, objects), f)

    shm_img.close()
    shm_img.unlink()
    shm_segments.close()
    shm_segments.unlink()
    print("Parallization Finished.")
    """
    # 加载已有的object_ids, objects
    with open("D:\Projects\VsCode\Python\img_processing_system\classification\supervised_classification\segment_features.pkl", "rb") as f:
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
    gdf_train = gdf.sample(frac=0.7)
    gdf_test = gdf.drop(gdf_train.index)
    print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
    gdf_train.to_file(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\train.shp')
    gdf_test.to_file(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\test.shp')

    #----------------- Rasterize Training Data---------------#
    # open NAIP image as a gdal raster dataset
    naip_ds = gdal.Open(naip_fn)

    # open the points file to use for training data
    train_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\train.shp'
    train_ds = ogr.Open(train_fn)
    lyr = train_ds.GetLayer()

    # create a new raster layer in memory
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    target_ds.SetProjection(naip_ds.GetProjection())
    
    # rasterize the training points
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    # retrieve the rasterized data and print basic stats
    data = target_ds.GetRasterBand(1).ReadAsArray() # target_ds 即 train.shp中的数据???
    print('min', data.min(), 'max', data.max(), 'mean', data.mean())
    print(np.unique(data)) # [0 1 2 3 4 5 6 7] TODO 为什么有8种数？
    ground_truth = target_ds.GetRasterBand(1).ReadAsArray() # 唯一值：[0 1 2 3 4 5 6 7]
    print(ground_truth.shape)
    # Get segments representing each land cover classification type and ensure no segment represents more than one class.
    classes = np.unique(ground_truth)[1:]
    print('class values', classes)
    segments_per_class = {}

    for klass in classes: # classes中没有0
        segments_of_class = segments[ground_truth == klass] 
        print(segments_of_class)
        print(np.unique(segments_of_class))
        exit()
        segments_per_class[klass] = set(segments_of_class)
        print("Training segments for class", klass, ":", len(segments_of_class))

    intersection = set()
    accum = set()

    for class_segments in segments_per_class.values():
        intersection |= accum.intersection(class_segments)
        accum |= class_segments
    assert len(intersection) == 0, "Segment(s) represent multiple classes"

    # ------------------Classify the image-----------------#
    train_img = np.copy(segments)
    threshold = train_img.max() + 1

    for klass in classes:
        class_label = threshold + klass
        for segment_id in segments_per_class[klass]:
            train_img[train_img == segment_id] = class_label

    train_img[train_img <= threshold] = 0
    train_img[train_img > threshold] -= threshold

    training_objects = []
    training_labels = []

    for klass in classes:
        class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
        training_labels += [klass] * len(class_train_object)
        training_objects += class_train_object
        print('Training objects for class', klass, ':', len(class_train_object))

    classifier = RandomForestClassifier(n_jobs=-1)
    classifier.fit(training_objects, training_labels)
    print('Fitting Random Forest Classifier')
    predicted = classifier.predict(objects)
    print('Predicting Classifications')

    # Prediction操作挺耗时
    clf = np.copy(segments)
    for segment_id, klass in zip(segment_ids, predicted):
        clf[clf == segment_id] = klass

    print('Prediction applied to numpy array')

    mask = np.sum(img, axis=2)
    mask[mask > 0.0] = 1.0
    mask[mask == 0.0] = -1.0
    clf = np.multiply(clf, mask)
    clf[clf < 0] = -9999.0

    print('Saving classificaiton to raster with gdal')
    clfds = driverTiff.Create(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\classified.tif', naip_ds.RasterXSize, naip_ds.RasterYSize,
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

    truth = target_ds.GetRasterBand(1).ReadAsArray()

    pred_ds = gdal.Open(r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\classified.tif')
    pred = pred_ds.GetRasterBand(1).ReadAsArray()

    idx = np.nonzero(truth)

    cm = metrics.confusion_matrix(truth[idx], pred[idx])

    # pixel accuracy
    print(cm)

    print(cm.diagonal())
    print(cm.sum(axis=0))

    accuracy = cm.diagonal() / cm.sum(axis=0)
    print(accuracy)
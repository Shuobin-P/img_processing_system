import cv2
import scipy
import numpy as np
from multiprocessing import shared_memory

def segment_features(segment_pixels): 
    """ 
    计算一个segment的像素的所有band的光谱特征，纹理特征
    """
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

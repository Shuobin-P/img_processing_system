import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import slic
import scipy
import time
from multiprocessing import shared_memory
import concurrent.futures

# ------------------ Step 1: Compute segment features ------------------
def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            band_stats[3] = 0.0
        features += band_stats
    return features

# ------------------ Step 2: Worker function ------------------
def process_segment(segment_id):
    # Access shared memory
    existing_img = shared_memory.SharedMemory(name=shared_names['img'])
    img_np = np.ndarray(shared_shapes['img'], dtype=shared_dtypes['img'], buffer=existing_img.buf)

    existing_segments = shared_memory.SharedMemory(name=shared_names['segments'])
    segments_np = np.ndarray(shared_shapes['segments'], dtype=shared_dtypes['segments'], buffer=existing_segments.buf)

    # Select pixels for this segment and compute stats
    segment_pixels = img_np[segments_np == segment_id]
    object_features = segment_features(segment_pixels)

    # Cleanup (close but don't unlink)
    existing_img.close()
    existing_segments.close()

    return (segment_id, object_features)

# ------------------ Step 3: Main ------------------
if __name__ == '__main__':
    # Load image
    naip_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\m_4211161_se_12_1_20160624\m_4211161_se_12_1_20160624.tif'
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
    segments = slic(img, n_segments=500000, compactness=0.1)

    # Save segments raster (optional)
    segments_fn = r'D:\Projects\VsCode\Python\img_processing_system\qgis_image\naip\segments_final.tif'
    segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,
                                    1, gdal.GDT_Float32)
    segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    segments_ds.SetProjection(naip_ds.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    segments_ds = None
    del naip_ds

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
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_segment, segment_ids))
    print(f"Feature extraction done in {time.time() - start_time:.2f} seconds")

    # ------------------ Step 6: Cleanup ------------------
    object_ids, objects = zip(*results)

    shm_img.close()
    shm_img.unlink()
    shm_segments.close()
    shm_segments.unlink()

    print("Finished.")

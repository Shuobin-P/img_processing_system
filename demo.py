import numpy as np
from multiprocessing import shared_memory
import concurrent.futures
import pickle

def process_segment(args):
    segment_id, shared_names, shared_shapes, shared_dtypes = args
    print(segment_id, shared_names, shared_shapes, shared_dtypes)
    return (segment_id, ["This is a attribute"])

if __name__ == "__main__":
    segment_ids = list(range(1, 20))
    shared_names = {
        'img': "xxx_img_name",
        'segments': "yyy_segments_name"
    }
    shared_shapes = {
        'img': (5,6),
        'segments': (7, 8)
    }
    shared_dtypes = {
        'img': int,
        'segments': int
    }
    args_list = [(segment_id, shared_names, shared_shapes, shared_dtypes) for segment_id in segment_ids]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_segment, args_list))
    object_ids, objects = zip(*results)
    with open("D:\Projects\VsCode\Python\img_processing_system\parallelization\segment_features.pkl", "wb") as f:
        pickle.dump((object_ids, objects), f)

    # 加载
    with open("D:\Projects\VsCode\Python\img_processing_system\parallelization\segment_features.pkl", "rb") as f:
        object_ids, objects = pickle.load(f)
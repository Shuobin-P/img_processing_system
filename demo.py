import numpy as np
import scipy

if __name__ == "__main__":
    segment_pixels = [
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3]
    ]
    arr = np.array(segment_pixels)
    stats = scipy.stats.describe(arr[:, 0])
    print(list(stats.minmax))
    print(list(stats)[2:])
    print(list(stats.minmax) + list(stats)[2:])


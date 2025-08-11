import numpy as np
from multiprocessing import shared_memory
import concurrent.futures

# ==============================
# Step 1: Create shared memory in main process
# ==============================
data = np.arange(10, dtype=np.int32)  # Our example array

# Create a shared memory block big enough for 'data'
shm = shared_memory.SharedMemory(create=True, size=data.nbytes)

# Create a NumPy array backed by shared memory
shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shared_array[:] = data[:]  # copy initial data into shared memory

# Metadata to send to workers
meta = {
    "name": shm.name,           # Name of shared memory block
    "shape": data.shape,        # Shape of array
    "dtype": data.dtype         # Data type
}

# ==============================
# Step 2: Worker function
# ==============================
def worker(index, value, meta):
    # Connect to existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=meta["name"])
    arr = np.ndarray(meta["shape"], dtype=meta["dtype"], buffer=existing_shm.buf)

    # Modify the array in-place
    arr[index] = value

    # Close (but do not unlink — main process owns it)
    existing_shm.close()

# ==============================
# Step 3: Run workers in parallel
# ==============================
if __name__ == "__main__":
    print("data: ",data)
    print("Before:", shared_array[:])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Each worker sets a different element to 100+index
        for i in range(len(shared_array)):
            executor.submit(worker, i, 100 + i, meta)

    print("After:", shared_array[:])

    # Cleanup: release and destroy the shared memory
    shm.close()
    shm.unlink()

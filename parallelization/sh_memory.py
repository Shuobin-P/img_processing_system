# In the first Python interactive shell
import numpy as np
a = np.array([1, 1, 2, 3, 5, 8])  # Start with an existing NumPy array
from multiprocessing import shared_memory
shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
# Now create a NumPy array backed by shared memory
b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
b[:] = a[:]  # Copy the original data into shared memory
b

type(b)

type(a)

shm.name  # We did not specify a name so one was chosen for us


# In either the same shell or a new Python shell on the same machine
import numpy as np
from multiprocessing import shared_memory
# Attach to the existing shared memory block
# Use the actual shared memory name from the previous step
existing_shm = shared_memory.SharedMemory(name=shm.name)
# Note that a.shape is (6,) and a.dtype is np.int64 in this example
c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
c

c[-1] = 888
print(c)


# Back in the first Python interactive shell, b reflects this change
print(b)


# Clean up from within the second Python shell
del c  # Unnecessary; merely emphasizing the array is no longer used
existing_shm.close()

# Clean up from within the first Python shell
del b  # Unnecessary; merely emphasizing the array is no longer used
shm.close()
shm.unlink()  # Free and release the shared memory block at the very end



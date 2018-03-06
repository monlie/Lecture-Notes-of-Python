# -*- coding: utf-8 -*-
"""
Created on 2017/9/25 14:30:43

@author: 李蒙
"""

from numba import cuda
import numpy as np

an_array = np.arange((20))

threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock


# device
@cuda.jit(device=True)
def h():
    pass

# global
@cuda.jit
def convolution(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1

convolution[blockspergrid, threadsperblock](an_array)
print(an_array)
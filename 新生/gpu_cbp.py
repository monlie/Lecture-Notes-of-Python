# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:56:59 2017

@author: 李蒙
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from get_data import open_exl
from create_img import img_trans
import matplotlib.pyplot as plt
from timer import run_time


@run_time
def gpu_convolution(spec):
    spec = spec.astype(np.float32)
    n, m = spec.shape
    x, y = n//32+1, m//32+1
    mod = SourceModule('''
          __device__ float h(int n)
          {   
              const float pi = 3.141592;
              return -2/(pi*pi * (4*n*n-1));
          }
    
    
          __global__ void convolution(float* sig, const float* spec)
          {
              int n = 512;
              int m = 180;
              int i = blockIdx.x*32 + threadIdx.x;
              int j = blockIdx.y*32 + threadIdx.y;
              if (i < n && j < m)
              {
                  int idx = i*m + j;
                  float g = 0;
                  for (int k=0; k<n; k++)
                  {
                      g += spec[k*m+j]*h(i-k);
                  }
                  sig[idx] = g;
              }
          }
          ''')
    sig = np.empty((n, m), dtype=np.float32)
    func = mod.get_function("convolution")
    func(cuda.Out(sig), cuda.In(spec), block=(32, 32, 1), 
         grid=(x, y))
    return sig


@run_time
def gpu_bp(sig):
    sig = sig.astype(np.float32)
    x, y = 512//32+1, 512//32+1
    mod = SourceModule('''
          __global__ void bp(float* img, const float* sig)
          {
              const float pi = 3.141592;
              float theta, d, r;
              int x, y, left;
              int n = 512;
              int m = 180;
              int i = blockIdx.x*32 + threadIdx.x;
              int j = blockIdx.y*32 + threadIdx.y;
              if (i < n && j < n)
              {
                  int idx = i*n + j;
                  float px = 0;
                  for (int k=0; k<m; k++)
                  {
                      theta = k*pi/180;
                      x = j-255;
                      y = 511-i-255;
                      d = x*cos(theta-pi/2)+y*sin(theta-pi/2)+255;
                      if (0 <= d && d < n-1)
                      {  
                          left = int(d);
                          r = d-left;
                          px += (1-r)*sig[left*180+k]+r*sig[(left+1)*180+k];
                      }
                  }
                  img[idx] = px/m;
              }
          }
          ''')
    img = np.empty((512, 512), dtype=np.float32)
    func = mod.get_function("bp")
    func(cuda.Out(img), cuda.In(sig), block=(32, 32, 1), 
         grid=(x, y))
    return img


if __name__ == '__main__':
    spec = open_exl('A.xls', 1)
    sig = gpu_convolution(spec)
    img = img_trans(gpu_bp(gpu_convolution(spec)), shape=(512, 512))
    plt.imshow(img)
    plt.show()
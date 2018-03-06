# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 00:33:58 2017

@author: 李蒙
"""
import math
import numpy as np
from get_data import *
from timer import run_time

#rl滤波器
def rl(n):
    if n == 0:
        return 0.25
    elif n%2 == 0:
        return 0
    else:
        return -1/(math.pi*n)**2

#sl滤波器
def sl(n):
    return -2/(math.pi**2 * (4*n**2-1))
    
#卷积
@run_time
def convolution(data, func):
    n, m = data.shape
    new = np.empty(shape=(n, m))
    for i in range(n):
        for j in range(m):
            g = 0
            for k in range(n):
                g += data[k, j]*func(i-k)
            new[i, j] = g
    return new
    
#极坐标反投影
def backprojection(data):
    n, m = data.shape
    img = np.empty(shape=(n, 360))
    for theta in range(360):
        for r in range(n):
            px = 0
            for phi in range(m):
                x = r*math.cos((theta-phi)*math.pi/180)+255
                if 0 <= x <n-1:
                    idx = math.floor(x)
                    k = x-idx
                    px += data[idx, phi] + k*(data[idx+1, phi]-data[idx, phi])
            img[r, theta] = px/m
    return img

#直角坐标反投影
@run_time
def backprojection2(data):
    n, m = data.shape
    img = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(n):
            px = 0
            for k in range(180):
                theta = k*math.pi/180
                x = j-255
                y = 511-i-255
                d = x*math.cos(theta-math.pi/2)+y*math.sin(theta-math.pi/2)+255
                if 0 <= d < n-1:
                    idx = math.floor(d)
                    r = d-idx
                    px += (1-r)*data[idx, k]+r*data[idx+1, k]
            img[i, j] = px/m
    return img
                
#载物台坐标系变换到ct坐标系
def coord_tran(x, phi=119):
    k = 1/0.2758
    a = np.array((-9.1667, 6.6667))
    phi = phi/180*math.pi
    mat = np.array(((math.cos(phi), -math.sin(phi)),
                    (math.sin(phi), math.cos(phi))))
    x = k*(np.dot(x-a, mat))
    x =[x[0]+255, 255-x[1]]
    return x

#图像变换
def img_trans(img, phi=119, shape=(256, 256)):
    d = 0.390625*256/shape[0]
    new = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            px = 0
            x = [(j-shape[1]/2+1)*d, (i-shape[1]/2+1)*d]
            x, y = coord_tran(x, phi)
            if 0 <= x < 511 and 0 <= y < 511:
                idx_x = math.floor(x)
                idx_y = math.floor(y)
                k_x = x-idx_x
                k_y = y-idx_y
                a1 = img[idx_y+1, idx_x]-img[idx_y, idx_x]
                a2 = img[idx_y, idx_x+1]-img[idx_y, idx_x]
                px = img[idx_y, idx_x] + a1*k_y + a2*k_x
            new[i, j] = px
    return new
    
#radon变换
def radon(img, n=512, m=180, theta=29):
    a = 0.56586812071847759
    o = np.array((-9.1667, 6.6667))+np.array((50, 50))
    k = 0.2758
    d = 0.390625/2
    def line_func(s, phi):
        s = (s-255)*k
        phi = (phi+theta)*math.pi/180
        #射线方程
        def line(t):
            t = t*d
            ox = o[0]+s*math.cos(phi)
            oy = o[1]+s*math.sin(phi)
            x = ox + t*math.cos(phi+math.pi/2)
            y = oy + t*math.sin(phi+math.pi/2)
            return x, y
        return line
    
    project = np.empty((n, m))
    for s in range(n):
        for phi in range(m):
            t = -n
            px = 0
            while 1:
                if t > int(n):
                    break
                x, y = line_func(s, phi)(t)
                x, y = x/d, y/d
                if 0 <= x <511 and 0 <= y <511:
                    idxi = math.floor(x)
                    idxj = math.floor(y)
                    ri = x-idxi
                    rj = y-idxj
                    px += (img[idxj, idxi]+
                           rj*(img[idxj+1, idxi]-img[idxj, idxi])+
                           ri*(img[idxj, idxi+1]-img[idxj, idxi]))
                t += 1
            project[s, phi] = px*d
    return project/a
            

    
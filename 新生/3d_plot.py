# -*- coding: utf-8 -*-
"""
Created on 2017/10/5 20:53:33

@author: 李蒙
"""
import numpy as np
from get_data import open_exl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


m = open_exl('pubg_3.xls', 0)


x,y,z = m[:,0],m[:,1],m[:,2]

plt.figure(figsize=(16, 9))
ax=plt.subplot(111,projection='3d')

#将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, s=15)


ax.set_zlabel('Headshot Kill Ratio') #坐标轴
ax.set_ylabel('KDA')
ax.set_xlabel('Win Rate')
plt.show()
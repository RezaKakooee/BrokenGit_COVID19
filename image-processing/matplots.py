# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:59:13 2020

@author: reza
"""
import numpy as np
import matplotlib.pyplot as plt#, mpld3
from mpl_toolkits.mplot3d import Axes3D
from Annotate_Images_to3D import ImageAnnotations3D

def plotting(tsne_val, imgs):
    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    xs = np.array(tsne_val[:,0]).reshape((-1,1))
    ys = np.array(tsne_val[:,1]).reshape((-1,1))
    zs = np.array(tsne_val[:,2]).reshape((-1,1))
    ax.scatter(xs, ys, zs, marker=".", s=0.1)
    ax2 = fig.add_subplot(111,frame_on=False) 
    ax2.axis("off")
    ax2.axis([0,1,0,1])
    ia = ImageAnnotations3D(np.c_[xs,ys,zs], imgs, ax, ax2)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
#        ax.set_title(title)
    plt.show()
        
#    return fig, ax
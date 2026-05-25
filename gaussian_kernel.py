# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 10:52:40 2024

@author: Sudipta
"""



import numpy as np
import math
import cv2 as cv

# gaussian filter
def gfunc(x,y,sigma):
    return (math.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2,j-size[1]//2, sigma )
    return out/np.sum(out)

(gfw,gfh) = (3,3)
gaussianFilter = gaussFilter((gfw,gfh),1)




"""
import math
import cv2
import numpy as np

def gkernel(l, sig):
    """\
   """ Gaussian Kernel Creator via given length and sigma
    """

   # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    #xx, yy = np.meshgrid(ax, ax)
#
 #   kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

  #  return kernel / np.sum(kernel)

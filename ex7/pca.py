#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:08:20 2019

@author: zoeyyang
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import random
import sys
from sklearn.decomposition import PCA
import math
from mpl_toolkits.mplot3d import Axes3D

def loaddata2():
    img_content=tf.read_file("./smallbird.png")
    image=tf.image.decode_png(img_content,channels=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img=sess.run((image)) #img 为三维数组 (width,height,color) color:[r,g,b]
        plt.imshow(img)
    return img, np.shape(img)
               
def loaddata():
    m=loadmat("./ex7faces.mat")
    faces=m["X"]
    return faces
    
def myPCA(x):
    x[:,0]=(x[:,0]-np.mean(x[:,0]))/np.std(x[:,0])
    x[:,1]=(x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
    m=np.shape(x)[0]
    sigma=np.dot(np.mat(x.T),np.mat(x))/m
    U,S,V=np.linalg.svd(sigma)
    U_reduce=U[:,0:k]
    Z=U_reduce.T
    
    print(S)
    return U
def main():
#    x,shape=loaddata2()
#    decomp=PCA(n_components=100)
#    decomp=PCA()
#    newdata=decomp.fit_transform(x)
#    
#    recover=decomp.inverse_transform(newdata)
#    plt.plot(recover)
#    
    f=loaddata()
    m,n=np.shape(f)
    decomp=PCA(n_components=100)
    newf=decomp.fit_transform(f)
    recoverf=decomp.inverse_transform(newf)
    
    width=32
    height=int(n/width)
    row=math.floor(math.sqrt(m))
    col=math.ceil(m/row)
    pad=1
    
#    display_array=np.zeros((pad+row*(height+pad),pad+col*(width+pad)))
    
    display_array=np.zeros((row*int(height),col*int(width)))# 形成画布，
    
    print(np.shape(display_array), (m,n))
    for i in range(row):
        for j in range(col):
            try:
                display_square=np.reshape(recoverf[i*col+j,:],(32,32))# 方格填入
                display_array[i*width:(i+1)*width,j*height:(j+1)*height]=display_square
            except IndexError:
                break
     
    
    
    plt.imshow(display_array.T,origin='upper', cmap='gray')
    plt.xlim([0,10*32])#zoom in
    plt.ylim([10*32,0])#
    plt.show()
   
    
    
    
    
main()
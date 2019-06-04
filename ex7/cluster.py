#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:14:30 2019

@author: zoeyyang
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import random
import sys
import tensorflow as tf

def loaddata():
    m=loadmat("./ex7data2.mat")
    x=m["X"]
    return x;

def findclosestcentroids(x,u):
    m=np.shape(x)[0]
    mu=np.shape(u)[0]
    
    c=[]
    total_distance=0
    for i in range(m):
        min_uk=99999
        for j in range(mu):
            a=x[i,0]-u[j,0]
            b=x[i][1]-u[j,1]
            distance = np.square(a)+np.square(b)
            if (distance<min_uk):
                min_uk = distance
                min_j=j
        total_distance += min_uk
        c.append(min_j)
        
    return c, total_distance

def updatecentroid(x,c,u):
    mu=np.shape(u)[0]
    for j in range(mu):
        all_uki=0
        num=0
        for i,el in enumerate(c):
            if (el==j):
                num +=1
                all_uki += x[i]
        if(num!=0):
            u[j]=all_uki/num  # if num==0 condition happen??
    return u
            
def plotdata(x,u):
    plt.plot(x[:,0],x[:,1],"r^")
    plt.plot(u[:,0],u[:,1],"go")
    plt.show                    
     
def randominit_u(x,k):
    m=np.shape(x)[0]
    size=round(m/k)
    init_u=[]
    for i in range(k):
        if (i==0):
            init_u= x[random.randint(i*1,(i+1)*size-1),:]
        else:
            init_u=np.vstack((init_u,x[random.randint(i*1,(i+1)*size-1),:]))#gotta prevent the same x
    return init_u
 
def loadimg():
    img_content=tf.read_file("./bird_small.png")
    image=tf.image.decode_png(img_content,channels=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img=sess.run((image)) #img 为三维数组 (width,height,color) color:[r,g,b]
#        plt.imshow(img)
    return img, np.shape(img)

def loaddata2():
    m=loadmat("./ex7data1.mat")
    x=m["X"]
    return x;
           
def PCA(x):
    x[:,0]=(x[:,0]-np.mean(x[:,0]))/np.std(x[:,0])
    x[:,1]=(x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
    print(x)
    U,S,V=np.linalg.svd(x)
    print(np.shape(S))
    return U
    
    


       
def main():
# =============================================================================
#     k=3
#     data=loaddata()
#     m=np.shape(data)[0]
#     u=randominit_u(data,k)
#     
# #    u=np.mat([(data[random.randint(0,100),:]),(data[random.randint(101,200),:]),(data[random.randint(200,299),:])])   
#     for run in range(50):
# #        plotdata(data,init_centroid)
#         c=findclosestcentroids(data,u)
#         u=updatecentroid(data,c,u)
# #    print(c)
#         plotdata(data,u)
#   我这个顺序是不是弄错了，感觉没有跟着课程那样做啊。
# =============================================================================
# =============================================================================
     k=16
     img,imgshape=loadimg()
     img_reshape=img.reshape(1,imgshape[0]*imgshape[1],3)[0]#
     img_reshape=img_reshape/255
     print(np.shape(img_reshape))
     
     
     u=randominit_u(img_reshape,k)
     last_min_distance=999999
     for _ in range(10):
        u=randominit_u(img_reshape,k)  
        for i in range(10):
            c,min_distance=findclosestcentroids(img_reshape,u)
            u=updatecentroid(img_reshape,c,u)
            if(min_distance<last_min_distance):
                 last_min_distance=min_distance
                 last_min_u=u
         
     print(np.shape(c))
     img_recover=last_min_u[c,:] # 这个有点难以理解呀，但是这么做很神奇啊。
     img_recover=img_recover.reshape(imgshape[0],imgshape[1],3)
     plt.imshow(img_recover)
     
# =============================================================================
#    x=loaddata2()
#    plt.plot(x[:,0],x[:,1],"bo")
#    plt.show()
#    PCA(x)
    
main()
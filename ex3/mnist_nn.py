#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:19:59 2019

@author: zoey.yang
"""

import numpy as np
from matplotlib import pyplot as plt
import sigmoidf as sf
from scipy.io import loadmat


def loaddata():
    trainm=loadmat("./ex3data1.mat")
    datax=trainm["X"]
    labely=trainm["y"]
    return datax,labely

def predict(t1,t2,X):
    m,n=np.shape(X)
    one=np.ones((m,1))
    X_one=np.c_[one,X]
    Z1=np.dot(X_one,t1.T) #[5000.25] A1=X
    A2=sf.sigmoidf(Z1) #[5000,25] 经过激活函数计算后称为A2， 做为下一层的输入
    
    A2_one=np.c_[one,A2]
    Z2=np.dot(A2_one,t2.T) #[5000,10]
    A3=sf.sigmoidf(Z2)
    return A3    

def main():
    m=loadmat("./ex3weights.mat")

    theta1=m["Theta1"]
    theta2=m["Theta2"]
    datax,labely=loaddata()
    m,n=np.shape(datax)
    prep=predict(theta1,theta2,datax)
    
    #matlab index
    labely = labely-1

    presum=0
    for i in range(len(labely)):
        hmax=0
        jmax=0
        for j in range(0,10):          
            if (prep[i][j]>hmax):
                hmax=prep[i][j]
                jmax=j
        presum += int(jmax==labely[i])
        
    print("Accuracy: {acc:.5f}%".format(acc=float(presum/len(labely)*100)))
  
    #        im = datax[i].reshape(20,20)
    #        plt.imshow(im,'gray')
    #        plt.pause(0.0000001)
    #        print("i:",i," jmax:",jmax,hmax,labely[i])
  
    
    
    
   
    
    
main()

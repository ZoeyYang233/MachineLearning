#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:40:01 2019

@author: zoeyyang
"""
import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt
import sigmoidf as sf 
from scipy.optimize import minimize
import ex3load as ld

    

def costfunreg(theta,X,Y,lamda):

    thetaX = np.dot(X, theta)
    m,n = np.shape(X) #python dont give the value whn column no= one

    jcost = 0
    
    h = sf.sigmoidf(thetaX)
   
    for i in range(m):
        jcost += -(Y[i]*np.log(h[i])+(1-Y[i])*np.log(1-h[i]))/m     #+lamda*thetaX[i]/m #sum of y

    thetaall=0
    for j in range(1,n):
        thetaall = thetaall + theta[j]**2
    jcost += lamda*thetaall/(2*m)

    return jcost
    
def bgd(theta,X,Y,lamda):
    m,n = np.shape(X)
    thetaX0 = np.dot(X[:,0],theta[0])
    h0 = sf.sigmoidf(thetaX0)
    #    print("h0",np.shape(h0),type(h0))
    x0 = X[:,0]
    grad0 = np.dot(x0.T,(h0-Y))/m  # x0[118,1] h0[118,1] grad0[1,1]
    #      print("grad0",grad0,type(grad0))
             
    thetaX = np.dot(X,theta.T)
    h = sf.sigmoidf(thetaX)

    x = X[:,1:]
    grad = np.dot(x.T,h-Y) + lamda*theta[1:]/m #grad[n,1], sum the x[j]*cost
    #    print("grad",np.shape(grad),type(grad) )
    gradnew=np.insert(grad,0,[grad0],axis = 0)
    #    print("gradnew",gradnew,type(grad) )
    
    return gradnew



#def main():
#    batch_size=500
#    data,trainlabels=ld.loaddata(batch_size)
#    traindata=np.ones((batch_size,785)) # add bias
#    traindata[:,1:]=data   
#    
#    theta =np.zeros(785) #[0]*n_x  #it's a list, zeros 产生np.array
#    lamda=1
#    """
#    用逻辑回归来进行多分类，就需要建立多个h（j),比如在这里的有10个类，我们先对第一个标签“0”，
#    以及其他非零进行分类，这里获得第一行trainlabels[0] 来得到y(j)
#    """
#    bgfs=[0]*10#这个声明方法错来
#    thetaall=np.zeros((10,785))
#    for j in range(0,9):
#        bgfs[j]=minimize(costfunreg,thetaall[j],(traindata,trainlabels[:,j],lamda),jac=bgd,method='BFGS')# the cost and the bgd should have
#        thetaall[j]=bgfs[j].x                                                      #the same parameter and default as (data_x,data_y)
#    #print(theta)
#    
#    testdata,testlabels=ld.loadtest()
#    m,n=np.shape(testdata)
#    one=np.ones((m,1))
#    testdata1=np.c_[one,testdata]
#    h=[0]*100
#    for i in range(100):
#        hmax=0
#        for j in range(0,10):
#            h[i]=sf.sigmoidf(np.dot(testdata1[i],thetaall[j]))
#            if (h[i]>hmax):
#                hmax=h[i]
#                jmax=j
#        im = testdata[i].reshape(28,28)
#        plt.imshow(im,'gray')
#        plt.pause(0.0000001)
#        print(jmax," ",hmax,testlabels[i])
#    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:52:10 2019

@author: zoeyyang
"""

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat 
from sklearn import svm
import numpy as np

def loaddata():
    m=loadmat("./ex6data3.mat")
    x=m["X"]
    y=m["y"]
    xv=m["Xval"]
    yv=m["yval"]
    
    return x,y,xv,yv

def plotdata(x,y):
    
    inds = [index for index in range(len(y)) if y[index]] #
    negs = list(set(range(len(y))) - set(inds))           # for set could use minus directly
    
    plt.plot(x[inds,0], x[inds,1],'rx', x[negs,0], x[negs,1],'go')
 
def plotcurve(cost,C,sigma):
    plt.plot(cost,C)
    plt.plot(cost,sigma)
    plt.show
    return ;

def main():
    #    x,y=loaddata()
    #    plotdata(x,y )
    #    clf=svm.SVC(kernel="rbf",C=100)
    #    clf.fit(x,y)
    #    clf.score()
    #    
    #    w = clf.coef_[0]
    #    a = -w[0] / w[1]  # a可以理解为斜率
    #    xx = np.linspace(0, 5)
    #    yy = a * xx - clf.intercept_[0] / w[1]  # 二维坐标下的直线方程
    #    h0 = plt.plot(xx, yy, 'k-', label='no weights')
    #    plt.show()
    x,y,xv,yv=loaddata()
    clf1=svm.SVC(C=0.3,kernel="rbf")
    clf2=svm.SVC(C=0.6,kernel="rbf")
    clf1.fit(x,y)
    clf2.fit(x,y)
    result1=clf1.predict(xv)
    result2=clf2.predict(xv) 
    print("result1 ",result1," result2 ",result2)
    
    return;
    
    
    
main()
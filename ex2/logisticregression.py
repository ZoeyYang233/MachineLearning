#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:58:17 2019

@author: zoeyyang
"""

import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt
import sigmoidf as sf 
import plotdata as pld
from scipy.optimize import minimize

def getdata(data):
    x_data=np.ones(np.shape(data))
    x_data[:,:-1]=data[:,:-1]
    y_data=data[:,-1]
    return x_data,y_data


def costfun(theta,X,Y):
    thetaX=np.dot(X,theta.T)  # [m,1]
    print("thetaX",np.shape(thetaX),type(thetaX))
    m=len(Y)#python dont give the value whn column no= one
    h=sf.sigmoidf(thetaX)
    jcost=0
    for i in range(m):
        jcost += -(Y[i]*np.log(h[i])+(1-Y[i])*np.log(1-h[i]))/m  #sum of y
    print("jcost",jcost)    
    return jcost

def bgd(theta,X,Y):
    thetaX=np.dot(X,theta.T)
    m=len(Y)
    h=sf.sigmoidf(thetaX)
    x_trans=X.T
    grad = np.dot(x_trans,(h-Y))/m  #grad[n,1], sum the x[j]*cost
    return grad
    
def getDB(theta,data_x):
    #    x1=(-theta[2]-x0*theta[0])/theta[1]
    #    x0=(-theta[2]-x1*theta[1])/theta[0]
    x1=(-theta[2])/theta[1]
    x0=(-theta[2])/theta[0]
    
    #    plotx=[np.min(data_x[:,0])-2,np.max(data_x[:,0])+2] #-2/+2 is a skill for plot to extend the line
    #    arrayplotx=np.array(plotx)
    #    intplotx=arrayplotx.astype(np.int16)
    #    ploty=(-theta[2]-intplotx*theta[0])/theta[1] # ploty= x1 
    
    return [0,x0],[x1,0]     

    
def main():
    
    data_path=r"./ex2data1.txt"
    data_set=genfromtxt(data_path,delimiter=',')
    m_x,n_x=np.shape(data_set)
    theta=np.zeros(n_x) #initial theta is important too, whether it's one or zero
    #    learnrate=0.00011  # at first when i set th learnrate to 0.001 it goes to fast
    #    iters_num=10000
    data_x,data_y=getdata(data_set)
    #    for i in range (iters_num):
    #        thetaX=np.dot(data_x,theta.T)
    #        grad=bgd(thetaX,data_x,data_y)
    #        jcost = costfun(thetaX,data_y)
    #        jcl += [jcost]
    #        tl0 += [thetaX[0]]
    #        tl1 += [thetaX[1]]
    #        tl2 += [thetaX[2]]
    bgfs=minimize(costfun,theta,(data_x,data_y),jac=bgd,method='BFGS')# the cost and the bgd should have
    theta=bgfs.x                                                      #the same parameter and default as (data_x,data_y)
    
    print(theta)
    xsmp=[45,85,1]
    ypred=sf.sigmoidf(np.dot(data_x,theta.T))
    ysmp=sf.sigmoidf(np.dot(xsmp,theta.T))
    print(ysmp)

    datapred = data_x
    datapred[:,-1]=ypred

    linex,liney=getDB(theta,data_x)
    
    # pld.plotdata(datapred)
    plt.xlim([np.min(data_x[:,0])-2, np.max(data_x[:,0])+2])# change the plot to a better scal 
    plt.ylim([np.min(data_x[:,1])-2, np.max(data_x[:,1])+2])# put the original point better place
    plt.plot([linex[0],linex[1]], [liney[0], liney[1]])     #确定
    pld.plotdata(data_set)
    
main()
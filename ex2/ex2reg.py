#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:05:12 2019

@author: zoeyyang
"""

import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt
import sigmoidf as sf 
import plotdata as pld
from scipy.optimize import minimize
import pandas as pd

def mappingfeature(x1,x2,power):
    #    data={}
    #    for i in np.arange(power + 1):
    #        for p in np.arange(i + 1):
    #            data["f{}{}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    #            n=n+1
    #
    #    data = {"f{}{}".format(i - p, p): np.power(x1, i - p) * np.power(x2, p)
    #                     for i in np.arange(power + 1)
    #                     for p in np.arange(i + 1)
    #           }
    #    print(pd.DataFrame(data))
    #    print(n)
    out = []       #it's a list growing
    for i in range(power+1):  # why +1
        for j in range (i+1):
            data = x1**(i-j) * x2**j
            out += [data]
            
    return np.array(out).T
    #    print(np.shape(data_set))
    #    print(data_set) 
    #pd.DataFrame(data)  #need to understand more!!! of 多项式映射

def costfunreg(theta,X,Y,lamda):

    thetaX = np.dot(X, theta.T)
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

def getDB(theta):
    #decisiongrid=[[0]*210]*210
    list2 = []
    for indi, xi in enumerate(range(-100, 110)):
        val_xi = xi / 100
        
        list1 = [0]*210
        for indj, xj in enumerate(range(-100, 110)):
            val_xj = xj / 100
            superX=mappingfeature(val_xj,val_xi,6)
            
            #decisiongrid[indi][indj] = 1 if (np.dot(superX,theta.T) <= 0) else 0
            list1[indj] = np.dot(theta.T,superX) > 0
            
            
            
            #            print(1 if (np.abs(np.dot(superX,theta.T)) < 0.1))
            #            #break
            #print(val_xi, val_xj, np.dot(superX,theta.T), decisiongrid[indi][indj])
        list2 += [list1]
        
        
    #print(np.max(decisiongrid[:]))
    #plt.imshow(decisiongrid)
    #plt.colorbar()
    
    #print(decisiongrid)

    plt.imshow(list2)
    


def main():
    data_path=r"./ex2data2.txt"
    data_set=genfromtxt(data_path,delimiter=',')
    pld.plotdata(data_set)
    featuredatax=mappingfeature(data_set[:,0],data_set[:,1],6)
    m_x,n_x=np.shape(featuredatax)
    print(m_x,n_x)
    data_y=data_set[:,-1]
    theta =np.zeros(n_x) #[0]*n_x  #it's a list, zeros 产生np.array
    lamda=1
    print(costfunreg(theta,featuredatax,data_y,lamda))
    
    bgd(theta,featuredatax,data_y,lamda)
    bgfs=minimize(costfunreg,theta,(featuredatax,data_y,lamda),jac=bgd,method='BFGS')# the cost and the bgd should have
    theta=bgfs.x                                                      #the same parameter and default as (data_x,data_y)
    
    print("theta")                        
    print(theta) 
    getDB(theta)

    
main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:59:15 2019

@author: zoeyyang
"""

import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat 
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loaddata():
    m=loadmat("./ex5data1.mat")
    xt=m["X"]
    yt=m["y"]
    xv=m["Xval"]
    yv=m["yval"]
    xtest=m["Xtest"]
    ytest=m["ytest"]
    return xt,yt,xv,yv,xtest,ytest

def mappingfeature(x1,power):
    out = []       #it's a list growing
    for i in range(1,power+1):  
            data = np.power(x1,i)
            out += [data]      
    return np.array(out).T[0]    

def hy(t,X):
    m=np.shape(X)[0]
    one=np.ones(m)
    X_one=(np.c_[one,X])
    h=np.dot(X_one,t.T)
    
    return h
    
def costfun(t,X,Y,lamda):
    h=hy(t,X)
    m=np.shape(X)[0]
    jcost=0
    for i in range(m):
        jcost += ((h[i]-Y[i])**2)
        
    jcost=(jcost+ lamda*(t[1]**2))/(2*m)
     
    grad0=0
    grad=0
    for i in range(m):
        grad0 += h[i]-Y[i]
        grad+=(h[i]-Y[i])*X[i]+ lamda*t[1]
        
    grad0=grad0/m
    grad=grad/m
    
    grad=np.insert(grad,0,grad0[0],axis=0)
    
    return jcost, grad
    
def featurenormalize(x):
    
    data_nor=(x-np.mean(x,axis=0))/np.std(x,axis=0)
    return data_nor       


def learningcurve(theta,xnor,y,xvnor,yv,lamda):
    train_err=[]
    trainx_num=[]
    val_err=[]
   
    for i in range(2,len(y),1):
        xsub=xnor[:i,:]
        ysub=y[:i]
        fmin=minimize(fun=costfun,x0=theta,args=(xsub,ysub,lamda),method="TNC",jac=True)
        if(i==2):
            train_theta=fmin.x
        else:
            train_theta=np.vstack((train_theta,fmin.x))
        err=costfun(fmin.x,xsub,ysub,lamda)[0]
        
        trainx_num=np.insert(trainx_num,0,i,axis=0)
        train_err=np.insert(train_err,0,err,axis=0)
       
    m=np.shape(train_theta)[0]
    for i in range(m):
        errval=costfun(train_theta[i,:],xvnor,yv,lamda)[0]
        val_err=np.insert(val_err,0,errval,axis=0)   
        
    #    plt.plot(trainx_num,train_err,"b-")
    #    plt.plot(trainx_num,val_err,"r-")
    #    plt.show()   
    return np.min(val_err),np.min(train_err)
        
        
    
def main():
    x,y,xv,yv,xtest,ytest=loaddata()
    #    plt.plot(x,y,"go")
    m=np.shape(x)[0]
    theta=np.zeros((9))
    lamda=[0,0.01,0.03,0.06,0.09,0.1,0.3,1 ]

    xp=mappingfeature(x,8)
    xnor=featurenormalize(xp)
    xvp=mappingfeature(xv,8)
    xvnor=featurenormalize(xvp)
    

    #    fmin=minimize(fun=costfun,x0=theta,args=(xnor,y,lamda),method="TNC",jac=True)
    #    thetatrain=fmin.x
    #    
    #    xpre=[]
    #    ypre=[]
    #    
    #    for xi in range(-80,80,2):
    #        xpre=np.insert(xpre,0,xi,axis=0)
    #    xpre=np.mat(xpre).T  
    #   
    #    xprep=mappingfeature(xpre,8)
    #    
    #    xprenor=featurenormalize(xprep)
    #    ypre= hy(thetatrain,xprenor)
    #
    #    plt.plot(x,y,"bo")
    #    plt.plot(xpre,ypre,"r-")
    #    plt.show()
    valerr_minset=[]
    trainerr_minset=[]
    lamda_reverse=[]
    for i in range(len(lamda)):
        valerr,trainerr=learningcurve(theta,xnor,y,xvnor,yv,lamda[i])
        valerr_minset=np.insert(valerr_minset,0,valerr,axis=0)
        trainerr_minset=np.insert(trainerr_minset,0,trainerr,axis=0)
        lamda_reverse=np.insert(lamda_reverse,0,lamda[i],axis=0)
    plt.plot(lamda,trainerr_minset,"b-")
    plt.plot(lamda,valerr_minset,"r-")
    plt.show()   
    
    
    
main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:25:01 2019

@author: zoeyyang
"""


import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt

#_,b=read_file('./ex1data1.txt') only get b
#a,_=read_file('./ex1data1.txt') only get a


def getData(data_set):
    m,n=np.shape(data_set)# return data_set r,c
    train_data=np.ones((m,n))# because the need to add bias aka x0=1, so replace x0 as y,
                             #so we create n column
    train_data[:,:-1]=data_set[:,:-1]# :=  train_data[m,n]
    train_result=data_set[:,-1] #[m,1] the last column 
    return train_data,train_result
 

#plot data
#train_data,train_result=getData(data_set)
#plt.plot(train_data,train_result, 'rx')
#plt.show()
def featurenormalization (x_data):
    x_mean=np.mean(x_data,axis=0)
    print(x_mean)
    x_std=np.std(x_data,axis=0,ddof=0)
    print(x_std)
    x_norm= (x_data-x_mean)/x_std
    return x_norm
    
    

def batchGradientDescent(x,y,theta,learnrate,m,num_iters):
    xTrains=x.transpose()  #x=[m,n], xTrains =
    for i in range(num_iters):  
        h=np.dot(x,theta)
        cost=h-y
        #print loss
        gradient=np.dot(xTrains,cost)/m  #
        theta=theta-learnrate*gradient
    return theta


def predict(x,theta):
    m,n=np.shape(x)
    xtest=np.ones((m,n+1))
    xtest[:,:-1]=x
    ypre=np.dot(xtest,theta)
    return ypre


data_path=r"./ex1data2.txt"

data_set=genfromtxt(data_path,delimiter=',')

train_data,train_result=getData(data_set)
train_data_norm=train_data
train_data_norm[:,:-1]=featurenormalization(train_data[:,:-1])


m,n=np.shape(train_data)


theta=np.ones(n) #question here why n?
learnrate=0.01
num_iters=1500
pre_theta=batchGradientDescent(train_data_norm,train_result,theta,learnrate,m,num_iters)


#x =np.array([[11.7],[5.5416],[4],[3],[2],[2.43],[4.2],[3.1],[6.103]])
#pre_y=predict(x,pre_theta)
#print(pre_theta)
#print (pre_y)



#GradientDescent

#

#theta=np.zeros(2,1)
#
#
##define target function
#def f(X):
#    return theta*x+bias
#
#
##define cost function
##def computecost(X,y,theta):
##    return 
#
#
##define gradient Descent 
#
#cost_history=np.zeros(num_iters,1)
#
#for n in range (num_iters):
#    w1=w1 - learnrate*x
#    bias=bias - learnrate*x
    





#plt.xlabel('population',color='green')
#plt.ylabel('profit',color='red')
#

#print(m)
#
#a=np.array([[1,2],[3,4],[5,6]])
#b=np.array([[1,2],[3,4],[5,6]])
#c=np.array([[1,2,3],[4,5,6]])

#print ("hello *,",a*b)
#print ("hello .,",a.dot(c))
#print("hello 5 eye",np.eye(5))
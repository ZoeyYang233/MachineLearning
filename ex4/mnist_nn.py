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
import random
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
import sys

n1=25
m1=401
n2=10
m2=26

def scalelabel(label):
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(label)
    return y_onehot

def reshape(s1,s2):
    
    rs1=s1.reshape(1,n1*m1)
    rs2=s2.reshape(1,n2*m2)
    rs=np.hstack((rs1,rs2))[0].T
    print("rs")
    print(type(rs))
    return rs  

def deshape(rs):
    
    s1=np.reshape(rs[:n1*m1],(m1,n1))    
    s2=np.reshape(rs[n1*m1:],(m2,n2))
    
    return s1, s2

def loaddata():
    trainm=loadmat("./ex4data1.mat")
    datax=trainm["X"]
    labely=trainm["y"]
     
    return datax,labely

def hy(t1,t2,X):
    m,n=np.shape(X)
    one=np.ones((m,1))
    X_one=np.c_[one,X]
    Z2=X_one*t1 #[5000.25] A1=X
    A2=sf.sg(Z2) #[5000,25] 经过激活函数计算后称为A2， 做为下一层的输入
    
    A2_one=np.c_[one,A2]
    Z3=A2_one*t2#[5000,10]
    A3=sf.sg(Z3)
  
    return X_one,Z2,A2_one,A3 

def costfun(t_squeeze,X,Y,lamda):  
    t1, t2 = deshape(t_squeeze)            
    t1=np.matrix(t1)
    t2=np.matrix(t2)
    
    m,n=np.shape(Y)#python dont give the value whn column no= one
    h=hy(t1,t2,X)[3]
    jcost=0
    
    try: 
        for i in range(m):
            term1=np.multiply(-Y[i,:],np.log(h[i,:]))
            term2=np.multiply((1-Y[i,:]),np.log(1-h[i,:]))
            jcost += np.sum(term1-term2) #sum of y
    except:
        if(h[i,0]==float("nan")):
            sys.exit()
    #!! 注意python下标，左闭右开，另外只需要去除bias，所以j并不需要也减掉1！！
    #以下是为了正则化
    
    theta1all = np.sum(np.power(t1[1:,:],2))
    theta2all = np.sum(np.power(t2[1:,:],2))
    
    jcost += float(lamda)*(theta1all+theta2all)/(2*m)

    return jcost

def  initweight():
     epsilon=0.12
     
     t1_rand=np.random.rand(n1,m1)*2*epsilon-epsilon
     t2_rand=np.random.rand(n2,m2)*2*epsilon-epsilon

     return t1_rand,t2_rand
 
def backprop(t_squeeze,X,Y,lamda):
    t1_init, t2_init = deshape(t_squeeze)
    
    t1_init=np.matrix(t1_init)
    t2_init=np.matrix(t2_init)
    m=np.shape(X)[0]
    
    a1_one,z2,a2_one,a3=hy(t1_init,t2_init,X)
   
    
    #    delta3=a3-Y #[5000,10]
    #    
    #    mid=np.dot(delta3,t2_init[:,1:])#在这里就去除了delta[0]
    #    delta2=mid*sf.sgrad(z2)#  update 函数有问题？？？
    #   
    #    
    #    
    #    Delta2=np.dot(delta3.T,a2_one)  #[10,26] 这种用np.dot的方法应该是没有累集
    #    
    #    Delta1=np.dot(delta2.T,a1_one) #[25,401]
    Delta1=np.zeros(np.shape(t1_init))
    Delta2=np.zeros(np.shape(t2_init))
    #    for t in range(m):
    #        delta3t=a3[t,:]-Y[t,:] #[1,10]
    #        delta2t=np.multiply((delta3t*t2_init.T)[:,1:],sf.sgrad(z2[t,:]))#[1,25]
    #        
    #        Delta2=Delta2+ a2_one[t].T*delta3t
    #        Delta1=Delta1+ a1_one[t].T*delta2t
    #       
    delta3t=a3-Y #[1,10]
    delta2t=np.multiply((delta3t*t2_init.T)[:,1:],sf.sgrad(z2))#[1,25]
    
    Delta2= a2_one.T*delta3t
    Delta1= a1_one.T*delta2t
        
    gradt2=Delta2/m 
    gradt2[1:,:]=Delta2[1:,:]/m + lamda*t2_init[1:,:]/m
    
    gradt1=Delta1/m   
    gradt1[1:,:]=Delta1[1:,:]/m + lamda*t1_init[1:,:]/m
 
    grad = np.concatenate((np.ravel(gradt1), np.ravel(gradt2)))#!!!
    
    return grad


#
#def gradcheck(t1,t1,X,Y):
#    
#    
#    epsilon=0.0001
#    m,n=np.shape(Y)
#    for i in range(n):    
#        t1plus=
#        t1minus=
#               
    

def main():
    m=loadmat("./ex4weights.mat")

    #    theta1=m["Theta1"]
    #    theta2=m["Theta2"]
    #    
    #    print(theta1[0])
    
    datax,labely=loaddata()
    m,n=np.shape(datax)
    scaley=scalelabel(labely)
    datax=np.matrix(datax)
    scaley=np.matrix(scaley)
    lamda=1
       
    theta= (np.random.random(size=25* (400 + 1) + 10 * (25 + 1)) - 0.5) * 0.25#随机给的初始参数theta1和theta2，现在为一个行向量，用时再按维数合成矩阵
    backprop(theta,datax,scaley,lamda)  
      
    for i in range(3): 
        bfgs=minimize(costfun,theta,(datax,scaley,lamda),jac=backprop,method='TNC',options={'maxiter': 250})
        theta=bfgs.x    
        print(theta[0])
        
    t1,t2=deshape(theta)
    pred=np.array(np.argmax(hy(t1,t2,datax)[3],axis=1)+1)  #!!!!

    presum=[1 if a==b else 0 for (a,b) in zip(pred,labely)]
   
    accuracy=(sum(map(int,presum))/float(len(presum)))
    print ('accuracy = {0}%'.format(accuracy * 100))
#      
    #        im = datax[i].reshape(20,20)
    #        plt.imshow(im,'gray')
    #        plt.pause(0.0000001)
    #        print("i:",i," jmax:",jmax,hmax,labely[i])
              
        

    
   
    
    
main()

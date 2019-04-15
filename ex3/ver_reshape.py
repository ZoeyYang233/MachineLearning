#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:52:26 2019

@author: zoeyyang
"""

import numpy as np

dim0=10
dim1=11
theta1=np.ones((10,11))

theta2=np.ones((10,11))*2
theta3=np.ones((1,11))*3

theta1=theta1.reshape(1,dim0*dim1)
theta2=theta2.reshape(1,dim0*dim1)
theta3=theta3.reshape(1,dim1)


thetaVec = np.hstack((theta1, theta2, theta3))[0]
#！！"对于python来说，类似这样的取值都是左闭右开的"
theta11=np.reshape(thetaVec[:dim0*dim1],(dim0,dim1))
theta22=np.reshape(thetaVec[dim0*dim1:2*dim0*dim1],(dim0,dim1))
theta33=np.reshape(thetaVec[dim0*dim1*2:2*dim0*dim1+dim1],(1,dim1))

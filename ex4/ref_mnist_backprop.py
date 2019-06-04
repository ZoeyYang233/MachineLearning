#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:13:39 2019

@author: zoeyyang
"""

"""
Created on Tue Nov 21 16:30:57 2017

@author: leisure
"""

# -*- coding: utf-8 -*-

import numpy as np                                  #引用numpy库，用np表示，方便矩阵运算
from scipy.io import loadmat                        #引用loadmat，读取题目给的输入和正确输出的mat文件

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):          #正向传播程序
    m = X.shape[0] #5000组数据
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)#在X最前 面加一列5000*1的向量
    print(type(a1))
    z2 = a1 * theta1                                 #得到5000*25的矩阵，400个特征转化为25个隐藏特征
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) #取激活后的z2，再加一列5000*1的向量
    z3 = a2 * theta2                                #得到5000*10的矩阵，最后10个特征
    h = sigmoid(z3)                                 #激活后得到10个概率
    print(type(h))
    return a1, z2, a2, z3, h

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))  #s函数求导

def backprop(params, input_size, hidden_size, num_labels, X, y,learning_rate): #反向传播算法计算梯度的函数，最后得到误差J和梯度grad（把两个梯度矩阵按行展开合成一行）
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], ((input_size + 1),hidden_size)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], ((hidden_size + 1),num_labels)))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (401, 25)
    delta2 = np.zeros(theta2.shape)  # (26, 10)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:])) #y_onehot的每一行和输出概率h的每一行数据相乘
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term) #乘出来的10*1的向量元素求和，再累加5000组的数据
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[1:,:], 2)) + np.sum(np.power(theta2[1:,:], 2))) #加上正则化的项
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10) 最后的误差向量
        
#        z2t = np.insert(z2t, 0, values=np.ones(1))  # 补1为(1, 26)
        d2t = np.multiply((d3t * theta2.T)[:,1:], sigmoid_gradient(z2t))  # (1, 25)
        
        delta1 = delta1 + a1t.T * d2t  #401*25
        delta2 = delta2 + a2t.T * d3t    #26*10
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # add the gradient regularization term
    delta1[1:,:] = delta1[1:,:] + (theta1[1:,:] * learning_rate) / m
    delta2[1:,:] = delta2[1:,:] + (theta2[1:,:] * learning_rate) / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad

from scipy.optimize import minimize

# initial setup
data = loadmat('ex4data1.mat') #字典形式的数据结构
X = data['X']
y = data['y']

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y) # 将y转为y_onehot，每组数据中概率最大的为1，其余为0，5000*10

input_size = 400 #输入层变量个数400
hidden_size = 25 #隐藏层变量个数为25
num_labels = 10  #输出层10个代表分别为1到10的概率
learning_rate = 1#正则化的系数

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25#随机给的初始参数theta1和theta2，现在为一个行向量，用时再按维数合成矩阵
print(type(params))
m = X.shape[0] #m为5000，组数据
X = np.matrix(X)#转为矩阵方便运算
y = np.matrix(y)

# minimize the objective function
#fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot,learning_rate), #用minimize函数通过梯度grad优化J得到最优的theta1和theta2让J最小
#                method='TNC', jac=True, options={'maxiter': 250})

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], ((input_size + 1),hidden_size ))) #优化得到的行向量按维数合成theta1和theta2
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], ((hidden_size + 1),num_labels )))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)#最优的theta1和theta2正向传播一次，看看预测下效果
y_pred = np.array(np.argmax(h, axis=1) + 1)#因为h索引从0开始，需要加1，得到y_pred为我们的预测结果

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]#比较预测和实际的结果。正确为1，错误为0
accuracy = (sum(map(int, correct)) / float(len(correct)))#累加求和，除以总个数，得到正确率
print ('accuracy = {0}%'.format(accuracy * 100))
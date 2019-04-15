#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:41:49 2019
plot data
@author: zoeyyang
"""

import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt



def plotdata(data_set):
    x = data_set[:,:-1]
    y = data_set[:,-1]
    
    inds = [index for index in range(len(y)) if y[index]] #
    negs = list(set(range(len(y))) - set(inds))           # for set could use minus directly
    
    plt.plot(x[inds,0], x[inds,1],'rx', x[negs,0], x[negs,1],'go')
    plt.show()

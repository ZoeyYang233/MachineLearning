#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:06:55 2019

@author: zoeyyang
"""

import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt

def sigmoidf(z):
    g_one=np.ones(np.shape(z))
    g=g_one/(g_one+np.exp(-z))
    return g
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

def sg(z):
   return 1 / (1 + np.exp(-z))

def sgrad(z):
    return np.multiply(sg(z), (1 - sg(z)))  
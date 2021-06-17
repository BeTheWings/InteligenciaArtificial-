# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:25:20 2021

@author: JeesooPark
"""

import tensorflow as tf

print(tf.__version__)
a=tf.random.uniform([2,3],0.1)
print(a)
print(type(a))
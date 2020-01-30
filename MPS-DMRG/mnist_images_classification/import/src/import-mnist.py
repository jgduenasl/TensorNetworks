#!/usr/bin/dev python3
#
# Author: DP
# Maintainer: DP & JG
# Coypright: 
#===============================


import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from math import pi
from skimage.transform import rescale
import datetime
import opt_einsum as oe

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#export the data into the folder input/

#!/usr/bin/dev python3
#
# Author: DP
# Maintainer: DP & JG
#Copyright:
#==========================


import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from math import pi
from skimage.transform import rescale
import datetime
import opt_einsum as oe

def plot_mnist(X_train, Y_train):
  fig, ax = plt.subplots(6,6, figsize = (12, 12))
  fig.suptitle('Primeras 36 imagenes de MNIST')
  fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
  for x, y in [(i, j) for i in range(6) for j in range(6)]:
      ax[x, y].imshow(X_train[x + y], cmap = 'gray')
      ax[x, y].set_title(Y_train[x + y])
  return ax


mnist_data = pd.read_csv('mnist.csv', sep='|', Low_memory=False)

X_train = X_train[:1000,:,:]/255.0
X_test = X_test[:100,:,:]/255.0
Y_train = Y_train[:1000]
Y_test = Y_test[:100]

#export this plot to otuput/ directory with the name orig_mnist.jpeg
plot_mnist(X_train, Y_train)

#Reduction of images by a factor of 2
l = []
for i in range(len(X_train)):
    l.append(rescale(X_train[i,:,:], 0.5, anti_aliasing=True,multichannel=False))

r = []
for i in range(len(X_test)):
    r.append(rescale(X_test[i,:,:], 0.5, anti_aliasing=True,multichannel=False))

X_train = np.array(l)
X_test = np.array(r)

#export this plot to otuput/ directory with the name reduced_mnist.jpeg
plot_mnist(X_train, Y_train)

#One-hot encoding representation
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#Feature map
dtmp = X_train*(pi/2)
F_train = np.concatenate((np.cos(dtmp.reshape(1,dtmp.shape[1]**2,dtmp.shape[0])),
                          np.sin(dtmp.reshape(1,dtmp.shape[1]**2,dtmp.shape[0]))),axis=0)
#F_train(m,:,n) es un vector de caracteristicas 2-dimensional para el n-esimo pixel y la m-esima imagen
dtmp = X_test*(pi/2)
F_test = np.concatenate((np.cos(dtmp.reshape(1,dtmp.shape[1]**2,dtmp.shape[0])),
                          np.sin(dtmp.reshape(1,dtmp.shape[1]**2,dtmp.shape[0]))),axis=0)

print(F_train.shape)
print(F_test.shape)

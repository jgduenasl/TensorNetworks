#!/usr/bin/dev python3
#
# Author: DP
# Maintainer: DP & JG
# Coypright:
#=============================
#/mnist_images_classification/import/src/mps_ini.py
#

import numpy as np
import numpy.linalg as LA
import opt_einsum as oe

#Random initialization of MPS
np.random.rand(1234)

Nsites = F_train.shape[1]
chid = 2
chi = 5

A = [0 for _ in range(Nsites)] # tensores de la red MPS
A[0] = np.random.rand(1,chid,chi) 
A[Nsites-1] = np.random.rand(chi,chid,1,label) 
for k in range(1,Nsites-1):
    A[k] = np.random.rand(chi,chid,chi)

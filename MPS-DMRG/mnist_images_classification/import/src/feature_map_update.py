#!/usr/bin/dev/ python3
#
# Author: DP
# Maintainer: DP & JG
# Copyrigth:
#=============================
#/mnist_images_classification/import/src/feature_map_update.py
#

import numpy as np

#Updating Feature map in an effective basis
Flr = [None for _ in range(Nsites+2)]

for p in range(Nsites-2):
    T = A[p]
    if np.all(Flr[p] != None):
        T = oe.contract('ri,ijk->rjk',Flr[p], T)
    T = oe.contract('rjk,jr->rk',T,F_train[:,p,:])
    Flr[p+1] = T/(np.amax(np.abs(T),axis=1)[:,None])

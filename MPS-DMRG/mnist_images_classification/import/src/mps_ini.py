#!/usr/bin/dev python3
#
# Author: DP
# Maintainer: DP & JG
# Coypright:
#=============================

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
    
#Left canonical form
for p in range(Nsites-1):
    chil = A[p].shape[0] ; chir = A[p].shape[2];
    Q,R= LA.qr(A[p].reshape(chil*chid,chir));
    A[p] = Q.reshape(chil,chid,Q.shape[1])
    if p < Nsites - 2:
        A[p+1] = oe.contract('ij,jkn->ikn',R/LA.norm(R),A[p+1])
    else:
        A[p+1] = oe.contract('ij,jknl->iknl',R/LA.norm(R),A[p+1])
        
np.allclose(oe.contract('ijk,ijn->kn',A[194],A[194]),np.eye(A[194].shape[2]))

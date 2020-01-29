#!/usr/bin/dev python3
#
# Author: DP
# Maintainer: DP & JG
# Coypright:
#=============================
#mnist_images_classification/import/src/left_canon_form.py
#require mps_ini.py

import numpy as np
import numpy.linalg as LA
import opt_einsum as oe


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

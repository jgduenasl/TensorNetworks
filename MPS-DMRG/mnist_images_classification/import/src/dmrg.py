#!/usr/bin/dev/ python3
#
# Author: DP
# Maintainer: DP & JG
# Copyrigth:
#=============================

import numpy as np

def MPS_update(B,L,M1,M2,R,y,update):
    
    fx = B
    if np.all(L != None):
        fx = oe.contract('ri,ijkml->rjkml', L, fx)
    fx = oe.contract('jr,rjkml->rkml', M1, fx)
    fx = oe.contract('kr,rkml->rml', M2, fx)
    if np.all(R != None):
        fx = oe.contract('rm,rml->rl', R, fx)
    else:
        fx = np.squeeze(fx,axis=1)
        
    # Medir tasa de error de la funcion de decision
    
    maxid = np.argmax(fx, axis=1)
    
    ERR = 1.0 - np.sum(y[[i for i in range(len(y))],[maxid]])/len(y)
    
    # Desviacion de la funcion de decision del valor correcto
    
    ydiff = y - fx
    
    # Calculo de la funcion de decision por conjunto de datos
    
    CFUN = np.sum(np.abs(ydiff**2))/(2*len(y))
    
    # Calculo de dB 
    
    if update:
        
        dB = ydiff
        
        if np.all(L != None ):
            dB = oe.contract('ri,rl->ril', L, dB)
        else:
            dB = dB[:,None,:]
        dB = oe.contract('jr,ril->rijl', M1, dB)
        dB = oe.contract('kr,rijl->rijkl', M2, dB)
        if np.all(R != None ):
            dB = oe.contract('rm,rijkl->rijkml', R, dB)
        else:
            dB = dB[...,None,:]
        
        dB = np.mean(dB,axis=0)
        
        return ERR, CFUN, dB
    return ERR, CFUN
    

#Updating Feature map in an effective basis
Flr = [None for _ in range(Nsites+2)]

for p in range(Nsites-2):
    T = A[p]
    if np.all(Flr[p] != None):
        T = oe.contract('ri,ijk->rjk',Flr[p], T)
    T = oe.contract('rjk,jr->rk',T,F_train[:,p,:])
    Flr[p+1] = T/(np.amax(np.abs(T),axis=1)[:,None])

#DMRG Right-left running
# Izq <-- Der
err = []; cfun = []; 

for itn in range(Nsites-1,0,-1):
    
    # Tensor de enlace
    
    B = oe.contract('ijk,knml->ijnml',A[itn-1],A[itn])
    
    # calcular la función de costo, tasa de error y gradiente de B
    er, cf, db = MPS_update(B,Flr[itn-1],F_train[:,itn-1,:],
                            F_train[:,itn,:],Flr[itn+2],
                            Y_train,True)

    # actualizar el tensor B
    B = B + db*0.001

    # Hacer SVD para actualizar A[itn-1] y A[itn]
    # Asociar el indice de etiquetas a A[itn-1] 
    
    chil = A[itn-1].shape[0];  chir = A[itn].shape[2];

    A[itn-1], S, A[itn] = LA.svd(B.reshape(chil*chid*label,chid*chir),full_matrices=False)

    chitemp = min(len(S),chi);

    A[itn] = A[itn][:chitemp,:].reshape(chitemp,chid,chir)
    S = S[:chitemp]/LA.norm(S[:chitemp])
    
    # orden de indices: left-physical-right-label
    A[itn-1] = (A[itn-1][:,:chitemp] @ np.diag(S)).reshape(chil,chid,chitemp,label)
                
    # actualizar Flr_train[itn+1] de acuerdo al nuevo A[int]
        
    T = A[itn].transpose(2,1,0);
    if np.all(Flr[itn+2] != None):
        T = oe.contract('ri,ijk->rjk', Flr[itn+2], T)
    T = oe.contract('rjk,jr->rk', T, F_train[:,itn,:])
    Flr[itn+1] = T/(np.amax(np.abs(T),axis=1)[:,None])
    
print(f"{er:.4f}\t",f"{cf:.4f}\t")

#what is the result of this script?

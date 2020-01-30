#!/usr/bin/env python3
#
# Author: DP 
# Mainteiner: DP & JG
# Copyright: PhysMLGr, 2020, GLP-3 or later
#===========================================

import numpy as np
import tensornetwork
import tensorflow as tf
net = tensornetwork.TensorNetwork() # Implementacion de una red tensorial


#Inicializar algunos tensores con numpy y tensornetwork

#Crear un tensor con entradas generadas aleatoriamente, de orden 3, dimensiones: 2,3,4

A = np.random.rand(2,3,4)
a = net.add_node(A)

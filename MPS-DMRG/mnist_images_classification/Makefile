  
# -*- coding: utf-8 -*-
#
# Author: JG
# Maintainer(s): JG
# License: (c) PhysMLGr 2020, GPL v2 or newer
#
# /mnist_images_classification/Makefile
# -----------------------------------------------------------
# This makefile implement the job of classification of mnist images
# There is no need to include the tasks related to tensor algebra

.PHONY: all clean

all: input/mnist.csv.gz 

clean:
    rm output/*
 
#import mnist dataset from a python script
input/mnist.csv.gz: \       # target object file
		import/import-mnist.py \ # requirements
	python3 $< \              # recipes: $< is equivalent to input/import-mnist.py
		--output=$@             # $@ is equivalent to input/mnist.csv.gz
    
#review the outputs of each task and split
mps_initialization:
    python3 import/mps_ini.py import/left_canon_form.py import/feature_map_update.py import/dmrg.py
 

# -*- coding: utf-8 -*-
#
# Author: JG
# Maintainer(s): JG
# License: (c) PhysMLGr 2020, GPL v2 or newer
#
# /import/Makefile
# -----------------------------------------------------------
# This makefile implement the job of classification of mnist images
# There is no need to include the tasks related to tensor algebra

.PHONY: all clean

all: input/mnist.csv.gz output/orig_minst.jpeg output/reduced_minst.jpeg

clean:
    rm output/*
 
#import mnist dataset from a python script
input/mnist.csv.gz: \       		# target object file
		input/import-mnist.py \ # requirements
	python3 $< \              	# recipes: $< is equivalent to input/import-mnist.py
		--output=$@      	# $@ is equivalent to input/mnist.csv.gz
		
output/orig_minst.jpeg: \
		src/clean.py \
		input/mnist.csv.gz 
	python3 $< \
		--output=$@

#how to generate two object files from the same source file?
output/reduced_minst.jpeg: \
		src/clean.py \
		input/mnist.csv.gz 
	python3 $< \
		--output=$@
		
#generate MPS state with Feature map
	#what is the result of this task?
	
#Implement DMRG
	#what is the result of this task?

 

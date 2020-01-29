# -*- coding: utf-8 -*-
#
# Author: JG
# Maintainer(s): JG
# License: (c) PhysMLGr 2020, GPL v2 or newer
#
# /import/Makefile
# -----------------------------------------------------------
#

.PHONY: all clean

all: 

clean:
    rm output/*
    
input/mnist.csv.gz: \       # target object file
		input/import-mnist.py \ # requirements
	python3 $< \              # recipes: $< is equivalent to input/import-mnist.py
		--output=$@             # $@ is equivalent to input/mnist.csv.gz
 
 
 

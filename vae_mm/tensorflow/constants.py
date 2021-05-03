#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""
import numpy as np
# change the current directory to specified directory
from pathlib import Path
import os
p = Path(__file__).parents[2]
os.chdir(p)
import design_evaluator.python as truss_model

E = 0.7e09; # Young's Modulus for polymeric material (example: 10000 Pa)
sel = 0.01; # Unit square side length (NOT individual truss length) (example: 5 cm)
r = 5e-4 #element radius in m
edges = np.array([[1,2], [1,6], [1,5], [1,4], [1,8], [2,3], [2,6], [2,5], [2,4], [2,7],
                   [2,9], [3,6], [3,5], [3,4], [3,8], [5,6], [6,7], [6,8], [6,9], [4,5], 
                   [5,7], [5,8], [5,9], [4,7], [4,8], [4,9], [7,8], [8,9]])-1;
edges_old = np.array([[1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [2,3], [2,4],
                       [2,5], [2,6], [2,7], [2,8], [2,9], [3,4], [3,5], [3,6], [3,7], [3,8], 
                       [3,9], [4,5], [4,6], [4,7], [4,8], [4,9], [5,6], [5,7], [5,8], [5,9],
                       [6,7], [6,8], [6,9], [7,8], [7,9], [8,9]])-1;
nucFac = 1;
sidenum = (2*nucFac) + 1; 
nodes = truss_model.generateNC(sel, sidenum);
target_c_ratio = 1; # Ratio of C22/C11 to target

#! /usr/bin/env python
# -*- coding:utf-8 -*-

#------------------------------------------------------#
# Instructions: Solving high dimensional cross product
# University:   Sichuan University
# Author:       rhythmli
# Data:         2022.10.15
#------------------------------------------------------#

import numpy as np
from scipy import linalg
from itertools import combinations

def high_dimension_cross(W):
    p = W.shape[0]
    V = np.zeros((p, p))
    comb = list(combinations(range(p), p-1))

    A, b = np.array([]), np.array([])
    for i in range(p):
        for j in range(p-1):
            A = np.append(A, np.array([W[comb[i][j]][0:p-1]]))
            b = np.append(b, np.array([W[comb[i][j]][p-1]]))
        
        A = A.reshape(p-1, p-1)
        b = b.reshape(p-1, 1)
        
        if(np.linalg.matrix_rank(A) != p-1):
            V[i] = np.random.rand(p,)
        else:
            V[i] = np.append(linalg.solve(A, b), -1)
        A, b = np.array([]), np.array([])

    return V
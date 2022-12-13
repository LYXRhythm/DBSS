#! /usr/bin/env python
# -*- coding:utf-8 -*-

#------------------------------------------------------#
# Instructions: Mapping function
# University:   Sichuan University
# Author:       rhythmli
# Data:         2022.8.15
#------------------------------------------------------#
import numpy as np

class Kernel():
    def __init__(self, params=['linear', 0, 0]) -> None:
        self.kernel_params = params

    def kernel_trans(self, X, W):
        if self.kernel_params[0]=="linear": 
            K = np.dot(X, W.T)
        elif self.kernel_params[0]=="fourier":
            K = self.kernel_params[1]*np.sin(self.kernel_params[2]*np.dot(X, W.T))
        else:
            raise NameError('The Kernel is not recognized')
        return K
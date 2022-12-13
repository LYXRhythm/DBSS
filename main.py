#! /usr/bin/env python
# -*- coding:utf-8 -*-

#------------------------------------------------------#
# Instructions: Main
# University:   Sichuan University
# Author:       rhythmli
# Data:         2022.11.10
#------------------------------------------------------#

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import linalg
from scipy.signal import stft, istft

from high_dimension_cross import high_dimension_cross
from itertools import permutations
from kernel import Kernel

import sys
#sys.path.append("../")
from utils import ASIR, wave_setting

## commucination
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)
def snr_out(signal_source, signal_source_noise):
    signal_noise = signal_source - signal_source_noise
    mean_signal_source = np.mean(signal_source)
    signal_source = signal_source - mean_signal_source
    snr = 10 * math.log(np.sum(signal_source**2) / np.sum(signal_noise**2),10)
    return snr

## algorithm
def weight_iterator(X, XA, XB, xi, kernel):
    num_XA = XA.shape[0]
    num_XB = XB.shape[0]
    e1 = np.real(np.ones((num_XA, 1)))
    e2 = np.real(np.ones((num_XB, 1)))
    con = num_XA - xi*num_XB

    if con != 0:
        tmp = np.dot(e1.T, XA) - xi * np.dot(e2.T, XB)
        Q = np.dot(XA.T, XA) - xi * np.dot(XB.T, XB) - 1/con*np.dot(tmp.T, tmp)
        d, v = np.linalg.eig(Q)
        sign = np.argmin(d)
        Wi = v[:, sign]
        bi = -1/con*np.dot(tmp, Wi)
        bi = bi[0]
    else:
        Q = np.dot(XA.T, XA) - xi*np.dot(XB.T, XB)
        d, v = np.linalg.eig(Q)
        sign = np.argmin(d)
        Wi = v[:, sign]
        bi = 1 / num_XA * np.dot(np.sum(XA, axis=0), Wi)
    return Wi, bi

def Matrix_sub_rank(A):
    A_new = A.T

    delete_index = []
    for i in range(A_new.shape[0]):
        for j in range(i+1, A_new.shape[0]):
            error = np.sum((np.ones(len(A_new[i]))-A_new[i]/A_new[j])*(np.ones(len(A_new[i]))-A_new[i]/A_new[j]))
            if error < 0.01:
                delete_index.append(j)
        
    delete_index = list(set(delete_index))

    if len(delete_index)==0:
        return A_new.T

    A_new = np.delete(A_new, delete_index, axis=0)

    return A_new.T

def estimate_s_hat(W, x):
    p = x.shape[0]    # W.shape[0]
    N = x.shape[1]    

    # Calculate A_Matrix_hat
    if(p>=2):
        V_temp = high_dimension_cross(W)

        A_Matrix_hat_temp = np.zeros((np.math.factorial(p), p, p))
        perm = list(permutations(range(p)))
        temp = []
        for i in range(len(perm)):
            for j in range(p):
                temp.append(V_temp[perm[i][j]])
            A_Matrix_hat_temp[i] = np.array(temp).T
            temp = []
    else:
        raise ValueError("Something may error in the observed signal!")

    A_Matrix_hat = []
    for i in range(A_Matrix_hat_temp.shape[0]):
        A_Matrix_hat.append(Matrix_sub_rank(A_Matrix_hat_temp[i]))
    
    try:
        A_Matrix_hat = np.array(A_Matrix_hat)
    except ValueError: 
        A_Matrix_hat = A_Matrix_hat_temp

    # Calculate s_hat in well-posed case
    if A_Matrix_hat[0].shape[0] == A_Matrix_hat[0].shape[1]:
        B_Matrix = np.zeros((np.math.factorial(p), p, p))
        s_hat_Matrix = np.zeros((np.math.factorial(p), p, N))
        s_hat_energy = []
        for i in range(np.math.factorial(p)):
            if np.linalg.matrix_rank(A_Matrix_hat[i]) != p:
                A_Matrix_hat[i] = np.random.rand(p, p)
                B_Matrix[i] = np.linalg.inv(A_Matrix_hat[i])
            else:
                B_Matrix[i] = np.linalg.inv(A_Matrix_hat[i])

            s_hat_Matrix[i] = B_Matrix[i].dot(x)
            s_hat_energy.append(sum(sum(s_hat_Matrix[i])))         
        
        s_hat = s_hat_Matrix[np.argmin(s_hat_energy)]
        return s_hat

    # Calculate s_hat in overcondition case
    else:
        B_Matrix = np.zeros((np.math.factorial(p), A_Matrix_hat.shape[2], A_Matrix_hat.shape[1]))
        s_hat_Matrix = np.zeros((np.math.factorial(p), A_Matrix_hat.shape[2], N))
        s_hat_energy = []
        for i in range(np.math.factorial(p)):
            B_Matrix[i] = np.linalg.pinv(A_Matrix_hat[i])

            s_hat_Matrix[i] = B_Matrix[i].dot(x)
            s_hat_energy.append(sum(sum(s_hat_Matrix[i])))         
        
        s_hat = s_hat_Matrix[np.argmin(s_hat_energy)]
        return s_hat

def algorithm(x, n_components=3, xi=0.000000001, iterator=2000, kernel_params=('fourier', 0.3, 2)):
    # insert function
    def compare(vec1, vec2):
        bool_com_list = np.zeros(len(vec1)) - 1
        for i in range(len(vec1)):
            if vec1[i] == vec2[i]:
                bool_com_list[i] = 0
            else:
                bool_com_list[i] = 1
        index = np.where(bool_com_list == -1)[0]
        return not index.size > 0

    # Hyperparameters setting
    X = x.T
    N = X.shape[0]
    p = n_components

    # Kernel function
    kernel = Kernel(params=kernel_params)

    # Init parameters
    pY = np.random.randint(0, p, (N))
    pY_next = np.zeros(N) - 1
    W, b = np.random.rand(p, p), np.random.rand(p, 1)
    count = 1

    # Iterator
    while(compare(pY, pY_next) and count<iterator):
        # print("count=", int(count))
        pY_next = pY
        count = count + 1
        for i in range(p):
            index_A = np.where(pY==i)[0]
            index_B = np.where(pY!=i)[0]
            XA = np.array([X[ii,:] for ii in index_A])   
            XB = np.array([X[ii,:] for ii in index_B])
            if np.where(pY==i)[0].shape[0] != 0: 
                W[i], b[i] = weight_iterator(np.real(X), np.real(XA), np.real(XB), xi, kernel)
                
        pY_matrix = np.abs(kernel.kernel_trans(X, W) + np.dot(np.ones((N, 1)), b.reshape(1, p)))
        pY = np.argmin(pY_matrix, axis=1).T

    # Estimate s_hat
    s_hat = estimate_s_hat(W, x)    
    
    return s_hat

## display
def display(s, x, s_hat, s_length=2000, x_length=2000, s_hat_length=2000):
    s_t = np.arange(s_length)
    x_t = np.arange(x_length)
    s_hat_t = np.arange(s_hat_length)
    s_num = s.shape[0]
    x_num = x.shape[0]

    line_1_pic = []
    for ii in range(0, s_num):
        line_1_pic.append(plt.subplot(3, x_num, 0*x_num+(ii+1)))
        line_1_pic[ii].plot(s_t, s[ii], c='black')

    line_2_pic = []
    for ii in range(0, x_num):
        line_2_pic.append(plt.subplot(3, x_num, 1*x_num+(ii+1)))
        line_2_pic[ii].plot(x_t, x[ii], c='black')

    line_3_pic = []
    for ii in range(0, s_num):
        line_3_pic.append(plt.subplot(3, x_num, 2*x_num+(ii+1)))
        line_3_pic[ii].plot(s_hat_t, s_hat[ii], c='black')

    plt.savefig("./result/result.png")
    plt.show()

if __name__=='__main__':
    snr = 30.0
    with open(r'./result/result.txt', 'w') as f:
        # load original signal
        s1 = scio.loadmat("./dataset/signal/signal1.mat")['temp'][0]
        s2 = scio.loadmat("./dataset/signal/signal2.mat")['temp'][0]
        s3 = scio.loadmat("./dataset/signal/signal3.mat")['temp'][0]

        s_time_domain = np.array([s1, s2, s3])
        # stft
        fs = 500               # sampling frequency 200
        window = 'hann'        # window function
        n = 512                # frame length 256
        f1, t1, Z1 = stft(s_time_domain[0], fs=fs, window=window, nperseg=n)
        s1_stft = np.real(Z1.reshape(-1,))
        f2, t2, Z2 = stft(s_time_domain[1], fs=fs, window=window, nperseg=n)
        s2_stft = np.real(Z2.reshape(-1,))
        f3, t3, Z3 = stft(s_time_domain[2], fs=fs, window=window, nperseg=n)
        s3_stft = np.real(Z3.reshape(-1,))
        s = np.array([s1_stft, s2_stft, s3_stft])

        # generate observations
        a = [[2, 1, 1], [2, 3, 1], [1, 2, -1]]
        x = np.dot(a, s)
            
        # add noise
        xn = np.zeros((x.shape[0], x.shape[1]))
        for i in range(len(x)):
            n = wgn(x[i], snr)
            xn[i] = x[i]+n                      # add snr-dB noise 
            
        # DBSS solving
        s_hat = xn
        result = float('-inf')
        for i in range(1):
            s_hat_temp = algorithm(xn, n_components=3, xi=1e-7, iterator=2000, kernel_params=('linear', 0, 0))
            s_hat_temp, judge = wave_setting(s, s_hat_temp) 
            if judge:          
                result_temp = ASIR(np.array(s), np.array(s_hat_temp))
            else:
                result_temp = 0
            if(result_temp>result):
                s_hat = s_hat_temp
                result = result_temp
                
        # istst
        s_time_domain = []
        for i in range(len(s)):
            _, s_time_domain_temp = istft(s, fs)
            s_time_domain.append(s_time_domain_temp)
        s_time_domain = np.array(s_time_domain)

        s_hat_time_domain = []
        for i in range(len(s_hat)):
            _, s_hat_time_domain_temp = istft(s_hat, fs)
            s_hat_time_domain.append(s_hat_time_domain_temp)
        s_hat_time_domain = np.array(s_hat_time_domain)

        x_time_domain = []
        for i in range(len(x)):
            _, x_time_domain_temp = istft(x, fs)
            x_time_domain.append(x_time_domain_temp)
        x_time_domain = np.array(x_time_domain)

        # show and save result
        print(np.array(s_time_domain).shape)
        print(np.array(x_time_domain).shape)
        print(np.array(s_hat_time_domain).shape)
        display(s_time_domain, x_time_domain, s_hat_time_domain, \
                    s_length=12848, x_length=12848, s_hat_length=12848)
            
        f.write("SNR="+str(snr)+"  ASIR="+str(result))
        print("SNR=", snr, "  ASIR=", result)

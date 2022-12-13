#! /usr/bin/env python
# -*- coding:utf-8 -*-

#------------------------------------------------------#
# Instructions: Additional functions
# University:   Sichuan University
# Author:       rhythmli
# Data:         2022.8.15
#------------------------------------------------------#

import math
import numpy as np

class Signal():
    def __init__(self, type, period=1, phase=0, up_peak=1, bottom_peak=0) -> None:
        self.type_list = ["sine", "cosine", "triangular", "square"]
        if type in self.type_list:
            self.type       = type
        else:
            self.type       = None
            raise ValueError("signal type is not support...")
        self.period     = period
        self.phase      = phase
        self.up_peak    = up_peak
        self.bottom_peak= bottom_peak
    
    def get_type(self):
        return self.type 

    def get_period(self):
        return self.period 

    def get_up_peak(self):
        return self.up_peak 

    def get_bottom_peak(self):
        return self.bottom_peak 

    def get_signal(self, length):
        t = np.arange(length)
        if self.type == "sine":
            signal = self.up_peak * np.sin(2 * np.pi * t / self.period + self.phase) 
        elif self.type == "cosine":
            signal = self.up_peak * np.cos(2 * np.pi * t / self.period + self.phase) 
        elif self.type == "triangular":
            if length % self.period == 0:
                signal = np.array(np.linspace(self.bottom_peak, self.up_peak, self.period).tolist() * int(length/self.period)).reshape(length, ) 
            else:
                raise ValueError("length cannot be divided by period...")
        elif self.type == "square":
            if length % self.period == 0:
                signal = np.array(int(length/self.period) * (int(self.period/2) * [self.up_peak] + int(self.period/2) * [self.bottom_peak])) 
            else:
                raise ValueError("length cannot be divided by period...")
        
        return signal

def SNR_singlech(S, SN):
    S = S-np.mean(S)
    S = S/np.max(np.abs(S))
    mean_S = (np.sum(S))/(len(S))
    PS = np.sum((S-mean_S)*(S-mean_S))
    PN = np.sum((S-SN)*(S-SN))
    snr=10*math.log((PS/PN), 10)
    return snr

def ASIR(S, S_hat):
    sum = 0
    signal_num = S.shape[0]
    for i in range(signal_num):
        sum = sum + 10 * np.log10(np.sum(S[i]**2)/np.sum((S_hat[i]-S[i])**2))
    return sum / signal_num

def wave_setting(S, S_hat):
    if S.shape[0] != S_hat.shape[0]:
        return S_hat, False

    signal_num = S.shape[0]
    for i in range(signal_num):
        S[i] = (S[i]-np.min(S[i]))/(np.max(S[i])-np.min(S[i]))
        S_hat[i] = (S_hat[i]-np.min(S_hat[i]))/(np.max(S_hat[i])-np.min(S_hat[i]))

    S_hat_negative = -S_hat + 1
    S_hat_all = np.concatenate([S_hat, S_hat_negative], axis=0)
    S_new_hat = []
    s_loss = []
    for i in range(signal_num):
        for j in range(2*signal_num):
            error = np.sum(np.abs(S[i]**2-S_hat_all[j]**2))
            s_loss.append(error)
        S_new_hat.append(S_hat_all[s_loss.index(min(s_loss))])
        s_loss.clear()

    return S_new_hat, True
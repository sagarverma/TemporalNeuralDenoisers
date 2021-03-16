# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:27:04 2018

@author: Yilin Liu

Paper: Condat, L.: A Direct Algorithm for 1-D Total Variation Denoising.
       IEEE Signal Processing Letters 20(11),1054–1057(2013).
       doi: 10.1109/lsp.2013.2278339

Algorithm for arg_min_X 0.5|Y - X|_2^2 + lamda*|X|_TV (|X|_TV = |DX|_1)
"""
import numpy as np

def denoising_1D_TV(Y, lamda):
    
    N = len(Y)
    X = np.zeros(N)

    k, k0, kz, kf = 0, 0, 0, 0
    vmin = Y[0] - lamda
    vmax = Y[0] + lamda
    umin = lamda
    umax = -lamda

    while k < N:
        
        if k == N - 1:
            X[k] = vmin + umin
            break
        
        if Y[k + 1] < vmin - lamda - umin:
            for i in range(k0, kf + 1):
                X[i] = vmin
            k, k0, kz, kf = kf + 1, kf + 1, kf + 1, kf + 1
            vmin = Y[k]
            vmax = Y[k] + 2 * lamda
            umin = lamda
            umax = -lamda
            
        elif Y[k + 1] > vmax + lamda - umax:
            for i in range(k0, kz + 1):
                X[i] = vmax
            k, k0, kz, kf = kz + 1, kz + 1, kz + 1, kz + 1
            vmin = Y[k] - 2 * lamda
            vmax = Y[k]
            umin = lamda
            umax = -lamda
            
        else:
            k += 1
            umin = umin + Y[k] - vmin
            umax = umax + Y[k] - vmax
            if umin >= lamda:
                vmin = vmin + (umin - lamda) * 1.0 / (k - k0 + 1)
                umin = lamda
                kf = k
            if umax <= -lamda:
                vmax = vmax + (umax + lamda) * 1.0 / (k - k0 + 1)
                umax = -lamda
                kz = k
                
        if k == N - 1:
            if umin < 0:
                for i in range(k0, kf + 1):
                    X[i] = vmin
                k, k0, kf = kf + 1, kf + 1, kf + 1
                vmin = Y[k]
                umin = lamda
                umax = Y[k] + lamda - vmax
                
            elif umax > 0:
                for i in range(k0, kz + 1):
                    X[i] = vmax
                k, k0, kz = kz + 1, kz + 1, kz + 1
                vmax = Y[k]
                umax = -lamda
                umin = Y[k] - lamda - vmin
                
            else:
                for i in range(k0, N):
                    X[i] = vmin + umin * 1.0 / (k - k0 + 1)
                break

    return X
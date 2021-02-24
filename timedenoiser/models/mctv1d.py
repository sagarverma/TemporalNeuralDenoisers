# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:26:18 2018

@author: Yilin Liu

Paper: Du, H. & Liu, Y.: Minmax-concave Total Variation Denoising. 
       Signal, Image and Video Processing (2018).
       doi: 10.1007/s11760-018-1248-2
       
Algorithm for arg_min_X 0.5|Y - X|_2^2 + lamda*|X|_MCTV
"""

import tv1d
import numpy as np

def denoising_1D_MCTV(Y, para):
    
    K, N = 0, len(Y)
    X = np.zeros(N)
    U = np.ones(N)
    lamda, alpha = para.regularization, para.nonconvexity
    num, err = para.most_iter_num, para.convergence
    
    while K <= num and np.linalg.norm(U - X) > err:
        
        C = Dxt(Dx(X)) - Dxt(shrink(Dx(X), 1 / alpha))
        Z = Y + lamda * alpha * C
        U = X
        X = tv1d.denoising_1D_TV(Z, lamda)
        K += 1
        
    return X

def shrink(Y, lamda):
    return np.fmax(np.fabs(Y) - lamda, 0) * np.sign(Y)

def Dx(Y):
    return np.ediff1d(Y, to_begin = Y[0] - Y[-1])

def Dxt(Y):
    X = np.ediff1d(Y[::-1])[::-1]
    return np.append(X, Y[-1] - Y[0])
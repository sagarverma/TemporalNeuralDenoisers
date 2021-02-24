# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:46:47 2018

@author: Yilin Liu

Paper: Selesnick, I.: Total Variation Denoising via the Moreau Envelope.
       IEEE Signal Processing Letters 24(2),216–220(2017).
       doi: 10.1109/lsp.2017.2647948
       
Algorithm for arg_min_X 0.5|Y - X|_2^2 + lamda*|X|_METV
"""

import tv1d
import numpy as np

def denoising_1D_METV(Y, para):
    
    K, N = 0, len(Y)
    X = np.zeros(N)
    U = np.ones(N)
    lamda, alpha = para.regularization, para.nonconvexity
    num, err = para.most_iter_num, para.convergence
    
    while K <= num and np.linalg.norm(U - X) > err:
        
        Z = Y + lamda * alpha * (X - tv1d.denoising_1D_TV(Y, 1 / alpha))
        U = X
        X = tv1d.denoising_1D_TV(Z, lamda)
        K += 1
        
    return X
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:40:02 2018

@author: Yilin Liu
"""

import tv1d, metv1d, mctv1d
import parameter as pa

import numpy as np
import scipy.io as sio
import random as rd
import matplotlib.pyplot as plt

def rmse(X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))

def sub_plot(sig, noi_sig, num):
    plt.subplot(2, 2, num)
    plt.plot(sig, linewidth = 0.5)
    plt.plot(noi_sig, color = 'red', linewidth = 0.5)
    plt.xlim(0, 255)
    plt.ylim(-1, 7)
    plt.text(110, 6, 'RMSE = %.4f'%(rmse_sums[num - 1]/experiment_time))

# load demo piecewise constant signal
load_file_name = 'signal'
signal = sio.loadmat(load_file_name)
sig = signal['sig'][0]

rmse_sums, N = [0] * 4, len(sig)
# parameters for Additive White Gaussian Noise(AWGN)
mu, sigma = 0, 0.5 
# calculating multiple times for average Root Mean Square Error(RMSE)
experiment_time = 100

for _ in range(experiment_time):
    
    noi_sig = sig + np.array([rd.gauss(mu, sigma) for _ in range(N)])
    rmse_sums[0] += rmse(noi_sig, sig)
    '''
    lamda is the regularization parameter in denoising models.
    For METV and MCTV,
    the value of lamda is usually between
    sqrt(sigma * N) / 10 ~ sqrt(sigma * N) / 4;
    for TV,
    the value of lamda is usually half of the above or less.
    K, err are both for the convergence condition in METV and MCTV.
    K is the maximum number of iterations;
    err is the error (measured by Euclidean distance)
    between results of two adjacent iterations.
    alpha is the nonconvexity parameter in METV and MCTV,
    which is usually chosen between 0.2 / lamda ~ 0.7 / lamda
    '''
    lamda = np.sqrt(sigma * N) / 5
    K = 100
    err = 0.001
    alpha = 0.3 / lamda
    para = pa.Parameter(lamda, K, err, alpha)
    
    X1 = tv1d.denoising_1D_TV(noi_sig, lamda / 2)
    rmse_sums[1] += rmse(X1, sig)
    
    X2 = metv1d.denoising_1D_METV(noi_sig, para)
    rmse_sums[2] += rmse(X2, sig)
    
    X3 = mctv1d.denoising_1D_MCTV(noi_sig, para)
    rmse_sums[3] += rmse(X3, sig)

# plot demo denoising results
plt.figure()
plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9,
                        wspace = 0.4, hspace = 0.6)

sub_plot(sig, noi_sig, 1)
plt.xlabel('(a)')
plt.title('Original & Noisy Signal')

sub_plot(sig, X1, 2)
plt.xlabel('(b)')
plt.title('TV')

sub_plot(sig, X2, 3)
plt.xlabel('(c)')
plt.title('METV')

sub_plot(sig, X3, 4)
plt.xlabel('(d)')
plt.title('MCTV')

if __name__ == '__main__':
    plt.show()
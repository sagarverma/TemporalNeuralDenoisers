# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:08:07 2018

@author: Yilin Liu
"""

class Parameter(object):
    
    def __init__(self, re, it, co, no):
        self.regularization = re
        self.most_iter_num = it
        self.convergence = co
        self.nonconvexity = no
    
    def print_value(self):
        print(self.regularization, self.most_iter_num,
                  self.convergence, self.nonconvexity)
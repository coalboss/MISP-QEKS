import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import sys

class module_feaFmask(nn.Module):
    def __init__(self, time_dim=2, time_prob=0.5, time_times=2, time_upth=32, freq_dim=3, freq_prob=0.5, freq_times=2, seed=27863875):
        super(module_feaFmask, self).__init__()
        self.rng = np.random.default_rng()
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        
        self.time_dim   = time_dim
        self.time_prob  = time_prob
        self.time_times = time_times
        self.time_upth  = time_upth

        self.freq_dim   = freq_dim
        self.freq_prob  = freq_prob
        self.freq_times = freq_times

    def forward(self, x):
        time_filter = x.shape[self.time_dim]
        if self.time_prob > 0.0 and self.rng.random(1)[0] < self.time_prob:
            for i in range(self.time_times):
                t_gap = self.rng.integers(0, self.time_upth)
                t_bgn = self.rng.integers(0, time_filter-t_gap)
                if self.time_dim == 3:
                    x[:, :, :, t_bgn:t_bgn+t_gap] = 0
                elif self.time_dim == 2:
                    x[:, :, t_bgn:t_bgn+t_gap, :] = 0
                else:
                    print("Error: time_dim only 2 or 3...")
                    sys.exit()

        freq_filter=x.shape[self.freq_dim]
        if self.freq_prob > 0.0 and self.rng.random(1)[0] < self.freq_prob:
            for i in range(self.freq_times):
                f_gap = self.rng.integers(0, freq_filter//8)
                f_bgn = self.rng.integers(0, freq_filter-f_gap)
                if self.freq_dim == 3:
                    x[:, :, :, f_bgn:f_bgn+f_gap] = 0
                elif self.freq_dim == 2:
                    x[:, :, f_bgn:f_bgn+f_gap, :] = 0
                else:
                    print("Error: freq_dim only 2 or 3...")
                    sys.exit()
        return x

class module_grl(torch.autograd.Function):
    def __init__(self):
        super(module_grl, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

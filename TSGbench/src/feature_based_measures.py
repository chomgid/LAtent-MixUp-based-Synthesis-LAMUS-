"""
This file is adapted from the official TSGBench implementation:
https://github.com/YihaoAng/TSGBench

The original code has been modified to fit the purposes of our research.
"""

import torch
import numpy as np
from torch import nn
from typing import List, Tuple

# Basic class of loss
class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

# =======================================
# MDD calculation and utilities functions
def histogram_torch(x, n_bins, density=False, normalize=True, fixed_min=None, fixed_max=None):
    if fixed_min is not None and fixed_max is not None:
        a, b = fixed_min, fixed_max
    else:
        a, b = x.min().item(), x.max().item()
        b = b + 1e-5 if b == a else b

    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0] 
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()

    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])

    if normalize:
        num_elements = float(x.shape[0] * x.shape[1])
        if num_elements == 0:
            count = torch.zeros_like(count) 
        else:
            count = count / num_elements
            
    return count, bins

class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.n_bins = n_bins 
        self.densities = list()
        self.bin_min_vals = list() 
        self.bin_max_vals = list() 
        
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_bin_min_vals = list()
            tmp_bin_max_vals = list()

            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, self.n_bins, density=False, normalize=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))

                a, b = x_ti.min().item(), x_ti.max().item()
                b = b + 1e-5 if b == a else b
                tmp_bin_min_vals.append(a)
                tmp_bin_max_vals.append(b) 
                
            self.densities.append(tmp_densities)
            self.bin_min_vals.append(tmp_bin_min_vals)
            self.bin_max_vals.append(tmp_bin_max_vals)


    def compute(self, x_fake):
        loss = list()

        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                x_ti_fake = x_fake[:, t, i].reshape(-1, 1)
                fixed_min = self.bin_min_vals[i][t]
                fixed_max = self.bin_max_vals[i][t]
                
                d_fake, _ = histogram_torch(x_ti_fake, self.n_bins, 
                                            density=False, normalize=True,
                                            fixed_min=fixed_min, fixed_max=fixed_max)
                
                abs_metric = torch.abs(
                    d_fake - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
                
        loss_componentwise = torch.stack(loss)
        return loss_componentwise
    
def calculate_mdd(ori_data, gen_data, device):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    mdd = (HistoLoss(ori_data[:, 1:, :], n_bins=50, name='marginal_distribution')(gen_data[:, 1:, :])).detach().cpu().numpy()
    return mdd.item()

# =======================================
# ACF calculation and utilities functions
def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        ##### Need to Check here
        y = x[:, i:, :] * x[:, :-i, :] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)

def non_stationary_acf_torch(X, symmetric=False):
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0))
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations

def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))

class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))

def calculate_acd(ori_data, gen_data, device):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    acf = (ACFLoss(ori_data, name='auto_correlation', stationary=True)(gen_data)).detach().cpu().numpy()
    return acf.item()

# =======================================
# SD calculation and utilities functions
def skew_torch(x, dim=0):
    x = x - x.mean(dim)
    x_3 = torch.pow(x, 3).mean(dim)
    x_std = x.std(dim, unbiased=True)
    x_std_3 = torch.pow(x_std, 3)
    skew = torch.where(x_std_3 == 0, torch.zeros_like(x_std_3), x_3 / x_std_3)
    return skew

class SkewnessLoss:
    def __init__(self, x_real, threshold, **kwargs):
        self.norm_foo = torch.abs
        self.skew_real = skew_torch(x_real)
        self.threshold = threshold

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(x_fake)
        relative_diff = torch.where(torch.abs(self.skew_real) >= self.threshold,
                                   (skew_fake - self.skew_real) / self.skew_real,
                                   (skew_fake - self.skew_real))
        return self.norm_foo(relative_diff)

def calculate_sd(ori_data, gen_data, device, threshold=1):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    if torch.any(torch.isnan(ori_data)) or torch.any(torch.isnan(gen_data)):
        raise ValueError("Input data contains NaN values")
    skewness = SkewnessLoss(ori_data, threshold, name='skew')
    sd = skewness.compute(gen_data)
    sd = torch.nanmean(sd)
    sd = float(sd.numpy())
    return sd

# =======================================
# KD calculation and utilities functions
def kurtosis_torch(x, dim=0, excess=True):
    x = x - x.mean(dim)
    x_4 = torch.pow(x, 4).mean(dim)
    x_var = torch.var(x, dim=dim, unbiased=False)
    x_var2 = torch.pow(x_var, 2)
    kurtosis = torch.where(x_var2 == 0, torch.zeros_like(x_var2), x_4 / x_var2)
    if excess:
        kurtosis = kurtosis - 3
    return kurtosis

class KurtosisLoss:
    def __init__(self, x_real, threshold, **kwargs):
        self.norm_foo = torch.abs
        self.kurtosis_real = kurtosis_torch(x_real)
        self.threshold = threshold

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(x_fake)
        relative_diff = torch.where(torch.abs(self.kurtosis_real) >= self.threshold,
                                   (kurtosis_fake - self.kurtosis_real) / self.kurtosis_real,
                                   (kurtosis_fake - self.kurtosis_real))
        return self.norm_foo(relative_diff)

def calculate_kd(ori_data, gen_data, device, threshold=1):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    if torch.any(torch.isnan(ori_data)) or torch.any(torch.isnan(gen_data)):
        raise ValueError("Input data contains NaN values")
    kurtosis = KurtosisLoss(ori_data, threshold, name='kurtosis')
    kd = kurtosis.compute(gen_data)
    kd = torch.nanmean(kd)
    kd = float(kd.numpy())
    return kd


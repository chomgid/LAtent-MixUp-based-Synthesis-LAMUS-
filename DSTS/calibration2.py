import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

def order_channel(ori_data):
    data = ori_data.reshape(-1,ori_data.shape[-1])  # shape (n*l, c)
    sd = np.std(data, axis=0)

    return np.argsort(sd)


def calibration(ori_data:np.ndarray, aug_data:np.ndarray, tot_iter, aug):
    n, l, c = ori_data.shape     # shape (n, l, c)
    m, _, _ = aug_data.shape     # shape (m, l, c)
    calib_weights = np.ones(len(aug_data)) / m
    index = order_channel(ori_data)
    
    # benchmark information
    desired_means = np.mean(ori_data, axis=0)      # shape (1, l, c)

    for k in tqdm(index, desc="Computing calibration weights", position=0):
        for i in range(l):
            benchmark = desired_means[i,k]
            x_to_calib = aug_data[:,i,k]
            def eq_to_calib(lambda_est):
                calib_weights_sum = calib_weights.sum()
                w_exp = calib_weights * np.exp(lambda_est[0] + lambda_est[1] * x_to_calib)
                w_exp_sum = w_exp.sum()
                return + w_exp_sum - (lambda_est[0] * calib_weights_sum) - (lambda_est[1] * benchmark)

            def eq_to_calib_jac(lambda_est):
                calib_weights_sum = calib_weights.sum()
                w_exp = calib_weights * np.exp(lambda_est[0] + lambda_est[1] * x_to_calib)
                w_exp_sum = w_exp.sum()
                w_exp_x_sum = (w_exp * x_to_calib).sum()
                eq_1 = + w_exp_sum - calib_weights_sum
                eq_2 = + w_exp_x_sum - benchmark
                return [eq_1, eq_2]

            def eq_to_calib_hess(lambda_est):
                calib_weights_sum = calib_weights.sum()
                w_exp = calib_weights * np.exp(lambda_est[0] + lambda_est[1] * x_to_calib)
                w_exp_sum = w_exp.sum()
                w_exp_x_sum = (w_exp * x_to_calib).sum()
                w_exp_xsq_sum = (w_exp * (x_to_calib**2)).sum()
                return [[w_exp_sum, w_exp_x_sum], [w_exp_x_sum, w_exp_xsq_sum]]

            lambda_0, lambda_1 = minimize(eq_to_calib, np.array([0.0, 0.0]), jac=eq_to_calib_jac, hess=eq_to_calib_hess, 
                                          method='trust-ncg').x
            calib_weights = calib_weights * np.exp(lambda_0 + lambda_1 * x_to_calib)

    eps = 1e-8
    weights_calib = calib_weights+ eps
    weights_calib /= weights_calib.sum()
    indices = np.random.choice(np.arange(len(aug_data)), size=int(aug*n), p=weights_calib, replace=False)
    calib_data = aug_data[indices]

    return calib_data


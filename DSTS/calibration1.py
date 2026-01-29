import numpy as np
from tqdm import tqdm

def num_Newton(lambda_, weights, aug_data, desired_means, n):
    """
    numerator of Newton's method (f(lambda))
    """
    numerator = np.sum(weights * np.exp(-aug_data * lambda_ )*aug_data)
    denominator = np.sum(weights*np.exp(-aug_data * lambda_ ))
    return ((numerator*n)/denominator - desired_means)

def denom_Newton(lambda_, weights, aug_data, n):
    """
    denominator of Newton's method (f'(lambda))
    """
    A = np.sum(weights * np.exp(-aug_data * lambda_ ) * aug_data)
    B = np.sum(weights * np.exp(-aug_data * lambda_ ))
    dA = -np.sum(weights * np.exp(-aug_data * lambda_ ) * (aug_data**2))
    dB = -A
    return (dA*B-A*dB)*n/(B**2)

def order_channel(ori_data):
    data = ori_data.reshape(-1,ori_data.shape[-1])  # shape (n*l, c)
    sd = np.std(data, axis=0)

    return np.argsort(sd)


# optimized by Newton's method
def calibration(ori_data:np.ndarray, aug_data:np.ndarray, tot_iter, aug):    
    n, l, c = ori_data.shape     # shape (n, l, c)
    m, _, _ = aug_data.shape     # shape (m, l, c)
    init_weights = np.ones(len(aug_data)) / m

    index = order_channel(ori_data)
    ori_data = ori_data[:,:,index]
    criterion = 1e-10

    # benchmark information
    desired_means = np.mean(ori_data, axis=0)   # shape (1, l, c)

    # lambda update usig Newton's method (iter*tot_iter times)
    weights=init_weights
    for k in tqdm(index, desc="Computing calibration weights", position=0):
        for t in range(tot_iter):
            for i in range(l):
                lamb = 0
                while True:
                    num = num_Newton(lamb, weights, aug_data[:, i, k], desired_means[i, k], 1)
                    denom = denom_Newton(lamb, weights, aug_data[:, i, k], 1)
                    eps = num/denom
                    if np.isinf(eps):
                        raise ValueError("eps is inf during Newton's method")
                    elif np.isnan(eps):
                        raise ValueError("eps is NaN during Newton's method")
                    elif np.abs(eps) < criterion:
                        break

                    lamb = lamb - eps

                weights_calib=weights*np.exp(-aug_data[:, i, k]*lamb)
                weights = weights_calib/np.sum(weights_calib)
                

    # weights normalization
    weights_calib = weights / np.sum(weights)

    # probability-proportional-to-size without replacement sampling using normalized weights
    indices = np.random.choice(np.arange(len(aug_data)), size=int(aug*n), p=weights_calib, replace=False)
    calib_data = aug_data[indices]

    return calib_data

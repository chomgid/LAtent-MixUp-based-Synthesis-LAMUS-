import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


# SNN matching
def make_rs_index(data_list, k):
    n = data_list[0].shape[0]
    index_array = np.arange(n)
    rs_index = np.empty((n, k), dtype=int)

    for i in tqdm(range(n), desc="Generating mixup indices"):
        mask = np.arange(n) != i
        xmats = [xmat[mask] for xmat in data_list]
        targets = [xmat[i:i+1] for xmat in data_list]
        prob = snn_proba(xmats, targets, None)
        if np.any(np.isnan(prob)):  
            raise ValueError("probability contains NaN values.")
        del_arr = index_array[np.arange(n) != i]
        new_indices = np.random.choice(del_arr, size=k, replace=True, p=prob)
        rs_index[i] = new_indices
    return rs_index

# calculate probabilities for SNN matching
def snn_proba(xmats, targets, inv_cov_, temp=1.0):
    distances = np.zeros(xmats[0].shape[0], dtype=float)
    for i in range(len(xmats)):
        xmat = xmats[i]
        target = targets[i]
        q = xmat.shape[1]
        distances += np.sum((xmat-target)**2, axis=1)/q
    
    # use log sum exp trick
    min = np.min(distances)
    denom = np.log(sum(np.exp(-(distances-min)/temp)))-min/temp
    num = -distances/temp
    prob = np.exp(num-denom)
    return prob


def mixup(X, Z, k, alpha=0.5, beta=0.5):
    n = Z[0].shape[0]
    rs_index = make_rs_index(Z, k) 

    mixup = []
    for j in range(k):
        rs = X[rs_index[:,j]]                                 # shape (n, l, c)               
        lamb = np.random.beta(a=alpha, b=beta, size=(n,1,1))
        mixup.append(lamb*X + (1-lamb)*rs)
    mixup_matrix = np.vstack(mixup)                              # shape (n*k, l, c)

    return mixup_matrix


def prin_mixup(Z, sample_means, k, alpha=0.5, beta=0.5):
    n = Z[0].shape[0]
    rs_index = make_rs_index(Z, k) 

    lambdas = np.random.beta(alpha, beta, size=(k, n, 1))

    mixup_Z = []
    mixup_mean = []
    for Z_c in Z:
        mixed_zc_list = []
        for j in range(k):
            Zj = Z_c[rs_index[:, j]]                      # (n, q_c)
            lam = lambdas[j]                              # (n, 1)
            mixed_zc = lam * Z_c + (1 - lam) * Zj         # (n, q_c)
            mixed_zc_list.append(mixed_zc)
        mixup_Z.append(np.vstack(mixed_zc_list))

    if sample_means:
        for mean_c in sample_means:
            mixed_mc_list = []
            for j in range(k):
                mj = mean_c[rs_index[:, j]]                   # (n, 1)
                lam = lambdas[j]                              # (n, 1)
                mixed_mc = lam * mean_c + (1 - lam) * mj      # (n, 1)
                mixed_mc_list.append(mixed_mc)
            mixup_mean.append(np.vstack(mixed_mc_list))

    return mixup_Z, mixup_mean



####################################################################

# random matching
def make_rs_index_random(data_list, k):
    n = data_list[0].shape[0]
    index_array = np.arange(n)
    rs_index = np.empty((n, k), dtype=int)

    for i in tqdm(range(n), desc="Generating mixup indices"):
        del_arr = index_array[np.arange(n) != i]
        rs_index[i] = np.random.choice(del_arr, k, replace=True)
    return rs_index

def make_rs_index_NN(data_list,  k):
    n = data_list[0].shape[0]
    index_array = np.arange(n)
    rs_index = np.empty((n, k), dtype=int)

    for i in tqdm(range(n), desc="Generating mixup indices"):
        mask = np.arange(n) != i
        xmats = [xmat[mask] for xmat in data_list]
        targets = [xmat[i:i+1] for xmat in data_list]
        nn_index = find_NN_index(xmats, targets, n)
        del_arr = index_array[mask]
        rs_index[i] = np.full(k, del_arr[nn_index])
    return rs_index

def find_NN_index(xmats, targets, n):
    distances = np.zeros(n-1)
    for i in range(len(xmats)):
        xmat = xmats[i]
        target = targets[i]
        q = xmat.shape[1]
        distances += np.sum((xmat-target)**2, axis=1)/q
    nn_index = np.argmin(distances)
    return nn_index
####################################################################

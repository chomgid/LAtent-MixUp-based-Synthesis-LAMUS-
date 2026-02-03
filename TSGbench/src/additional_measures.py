import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from torch import nn
from typing import List, Tuple
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial import distance_matrix
import gc

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
# MD calculation and utilities functions
def mean_torch(x, dim=0):
    return x.mean(dim)

class MeanLoss(Loss):
    def __init__(self, x_real, threshold, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean_real = mean_torch(x_real)
        self.threshold = threshold

    def compute(self, x_fake, **kwargs):
        mean_fake = mean_torch(x_fake)
        relative_diff = torch.where(torch.abs(self.mean_real) >= self.threshold,
                                   (mean_fake - self.mean_real) / self.mean_real,
                                   (mean_fake - self.mean_real)) 
        return self.norm_foo(relative_diff)
    
def calculate_md(ori_data, gen_data, device, threshold=1):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    mean = MeanLoss(ori_data, threshold, name='mean')
    md = np.nanmean(mean.compute(gen_data).numpy())
    md = float(md)
    return md

# =======================================
# STD calculation and utilities functions
def std_torch(x, dim=0):
    return x.std(dim)

class STDLoss(Loss):
    def __init__(self, x_real, threshold, **kwargs):
        super(STDLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = std_torch(x_real)
        self.threshold = threshold

    def compute(self, x_fake):
        std_fake = std_torch(x_fake)
        relative_diff = torch.where(torch.abs(self.std_real) >= self.threshold,
                                   (std_fake - self.std_real) / self.std_real,
                                   (std_fake - self.std_real))
        return self.norm_foo(relative_diff)
    
def calculate_std(ori_data, gen_data, device, threshold=1):
    if not torch.is_tensor(ori_data):
        ori_data = torch.tensor(ori_data)
    if not torch.is_tensor(gen_data):
        gen_data = torch.tensor(gen_data)
    std = STDLoss(ori_data, threshold, name='std')
    s = np.nanmean(std.compute(gen_data).numpy())
    s = float(s)
    return s

# =======================================
# DUPI-like score 
def calculate_dupilike(ori_data, fake_data):
    sum = 0
    ori_data = ori_data.squeeze()
    fake_data = fake_data.squeeze()
    for i in range(len(ori_data)):
        data = ori_data[i, :]
        
        ori_data_excl = np.delete(ori_data, i, axis=0) 
        combined = np.vstack((fake_data, ori_data_excl))
        # Calculate distances
        distances = np.linalg.norm(combined - data, ord=1)
        if np.argmin(distances)<len(fake_data):
            sum += 1

    dupi_like = sum/len(ori_data)
    return dupi_like

# =======================================
# Time Series DUPI
def calculate_dupi(ori, syn, k, device, batch_size = 64):
    ori = torch.tensor(ori, dtype=torch.float32)
    syn = torch.tensor(syn, dtype=torch.float32)

    n_samples = ori.shape[0]
    m_samples = syn.shape[0]

    # Error handling
    if n_samples == 0:
        print("Error: n_samples is 0")
        return float('nan')
    
    if k > n_samples - 1:
        print(f"Error: k={k} is too large for ori with {n_samples} samples")
        return float('nan')
    if k > syn.shape[0]:
        print(f"Error: k={k} is too large for syn with {m_samples} samples")
        return float('nan')
    
    ori_flat = ori.reshape(n_samples, -1)
    syn_flat = syn.reshape(m_samples, -1)
    
    distance_x = []
    for i in range(0, n_samples, batch_size):
        i_end = min(i + batch_size, n_samples)
        batch = ori_flat[i:i_end]
        dists = torch.cdist(batch, ori_flat, p=2)
        for j in range(i_end - i):
            dists[j, i + j] = float('inf')  # exclude self
        kth_vals = torch.kthvalue(dists, k, dim=1).values
        distance_x.append(kth_vals)
        del dists
        gc.collect()
    distance_x = torch.cat(distance_x, dim=0)

    # Compute kth nearest neighbor distance from syn
    distance_y = []
    for i in range(0, n_samples, batch_size):
        i_end = min(i + batch_size, n_samples)
        batch = ori_flat[i:i_end]
        dists = torch.cdist(batch, syn_flat, p=2)
        kth_vals = torch.kthvalue(dists, k, dim=1).values
        distance_y.append(kth_vals)
        del dists
        gc.collect()
    distance_y = torch.cat(distance_y, dim=0)
    
    count = (distance_y <= distance_x).sum().item()
    
    # Compute DUPI
    dupi = count / n_samples

    return dupi



# =======================================
# Quantile difference score 
def calculate_qd(ori_data, fake_data, device, quantile_list=[0.1, 0.3, 0.5, 0.7, 0.9], 
                 threshold=1):
    
    quantile_list = np.array(quantile_list)

    ori_quantile = np.quantile(ori_data, q=quantile_list, axis=0)
    fake_quantile = np.quantile(fake_data, q=quantile_list, axis=0)

    abs_diff = np.abs(fake_quantile - ori_quantile)

    mask = np.abs(ori_quantile) >= threshold

    qd = abs_diff.copy()
    qd[mask] = abs_diff[mask] / np.abs(ori_quantile[mask])

    return np.nanmean(qd)


# =======================================
# Cluster score
def find_optimal_clusters(data, max_k=10):
    best_k = 2
    best_score = 0

    for k in range(2, max_k + 1):
        try:
            model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", random_state=0)
            labels = model.fit_predict(data)
            # score = silhouette_score(data.reshape((data.shape[0], -1)), labels)
            score = calinski_harabasz_score(data.reshape((data.shape[0], -1)), labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            continue
    return best_k

def calculate_cls(ori_data, syn_data, data_name, caliper_scale=1.0):
    try:
        with open(f'/data/chomgid14/TSdata/{data_name}/tskm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        ori_labels = model.predict(ori_data)
        n_clusters = model.n_clusters
    except:
        print("No model found")
        # n_clusters = find_optimal_clusters(ori_data)
        n_clusters = 7
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean")
        with open(f'/data/chomgid14/TSdata/{data_name}/tskm_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        ori_labels = model.fit_predict(ori_data)

    centroids = model.cluster_centers_

    cluster_thresholds = []
    for k in range(n_clusters):
        cluster_members = ori_data[ori_labels == k]
        if len(cluster_members) == 0:
            cluster_thresholds.append(0.0)
            continue
        dists = [np.linalg.norm(np.squeeze(x) - np.squeeze(centroids[k])) for x in cluster_members]
        threshold = caliper_scale * max(dists)
        cluster_thresholds.append(threshold)

    syn_labels = []
    for x in syn_data:
        x = np.squeeze(x)
        dists = [np.linalg.norm(x - np.squeeze(c)) for c in centroids]
        min_idx = np.argmin(dists)
        if dists[min_idx] <= cluster_thresholds[min_idx]:
            syn_labels.append(min_idx)
        else:
            syn_labels.append(n_clusters)  # outlier

    syn_labels = np.array(syn_labels)

    ori_counts = np.bincount(ori_labels, minlength=n_clusters)
    syn_counts = np.bincount(syn_labels, minlength=n_clusters + 1)

    ori_prop = ori_counts / len(ori_data)
    syn_prop = syn_counts / len(syn_data)

    epsilon = 1e-12
    syn_prop = np.clip(syn_prop, epsilon, 1.0)
    ori_prop = np.pad(ori_prop, (0, 1), constant_values=0)

    print("Original proportions:", np.array2string(ori_prop, precision=4, suppress_small=True))
    print("Synthetic proportions:", np.array2string(syn_prop, precision=4, suppress_small=True))

    # Cross-entropy
    cross_entropy = -np.sum(ori_prop * np.log(syn_prop))

    return cross_entropy


# =======================================
# Cosine similarity difference
def calculate_csd(ori_data, syn_data, threshold=1e-8):
    ori_arr = np.asarray(ori_data)
    syn_arr = np.asarray(syn_data)

    def time_time_cosine_matrix(x):
        norms = np.linalg.norm(x, axis=0, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)  # prevent division by 0
        normed = x / norms
        return normed.T @ normed

    C = ori_arr.shape[2]
    scores = []
    for c in range(C):
        cos_ori = time_time_cosine_matrix(ori_arr[:, :, c])
        cos_syn = time_time_cosine_matrix(syn_arr[:, :, c])
        diff = cos_syn - cos_ori
        score = np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(cos_ori ** 2) + 1e-12)  # Stability
        scores.append(score)

    return float(np.nanmean(scores))


# =======================================   
# Precision and Recall
def precision_recall(orig, synth, pca = True, exp_var=0.9, disc_dim=5, binning_method='quantile'):
    n_samples, length, dim = orig.shape
    
    all_orig_cat = []
    all_synth_cat = []

    for d in range(dim):
        orig_d = orig[:, :, d]
        synth_d = synth[:, :, d]
        
        if pca:
            pca = PCA()
            pca.fit(orig_d)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            # q = max(np.searchsorted(cum_var, exp_var) + 1, 2) 
            q=5
            orig_q = pca.transform(orig_d)[:, :q]
            synth_q = pca.transform(synth_d)[:, :q]
        else:
            q = orig_d.shape[1]
            orig_q = orig_d
            synth_q = synth_d

        # 범주화 (binning)
        orig_cat = np.zeros_like(orig_q, dtype=int)
        synth_cat = np.zeros_like(synth_q, dtype=int)

        for i in range(q):
            if binning_method == 'quantile':
                orig_cat[:, i] = pd.qcut(orig_q[:, i], q=disc_dim, labels=False, duplicates='drop') + 1
                _, bin_edges = pd.qcut(orig_q[:, i], q=disc_dim, retbins=True, duplicates='drop')
                internal_edges = bin_edges[1:-1]
                s_min = min(np.min(synth_q[:, i]), bin_edges[0])
                s_max = max(np.max(synth_q[:, i]), bin_edges[-1])
                new_edges = np.concatenate([[s_min], internal_edges, [s_max]])

                synth_cat[:, i] = pd.cut(synth_q[:, i], bins=new_edges, labels=False, include_lowest=True, duplicates='drop') + 1

            elif binning_method == 'equal_length':
                min_val_orig = np.min(orig_q[:, i])
                max_val_orig = np.max(orig_q[:, i])
                bin_edges = np.linspace(min_val_orig, max_val_orig, disc_dim + 1)
                orig_cat[:, i] = pd.cut(orig_q[:, i], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop') + 1

                internal_edges = bin_edges[1:-1]
                s_min = min(np.min(synth_q[:, i]), bin_edges[0])
                s_max = max(np.max(synth_q[:, i]), bin_edges[-1])
                new_edges = np.concatenate([[s_min], internal_edges, [s_max]])
                
                synth_cat[:, i] = pd.cut(synth_q[:, i], bins=new_edges, labels=False, include_lowest=True, duplicates='drop') + 1
            else:
                raise ValueError("binning_method must be 'quantile' or 'equal_length'")

        all_orig_cat.append(orig_cat)
        all_synth_cat.append(synth_cat)

    orig_all = np.concatenate(all_orig_cat, axis=1)
    synth_all = np.concatenate(all_synth_cat, axis=1)

    orig_unique_keys, orig_counts = np.unique([tuple(row) for row in orig_all], axis=0, return_counts=True)
    synth_unique_keys, synth_counts = np.unique([tuple(row) for row in synth_all], axis=0, return_counts=True)

    A = set(map(tuple, orig_unique_keys))  # Original unique string vector set 
    A_star = set(map(tuple, synth_unique_keys)) # Synthetic unique string vector set 
    A_intersect_A_star = A.intersection(A_star)

    # Calculate m_{A \cap A*} and m_{A*} for Precision 
    m_A_intersect_A_star = 0
    for i, key in enumerate(synth_unique_keys):
        if tuple(key) in A_intersect_A_star:
            m_A_intersect_A_star += synth_counts[i]

    m_A_star = len(synth_all) 

    # Calculate n_{A \cap A*} and n_A for Recall 
    n_A_intersect_A_star = 0
    for i, key in enumerate(orig_unique_keys):
        if tuple(key) in A_intersect_A_star:
            n_A_intersect_A_star += orig_counts[i]

    n_A = len(orig_all)

    precision = m_A_intersect_A_star / m_A_star if m_A_star > 0 else 0
    recall = n_A_intersect_A_star / n_A if n_A > 0 else 0

    return precision, recall
"""
This file is adapted from the official TSGBench implementation:
https://github.com/YihaoAng/TSGBench

The original code has been modified to fit the purposes of our research.
"""

import os
import datetime
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from .ts2vec import initialize_ts2vec
from .feature_based_measures import *
from .additional_measures import *
from .visualization import visualize_tsne, visualize_distribution
from .utils import show_with_start_divider, show_with_end_divider, determine_device, write_json_data
from .diffts.context_fid import Context_FID
from .dspstorch.ds_torch import discriminative_score_metrics
from .dspstorch.ps_torch import predictive_score_metrics
from .dspstorch.ps_mlp import predictive_score_mlp

# from .timegan.predictive_metrics import predictive_score_metrics
# from .timegan.discriminative_metrics import discriminative_score_metrics


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_ed(ori_data,gen_data):
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0
        for j in range(n_series):
            distance = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_distance_eu += distance
        distance_eu.append(total_distance_eu / n_series)

    distance_eu = np.array(distance_eu)
    average_distance_eu = distance_eu.mean()
    return average_distance_eu

def calculate_dtw(ori_data,comp_data):
    distance_dtw = []
    n_samples = ori_data.shape[0]
    for i in range(n_samples):
        distance = multi_dtw_distance(ori_data[i].astype(np.double), comp_data[i].astype(np.double), use_c=True)
        distance_dtw.append(distance)

    distance_dtw = np.array(distance_dtw)
    average_distance_dtw = distance_dtw.mean()
    return average_distance_dtw




def evaluate_data(cfg, ori_data, gen_data, device):
    show_with_start_divider(f"Evalution with settings:{cfg}")

    # Parse configs
    method_list = cfg.get('method_list','[C-FID,MDD,ACD,SD,KD,ED,DTW,t-SNE,Distribution]')
    #result_path = cfg.get('result_path',r'./result/')
    dataset_name = cfg.get('dataset_name','dataset')
    model_name = cfg.get('model','TimeVAE')
    no_cuda = cfg.get('no_cuda',False)
    result_path = cfg.get('result_path','./result/')

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d-%H%M%S")
    combined_name = f'{model_name}_{dataset_name}_{formatted_time}'

    if not isinstance(method_list,list):
        method_list = method_list.strip('[]')
        method_list = [method.strip() for method in method_list.split(',')]

    # Check original data
    if ori_data is None:
        show_with_end_divider('Error: Original data not found.')
        return None
    if isinstance(ori_data, (list, tuple)) and len(ori_data) == 2:
        train_data, valid_data = ori_data
        indices = np.random.choice(train_data.shape[0], size=min(train_data.shape[0],gen_data.shape[0]), replace=False)
        ori_data = train_data[indices]
       
    else:
        show_with_end_divider('Error: Original data is invalid.')
        return None

    # Check original data
    if gen_data is None:
        show_with_end_divider('Error: Generated data not found.')
        return None
    if ori_data.shape != gen_data.shape:
        print(f'Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}.')
        show_with_end_divider('Error: Generated data does not have the same shape with original data.')
        return None
    
    # Execute eval method in method list
    results = {}


   
    ##################### Marginal measures ########################
    shape = ori_data.shape
    result = {}

    scaler = MinMaxScaler()
    norm_ori = np.zeros_like(ori_data)
    norm_gen = np.zeros_like(gen_data)

    # feature-wise normalization
    for d in range(shape[2]):
        combined = np.concatenate([
            ori_data[:, :, d].reshape(-1, 1),
            gen_data[:, :, d].reshape(-1, 1)
        ], axis=0)
        scaler.fit(combined)
        norm_ori[:, :, d] = scaler.transform(ori_data[:, :, d].reshape(-1, 1)).reshape(shape[0], shape[1])
        norm_gen[:, :, d] = scaler.transform(gen_data[:, :, d].reshape(-1, 1)).reshape(shape[0], shape[1])


    mcentered_ori = ori_data - np.mean(ori_data, axis=1, keepdims=True)
    mcentered_gen = gen_data - np.mean(gen_data, axis=1, keepdims=True)
    
    # Model-based measures
    if 'DS' in method_list:
        print('DS')
        start = time.time()
        disc = discriminative_score_metrics(norm_ori, norm_gen, device)
        elapsed = time.time() - start
        print(f'DS took {elapsed:.4f} seconds')
        result['DS'] = disc
    if 'C-FID' in method_list:
        print('C-FID')
        start = time.time()
        result['C-FID'] = Context_FID(norm_ori, norm_gen, device)
        elapsed = time.time() - start
        print(f'C-FID took {elapsed:.4f} seconds')

    # Feature-based measures
    if 'MDD' in method_list:
        print('MDD')
        start = time.time()
        mdd = calculate_mdd(norm_ori, norm_gen, device)
        elapsed = time.time() - start
        print(f'MDD took {elapsed:.4f} seconds')
        result['MDD'] = mdd
    if 'ACD' in method_list:
        print('ACD')
        start = time.time()
        acd = calculate_acd(mcentered_ori, mcentered_gen, device)
        elapsed = time.time() - start
        print(f'ACD took {elapsed:.4f} seconds')
        result['ACD'] = acd
    if 'SD' in method_list:
        print('SD')
        start = time.time()
        sd = calculate_sd(ori_data, gen_data, device)
        elapsed = time.time() - start
        print(f'SD took {elapsed:.4f} seconds')
        result['SD'] = sd

    if 'KD' in method_list:
        print('KD')
        start = time.time()
        kd = calculate_kd(ori_data, gen_data, device)
        elapsed = time.time() - start
        print(f'KD took {elapsed:.4f} seconds')
        result['KD'] = kd
    # Additional measures
    if 'DUPI' in method_list:
        print('DUPI')
        start = time.time()
        dupi = calculate_dupi(ori_data, gen_data, 1, device)
        elapsed = time.time() - start
        print(f'DUPI took {elapsed:.4f} seconds')
        result['DUPI'] = dupi
    if 'QD' in method_list:
        print('QD')
        start = time.time()
        qd = calculate_qd(ori_data, gen_data, device)
        elapsed = time.time() - start
        print(f'QD took {elapsed:.4f} seconds')
        result['QD'] = qd
    if 'MD' in method_list:
        print('MD')
        start = time.time()
        md = calculate_md(ori_data, gen_data, device)
        elapsed = time.time() - start
        print(f'MD took {elapsed:.4f} seconds')
        result['MD'] = md
    if 'STD' in method_list:
        print('STD')
        start = time.time()
        std = calculate_std(ori_data, gen_data, device)
        elapsed = time.time() - start
        print(f'STD took {elapsed:.4f} seconds')
        result['STD'] = std
    if 'CLS' in method_list:
        print('CLS')
        start = time.time()
        cls = calculate_cls(mcentered_ori, mcentered_gen, dataset_name)
        elapsed = time.time() - start
        print(f'CLS took {elapsed:.4f} seconds')
        result['CLS'] = cls
    if 'CSD' in method_list:
        print('CSD')
        start = time.time()
        csd = calculate_csd(mcentered_ori, mcentered_gen)
        elapsed = time.time() - start
        print(f'CSD took {elapsed:.4f} seconds')
        result['CSD'] = csd
    if 'PS' in method_list:
        print('PS')
        start = time.time()
        pred1 = predictive_score_metrics(norm_ori, norm_gen, device)
        elapsed = time.time() - start
        print(f'PS took {elapsed:.4f} seconds')
        result['PS'] = pred1
    if 'PS2' in method_list:
        print('PS using MLP')
        start = time.time()
        pred1 = predictive_score_mlp(norm_ori, norm_gen, device)
        elapsed = time.time() - start
        print(f'PS2 took {elapsed:.4f} seconds')
        result['PS2'] = pred1
    if 'PRQ' in method_list:
        print('Precision and Recall')
        start = time.time()
        precision, recall  =precision_recall(mcentered_ori, mcentered_gen, pca = True, exp_var=0.9, disc_dim=5, binning_method='quantile')
        elapsed = time.time() - start
        print(f'PR took {elapsed:.4f} seconds')
        result['precisionQ'] = precision
        result['recallQ'] = recall
    if 'PRE' in method_list:
        print('Precision and Recall')
        start = time.time()
        precision, recall  =precision_recall(mcentered_ori, mcentered_gen, pca = True, exp_var=0.9, disc_dim=5, binning_method='equal_length')
        elapsed = time.time() - start
        print(f'PR took {elapsed:.4f} seconds')
        result['precisionE'] = precision
        result['recallE'] = recall


    # Distance-based measures
    if 'ED' in method_list:
        print('ED')
        ed = calculate_ed(ori_data,gen_data)
        result['ED'] = ed
    if 'DTW' in method_list:
        print('DTW')
        dtw = calculate_dtw(ori_data,gen_data)
        result['DTW'] = dtw

    # Visualization
    if 't-SNE' in method_list:
        print('t-SNE')
        visualize_tsne(ori_data, gen_data, result_path, combined_name)
    if 'Distribution' in method_list:
        print('Distribution')
        visualize_distribution(ori_data, gen_data, result_path, combined_name)

    # if isinstance(result, dict):
    #     result_path = os.path.join(result_path, f'{combined_name}.json')
    #     # write_json_data(result, result_path)
    #     print(f'Evaluation results saved to {result_path}.')

    show_with_end_divider(f'Evaluation done. Results:{result}.')

    return result

import os
import mgzip
import pickle
import json
import torch
import numpy as np

PREPROCESSING_PARAS = ['do_preprocessing','original_data_path','output_ori_path','dataset_name','use_ucr_uea_dataset','ucr_uea_dataset_name','seq_length','valid_ratio','do_normalization']
GENERATION_PARAS = ['do_generation','model','dataset_name']

def show_divider():
    print("=" * 20)

def show_with_start_divider(content):
    show_divider()
    print(content)

def show_with_end_divider(content):
    print(content)
    show_divider()
    print()

def make_sure_path_exist(path):
    if os.path.isdir(path) and not path.endswith(os.sep):
        dir_path = path
    else:
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

def read_mgzip_data(path):
    with mgzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_mgzip_data(content, path):
    make_sure_path_exist(path)
    with mgzip.open(path, 'wb') as f:
        pickle.dump(content, f)

def write_json_data(content, path):
    make_sure_path_exist(path)
    with open('data.json', 'w') as json_file:
        json.dump(content, json_file, indent=4)

def determine_device(no_cuda=False, cuda_device=0):
    if no_cuda or not torch.cuda.is_available():
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()

    if not isinstance(cuda_device, int) or cuda_device < 0:
        raise ValueError(f"cuda_device must be a non-negative integer, got {cuda_device}")
    if cuda_device >= num_gpus:
        raise ValueError(f"cuda_device {cuda_device} is invalid; only {num_gpus} GPU(s) available")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        torch.cuda.init()
        device = torch.device('cuda:0')
    else:
        device = torch.device(f'cuda:{cuda_device}')
    torch.cuda.set_device(device)
    return device

class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data

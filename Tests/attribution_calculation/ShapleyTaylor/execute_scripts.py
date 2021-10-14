import numpy as np
import os
import sys
os.chdir('../../../')
sys.path.append('./BivariateShapley')
from utils_shapley import *

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = './Tests/attribution_calculation/BivariateShapley/slurm_scripts'
script_path = './Tests/attribution_calculation/BivariateShapley/'
# Make top level directories
mkdir_p(job_directory)

m = {
    'drug': 500,
    'divorce': 500,
    'census': 500,
    'MNIST': 500,
    'CIFAR10': 500,
    'IMDB': 500,
    'COPD': 500,
}
samples_per_job = {
    'drug': 25,
    'divorce': 25,
    'census': 25,
    'MNIST': 1,
    'CIFAR10': 1,
    'IMDB': 1,
    'COPD': 1,
}

start_indices = {
    'drug': np.arange(0,500,samples_per_job['drug']).tolist(),
    'divorce': np.arange(0,500,samples_per_job['divorce']).tolist(),
    'census': np.arange(0,500,samples_per_job['census']).tolist(),
    'MNIST': [0, 1, 2, 3, 4],
    'CIFAR10': [0, 1, 2, 3, 4], 
    'IMDB': [0, 1, 2, 3, 4], 
    'COPD': [0, 1, 2, 3, 4],
}

gpu = {
    'drug': False,
    'divorce': False,
    'census': False,
    'MNIST': True,
    'CIFAR10': True,
    'IMDB': True,
    'COPD': False,
}

datasets = ['census', 'drug', 'divorce']

for dataset in datasets:
    for start in start_indices[dataset]:
        job_file = os.path.join(job_directory,"%s_%s.job" %(dataset, start))
        python_script = 'iterate_%s.py' %dataset

        cmd = os.path.join(script_path, python_script)
        cmd = cmd + ' --dataset_min_index %s --dataset_samples %s --m %s' % (str(start), str(samples_per_job[dataset]), str(m[dataset]))
        submit_slurm(cmd, job_file, conda_env = 'shap', gpu = gpu[dataset], mem = 32, job_name = python_script[-11:-3])
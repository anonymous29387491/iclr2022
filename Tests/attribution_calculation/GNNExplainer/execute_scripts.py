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


samples_per_job = {
    'drug': 500,
    'divorce': 500,
    'census': 500,
    'MNIST': 50,
    'CIFAR10': 50,
    'IMDB': 50,
    'COPD': 10,
}

start_indices = {
    'drug': np.arange(0,500,samples_per_job['drug']).tolist(),
    'divorce': np.arange(0,500,samples_per_job['divorce']).tolist(),
    'census': np.arange(0,500,samples_per_job['census']).tolist(),
    'MNIST': np.arange(0,500,samples_per_job['MNIST']).tolist(),
    'CIFAR10': np.arange(0,500,samples_per_job['CIFAR10']).tolist(),
    'IMDB': np.arange(0,250,samples_per_job['IMDB']).tolist() + np.arange(12500,12750,samples_per_job['IMDB']).tolist(),
    'COPD': np.arange(0,500,samples_per_job['COPD']).tolist(),
}

gpu = {
    'drug': True,
    'divorce': True,
    'census': True,
    'MNIST': True,
    'CIFAR10': True,
    'IMDB': True,
    'COPD': False,
}

datasets = ['IMDB', 'CIFAR10', 'MNIST']
datasets = ['COPD']
datasets = ['census', 'drug', 'divorce']
datasets = ['drug']
datasets = ['census', 'drug', 'divorce', 'COPD', 'IMDB', 'CIFAR10', 'MNIST']
datasets = ['census', 'drug', 'divorce']

for dataset in datasets:
    for start in start_indices[dataset]:
        job_file = os.path.join(job_directory,"%s_%s.job" %(dataset, start))
        python_script = 'iterate_%s.py' %dataset

        cmd = os.path.join(script_path, python_script)
        cmd = cmd + ' --dataset_min_index %s --dataset_samples %s --m %s' % (str(start), str(samples_per_job[dataset]), str(m[dataset]))
        submit_slurm(cmd, job_file, conda_env = 'shap', gpu = gpu[dataset], mem = 32, job_name = python_script[-11:-3])
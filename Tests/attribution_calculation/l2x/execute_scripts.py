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

n_feats = {
    'drug': 6,
    'divorce': 54,
    'census': 12,
    'MNIST_196': 196,
    'CIFAR10_255': 255,
    'IMDB': 400,
    'COPD': 1079,
}

datasets = ['census', 'drug', 'divorce', 'IMDB', 'COPD', 'CIFAR10_255', 'MNIST_196']
pct_mask = np.arange(0, 1.04, 0.04)

for dataset in datasets:

    n_feat = n_feats[dataset]
    #n_feat_mask = np.unique(np.floor(n_feat * pct_mask)).astype(int)
    n_feat_mask = np.unique(np.floor(n_feat * pct_mask).astype(int))
    iter_list = (n_feat - n_feat_mask).tolist() # k is the parameter for important features to keep
    for k in iter_list:

        job_file = os.path.join(job_directory,"%s_%s.job" %(dataset, k))
        python_script = 'iterate_%s_l2x.py' %dataset
        path = os.path.join(script_path, python_script)

        cmd = os.path.join(script_path, python_script)
        cmd = cmd + ' --k %s' % (str(k))
        submit_slurm(cmd, job_file, conda_env = 'shap', gpu = True, mem = 32, job_name = python_script[-11:-3])

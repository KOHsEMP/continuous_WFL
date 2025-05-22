import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import requests
import pickle
import arff
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from scipy.stats import norm
from math import sqrt, floor, log
from joblib import Memory
from scipy.optimize import root_scalar
from scipy.special import comb
from scipy.optimize import fsolve

from utils import *
# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
def exec_ohe(df, ohe_cols, is_comp=False):
    ohe = OneHotEncoder(sparse_output=False, categories='auto')
    ohe.fit(df[ohe_cols])

    tmp_columns = []
    for i, col in enumerate(ohe_cols):
        tmp_columns += [f'{col}_{v}' for v in ohe.categories_[i]]
    
    df_tmp = pd.DataFrame(ohe.transform(df[ohe_cols]), columns=tmp_columns)
    
    # if the features are represented as complementary label, the value of the index assigned to the complementary label should be 0
    if is_comp: 
        df_tmp = df_tmp * (-1) + 1
    output_df = pd.concat([df.drop(ohe_cols, axis=1), df_tmp], axis=1)

    return output_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def missing_MCAR(df, miss_col_name, miss_rate, seed):
    set_seed(seed)
    output_df = df.copy()

    miss_index = sorted(random.sample(output_df.index.tolist(), int(output_df.shape[0] * miss_rate)))
    output_df.loc[miss_index, miss_col_name] = np.nan

    return output_df


def missing_MAR_logistic(miss_data, obs_data, miss_rate, seed):
    '''
    Ref: https://github.com/marineLM/NeuMiss/blob/58d54f1815bba847f9f06fe1910c95b754d9578c/python/amputation.py
    Args:
    '''

    set_seed(seed)

    n_samples, n_features = miss_data.shape
    missing_matrix = np.zeros_like(miss_data)

    all_data = np.concatenate([miss_data, obs_data], axis=1)
    mu = all_data.mean(0)
    cov = (all_data - mu).T.dot(all_data - mu) / all_data.shape[0]
    cov_obs = cov[np.ix_([i for i in range(miss_data.shape[1], all_data.shape[1])], [i for i in range(miss_data.shape[1], all_data.shape[1])])]
    coeffs = np.random.randn(obs_data.shape[1], miss_data.shape[1])
    v = np.array( [ coeffs[:, j].dot(cov_obs).dot(coeffs[:, j]) for j in range(miss_data.shape[1]) ] )
    steepness = np.random.uniform(low=0.1, high=0.5, size=miss_data.shape[1])
    coeffs /= steepness * np.sqrt(v)

    intercepts = np.zeros((miss_data.shape[1]))
    for j in range(miss_data.shape[1]):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(obs_data.dot(w) + b) - miss_rate
            return s.mean()

        res = fsolve(f, x0=0)
        intercepts[j] = res[0]
    
    ps = sigmoid(obs_data.dot(coeffs) + intercepts)
    ber = np.random.rand(n_samples, miss_data.shape[1])
    missing_matrix = ber < ps

    np.putmask(miss_data, missing_matrix, np.nan)

    return miss_data



def gen_params_selfmasking(n_features, miss_rate, prop_latent, sm_type, lam=None, k=None, seed=42,):
    '''
    Ref: https://github.com/marineLM/NeuMiss/blob/master/python/ground_truth.py
    miss_rate: float (0,1)
        missing rate
    prop_latent: float (0, 1)
    sm_type: str
        'gaussian' or 'probit'
    lam: 
        it is used when sm_type == 'probit'
    k:
        it is used when sm_type == 'gaussian'
    seed: int 
        random_seed
    '''

    assert miss_rate > 0 and miss_rate < 1
    assert prop_latent > 0 and prop_latent < 1

    set_seed(seed)

    # Generate covariance and mean
    # ---------------------------
    B = np.random.randn(n_features, int(prop_latent*n_features))
    cov = B.dot(B.T) + np.diag(
        np.random.uniform(low=0.01, high=0.1, size=n_features))

    mean = np.random.randn(n_features)

    # Adapt the remaining parameters of the selfmasking function to obtain the
    # desired missing rate
    # ---------------------
    sm_params = {}

    if sm_type == 'probit':
        sm_params['lambda'] = lam
        sm_params['c'] = np.zeros(n_features)
        for i in range(n_features):
            sm_params['c'][i] = lam*(mean[i] - norm.ppf(miss_rate)*np.sqrt(
                1/lam**2+cov[i, i]))

    elif sm_type == 'gaussian':
        sm_params['k'] = k
        sm_params['sigma2_tilde'] = np.zeros(n_features)

        min_x = miss_rate**2/(1-miss_rate**2)

        def f(x):
            y = -2*(1+x)*log(miss_rate*sqrt(1/x+1))
            return y

        for i in range(n_features):
            max_x = min_x
            while f(max_x) < k**2:
                max_x += 1
            sol = root_scalar(lambda x: f(x) - k**2, method='bisect',
                              bracket=(max_x-1, max_x), xtol=1e-3)

            sm_params['sigma2_tilde'][i] = sol.root*cov[i, i]


    return (n_features, sm_type, sm_params, mean, cov)


def missing_MNAR_selfmasking(data, miss_rate, prop_latent, sm_type, lam=None, k=None, seed=42):
    '''
    Ref: https://github.com/marineLM/NeuMiss/blob/master/python/ground_truth.py
    Args:
        data: np.array (n_samples, n_features)
        data_params: 
        seed: int
            random_seed
    '''
    n_samples = data.shape[0]
    n_features = data.shape[1]


    n_features, sm_type, sm_params, mean, cov = \
        gen_params_selfmasking(n_features=n_features, miss_rate=miss_rate, prop_latent=prop_latent, 
                               sm_type=sm_type, lam=lam, k=k, 
                                seed=seed,
                                )
    
    set_seed(seed)

    # decide missing vals
    missing_matrix = np.zeros_like(data)
    for j in range(n_features):
        X_j = data[:, j]
        if sm_type == 'probit':
            lam = sm_params['lambda']
            c = sm_params['c'][j]
            prob = norm.cdf(lam*X_j - c)
        elif sm_type == 'gaussian':
            k = sm_params['k']
            sigma2_tilde = sm_params['sigma2_tilde'][j]
            mu_tilde = mean[j] + k*sqrt(cov[j, j])
            prob = np.exp(-0.5*(X_j - mu_tilde)**2/sigma2_tilde)
        
        missing_matrix[:,j] = np.random.binomial(n=1, p=prob, size=len(X_j))

    # missing
    np.putmask(data, missing_matrix, np.nan)

    return data


def missing_process(dataset_name, df, miss_cols, miss_pattern, miss_rate, sm_k=None, sm_lam=None, sm_prop_latent=None, seed=42):
    '''
    Args:
        df: pd.DataFrame
        miss_pattern: str
            "MCAR", "MAR_logistic", "MNCR_gsm"
        miss_rate: float (0, 1)
            using miss_pattern == "MCAR"
        sm_k: float (0, 1)
            using miss_pattern == "gaussian-sm"
    '''

    set_seed(seed)
    miss_cols = sorted(miss_cols) # to guarantee reproducibility
    obs_cols = sorted( list( set(df.columns)  - set(miss_cols) - set(['target']) ) )
    
    output_df = df.copy()
    if miss_pattern == "MCAR":

        for i, miss_col in enumerate(miss_cols):
            output_df = missing_MCAR(df=output_df, miss_col_name=miss_col, miss_rate=miss_rate, 
                                     seed=seed + i)

    elif miss_pattern == "MAR_logistic":
        output_df[miss_cols] = missing_MAR_logistic(miss_data=output_df[miss_cols].values,
                                                    obs_data = output_df[obs_cols].values,
                                                    miss_rate=miss_rate, seed=seed)
    elif miss_pattern == "MNAR_gsm":
        assert sm_k is not None
        
        output_df[miss_cols] = missing_MNAR_selfmasking(data=output_df[miss_cols].values, 
                                           miss_rate=miss_rate, prop_latent=sm_prop_latent, 
                                           sm_type='gaussian', k=sm_k, lam=sm_lam, seed=seed) 

    else:
        raise NotImplementedError
    
    return output_df


def random_error_imputation(data, miss_index, mse, k=2, seed=42):
    '''
    data: np.array (n_samples, )
    miss_index: list
    mse: float
    k: float
        sigma = \sqrt{mse} / k
    seed: int
        random seed
    '''

    set_seed(seed)

    output_data = data.copy()
    n_miss = len(miss_index)


    # decide maen of noise
    mean = np.sqrt(mse)
    # decide var (sigma^2) of noise by k * \sigma = \sqrt{a} => \sigma^{2} = a / k^2
    var = mse / (k*k)

    # create noise following normal distributions
    pos_noise = np.random.normal(mean, var, size=n_miss)
    neg_noise = np.random.normal(-1*mean ,var, size=n_miss)
    
    # bernoulli samples for GMM
    select_flag = np.array(np.random.binomial(n=1, p=0.5, size=n_miss))


    output_data[miss_index] = output_data[miss_index].flatten() + select_flag * pos_noise + (1 - select_flag) * neg_noise

    return output_data


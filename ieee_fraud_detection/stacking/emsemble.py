# %%

# General imports
import numpy as np
import pandas as pd
import os
import gc
import warnings
import random
import datetime
from sklearn import metrics
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, kurtosis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import lightgbm as lgb
import time
warnings.filterwarnings('ignore')


# %%
data5 = pd.read_csv('kernel1_remodel10_2/submission10_ver2.csv')
data1 = pd.read_csv('stacking/submission_nn.csv')

# %%
# 算術平均
mean = data1['isFraud'] + data2['isFraud'] + data3['isFraud'] + data4['isFraud'] + data5['isFraud']
mean = mean / 5

# %%
mean_sub = data1.copy()
mean_sub['isFraud'] = mean

# %%
mean_sub.to_csv('random_seed_emsemble/mean_submission.csv', index=False)

# %%
from scipy.stats.mstats import gmean

# %%
gmean(np.array([[1, 2], 
                [3, 4]]), axis=1)

# %%
concat = np.concatenate([np.array(data1['isFraud']).reshape(-1, 1), 
                         np.array(data2['isFraud']).reshape(-1, 1), 
                         np.array(data3['isFraud']).reshape(-1, 1), 
                         np.array(data4['isFraud']).reshape(-1, 1), 
                         np.array(data5['isFraud']).reshape(-1, 1)], 1)

concat.shape
# %%
gmean_sub = data1.copy()
gmean_sub['isFraud'] = gmean(concat, axis=1)

# %%
gmean_sub.to_csv('random_seed_emsemble/gmean_submission.csv', index=False)

#%%
mean_sub.head(10)

#%%
from scipy.stats import rankdata


# %%
rank1 = rankdata(data1['bce_predict'].values).reshape(-1, 1)
rank2 = rankdata(data1['focal_predict'].values).reshape(-1, 1)
rank5 = rankdata(data5['isFraud'].values).reshape(-1, 1)

# %%
concat_rank = np.concatenate([rank1,
                              rank2,
                              rank5], 1)

np.mean(concat_rank, axis=1)

#%%
np.mean(concat_rank, axis=1).shape


#%%
from sklearn.preprocessing import minmax_scale

minmax_scale(np.mean(concat_rank, axis=1))

#%%
rank_sub = data5.copy()
rank_sub['isFraud'] = minmax_scale(np.mean(concat_rank, axis=1))

rank_sub.to_csv('random_seed_emsemble/nn_lgbm_rank_submission.csv', index=False)

#%%

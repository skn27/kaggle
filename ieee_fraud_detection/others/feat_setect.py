# %%

# General imports
import numpy as np
import pandas as pd
import os
import gc
import warnings
import random
import datetime
import xgboost as xgb

from sklearn import metrics
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, kurtosis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import lightgbm as lgb
import time
warnings.filterwarnings('ignore')

# %%
# Helpers
# Seeder
# :seed to make all processes deterministic     # type: int


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# %%
# Vars


SEED = 42
seed_everything(SEED)
LOCAL_TEST = True
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

# %%
# DATA LOAD
print('Load Data')
train_df = pd.read_pickle('data/train_df.pkl')

if LOCAL_TEST:
    
    # Convert TransactionDT to "Month" time-period. 
    # We will also drop penultimate block 
    # to "simulate" test set values difference
    # 擬似テストデータを作っていろいろやっている感じ
    test_df = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)
    train_df = train_df[train_df['DT_M'] < (train_df['DT_M'].max()-1)].reset_index(drop=True)
        
else:
    test_df = pd.read_pickle('data/test_df.pkl')
    
print('Shape control:', train_df.shape, test_df.shape)

# %%
# 定数の列を探す
one_value_cols = [col for col in train_df.columns if train_df[col].nunique() <= 1] #[]
one_value_cols_test = [col for col in test_df.columns if test_df[col].nunique() <= 1] #['V107']
one_value_cols == one_value_cols_test

# %%
# あらかじめ有用じゃないやつは(どのエントリでも同じとか、欠損がありすぎとか)))
many_null_cols = [col for col in train_df.columns if train_df[col].isnull().sum()/train_df.shape[0] > 0.9]
many_null_cols_test = [col for col in test_df.columns if test_df[col].isnull().sum()/test_df.shape[0] > 0.9]

# %%
# ほぼ全てのユーザーが共通の値を持っている変数(=特徴が出づらい変数)
big_top_value_cols = [col for col in train_df.columns
                     if train_df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test_df.columns
                          if test_df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove(TARGET)
len(cols_to_drop)

# %%
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

# %%
mod = xgb.XGBClassifier
select = RFE(XGBClassifier(learning_rate = 0.1,
                           n_estimators = 1000,
                           max_depth = 5,
                           min_child_weight = 1,
                           max_delta_step = 5,
                           gamma = 0,
                           subsample = 0.8,
                           colsample_bytree = 0.8,
                           objective = 'binary:logistic',
                           nthread = 4,
                           scale_pos_weight = 1,
                           seed =SEED
                           ), n_features_to_select=100)

# %%
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'uid','uid2','uid3',             # Our new client uID -> very noisy data
    'bank_type',                     # Victims bank could differ by time
    'DT','DT_M','DT_W','DT_D',       # Temporary Variables
    'DT_hour','DT_day_week','DT_day',
    'DT_D_total','DT_W_total','DT_M_total',
    'id_30','id_31','id_33',
    'shifted_uid_TransactionAmt', 'shifted_uid2_TransactionAmt',  # 追加
    'shifted_uid3_TransactionAmt', 'shifted_opposite_uid_TransactionAmt', # 追加
    'shifted_opposite_uid2_TransactionAmt', 'shifted_opposite_uid3_TransactionAmt', # 追加
    'isTrain' # 追加
]
final_feat = [col for col in train_df.columns if not col in rm_cols]
select.fit(train_df[final_feat], train_df[TARGET])

# %%
train_df.columns
# %%
# 欠損値を平均で補完or欠損値を中央値で補完、共通の相関が高いペアを削除する
temp_df = pd.concat([train_df, test_df])
cols = list(temp_df.columns)
temp_df.fillna(temp_df.mean(), inplace=True)
cm = np.corrcoef(temp_df.values.T)
high_corr_row_idx, high_corr_col_idx = np.where(0.85 <= cm)
high_corr_list_mean = []
for r_idx, c_idx in zip(high_corr_row_idx, high_corr_col_idx):
    if r_idx != c_idx:
        print(cols[r_idx], cols[c_idx])
        high_corr_list_mean.append([cols[r_idx], cols[c_idx]])
# %%
# 欠損値を平均で補完or欠損値を中央値で補完、共通の相関が高いペアを削除する
temp_df = pd.concat([train_df, test_df])
cols = list(temp_df.columns)
temp_df.fillna(temp_df.median(), inplace=True)
cm = np.corrcoef(temp_df.values.T)
high_corr_row_idx, high_corr_col_idx = np.where(0.85 <= cm)
high_corr_list_median = []
for r_idx, c_idx in zip(high_corr_row_idx, high_corr_col_idx):
    if r_idx != c_idx:
        print(cols[r_idx], cols[c_idx])
        high_corr_list_median.append([cols[r_idx], cols[c_idx]])# %%

# %%
high_corr_list = list(set(high_corr_list_mean) & set(high_corr_list_mean))
# %%
# あとは地道に可視化してどっちを残すか検証する
vis_df = temp_df[high_corr_list[0]]
 
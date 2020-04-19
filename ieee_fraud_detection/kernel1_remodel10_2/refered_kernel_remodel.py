# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again/data

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
from sklearn.model_selection import KFold

import lightgbm as lgb
import time
import optuna, uuid, read_pickle
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


# optunaで繰り返しやってチューニングしたい内容をここに入れる
def objectives(X_tr, y_tr, trial, nfolds=3):
    # 試行にUUIDを設定
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr('uuid', trial_uuid)

    # lgbmのチューニングしたいパラメータをここに
    params = {
        'objective':'binary',
        'boosting_type':'gbdt',
        'metric':'auc',
        'n_jobs':-1,
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 0.5),
        'num_leaves': trial.suggest_uniform('num_leaves', 10, 500),
        'max_depth':-1,
        'tree_learner':'serial',
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
        'subsample_freq':2,  # バギングの頻度
        'subsample':0.7,
        'n_estimators':800,
        'max_bin':255,
        'verbose':-1,
        'seed': SEED,
        'early_stopping_rounds':100, 
    }
    
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=SEED)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_tr, y_tr)):
        print('Fold: ', fold_)
        start = time.time()
        tr_x, tr_y = X_tr.iloc[trn_idx, :], y_tr[trn_idx]
        vl_x, vl_y = X_tr.iloc[val_idx, :], y_tr[val_idx]
        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=vl_data, 
            verbose_eval=200,
        )
        pp_p = estimator.predict(vl_x)
        predictions += pp_p/nfolds
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),
                                                X_tr.columns)),
                                    columns=['Value', 'Feature'])
        trial.set_user_attr('feat_imp', feature_imp['Feature'].iloc[:10])
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()




# %%
# Model(cv method→kfold)
# 時間表示が見えるようにすること #


def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X, y = tr_df[features_columns], tr_df[target]
    P, P_y = tt_df[features_columns], tt_df[target]

    tt_df = tt_df[['TransactionID', target]]
    predictions = np.zeros(len(tt_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:', fold_)
        start = time.time()
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            # こっちはtestデータを予測している
            vl_data = lgb.Dataset(P, label=P_y)
        else:
            # こっちはvalidationデータを予測している
            vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=[tr_data, vl_data],
            verbose_eval=200,
        )
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),
                                                  X.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
    tt_df['prediction'] = predictions
    return tt_df


# %%
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# %%
# Vars


SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
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
# PAST DATA LOAD
print('Load Past Data')
past_train_df = pd.read_pickle('data/train_transaction.pkl')

if LOCAL_TEST:
    
    # Convert TransactionDT to "Month" time-period. 
    # We will also drop penultimate block 
    # to "simulate" test set values difference
    # 擬似テストデータを作っていろいろやっている感じ
    past_train_df['DT_M'] = past_train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    past_train_df['DT_M'] = (past_train_df['DT_M'].dt.year-2017)*12 + past_train_df['DT_M'].dt.month 
    past_test_df = past_train_df[past_train_df['DT_M'] == past_train_df['DT_M'].max()].reset_index(drop=True)
    past_train_df = past_train_df[past_train_df['DT_M'] < (past_train_df['DT_M'].max()-1)].reset_index(drop=True)
    
    past_train_identity = pd.read_pickle('data/train_identity.pkl')
    past_test_identity = past_train_identity[past_train_identity['TransactionID'].isin(
                                    past_test_df['TransactionID'])].reset_index(drop=True)
    past_train_identity = past_train_identity[past_train_identity['TransactionID'].isin(
                                    past_train_df['TransactionID'])].reset_index(drop=True)
    del past_train_df['DT_M'], past_test_df['DT_M']
    
else:
    past_test_df = pd.read_pickle('data/test_transaction.pkl')
    past_train_identity = pd.read_pickle('data/train_identity.pkl')
    past_test_identity = pd.read_pickle('data/test_identity.pkl')
    
base_columns = list(past_train_df) + list(past_train_identity)
del past_test_df, past_test_identity, past_train_df, past_train_identity
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
########################### Model Features 
## We can use set().difference() but the order matters
## Matters only for deterministic results
## In case of remove() we will not change order
## even if variable will be renamed
## please see this link to see how set is ordered
## https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets
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


# %%
# Features elimination 
features_check = []
columns_to_check = set(list(train_df)).difference(base_columns+rm_cols)
for i in columns_to_check:
    features_check.append(ks_2samp(test_df[i], train_df[i])[1])

features_check = pd.Series(features_check, index=columns_to_check).sort_values() 
features_discard = list(features_check[features_check==0].index)
print(features_discard)

# We will reset this list for now (use local test drop),
# Good droping will be in other kernels
# with better checking
features_discard = [] 

# Final features list
features_columns = [col for col in list(train_df) if col not in rm_cols + features_discard]


# %%
########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# %%
########################### Model Train
if LOCAL_TEST:
    lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 20000
    lgb_params['early_stopping_rounds'] = 100
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    lgb_params['n_estimators'] = 1800
    lgb_params['early_stopping_rounds'] = 100    
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=10)

# %%
########################### Export
if not LOCAL_TEST:
    test_predictions['isFraud'] = test_predictions['prediction']
    test_predictions[['TransactionID','isFraud']].to_csv('kernel1_remodel10_2/submission10_ver2.csv', index=False)

# %%
## LOCAL_TEST = True
# auc→0.9230216935319279(ver7を追加))(今までのmax→0.9211773468405661)
# auc→0.9211773468405661(ver4)
# feat_imp
# 606   1412                   card1_id_02_skew
# 607   1420                              id_20
# 608   1469                                 D4
# 609   1472  uid3_TransactionAmt_quantile_0.25
# 610   1475           uid3_id_02_quantile_0.75
# 611   1500                    uid3_id_02_mean
# 612   1530                      uid3_D15_mean
# 613   1533           uid3_TransactionAmt_mean
# 614   1562              uid3_D15_quantile_0.5
# 615   1572               P_emaildomain_fq_enc
# 616   1601            uid3_TransactionAmt_std
# 617   1628                                 D2
# 618   1685                        uid3_fq_enc
# 619   1717            uid3_id_02_quantile_0.5
# 620   1719                       addr1_fq_enc
# 621   1723                           uid_DT_M
# 622   1738                                C13
# 623   1868                     uid3_id_02_iqr
# 624   1897                              dist1
# 625   1914                              id_02
# 626   2026                              card1
# 627   2056                                D15
# 628   2119           uid3_id_02_quantile_0.25
# 629   2182                              addr1
# 630   2193                       uid3_D15_std
# 631   2399                     uid3_id_02_std
# 632   2528                    uid3_id_02_skew
# 633   2577                           uid_DT_W
# 634   3121                           uid_DT_D
# 635   4601                     TransactionAmt

# 
## LOCAL_TEST = False
# max auc(lb)→0.9485(feature_eliminateした場合)
# max auc(cvの)→0.976896



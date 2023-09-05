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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, kurtosis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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

def make_predictions_stacking(tr_df, tt_df, features_columns, target, lgb_params, n_folds):

    X, y = tr_df[features_columns], tr_df[target]
    P, P_y = tt_df[features_columns], tt_df[target]

    tt_df = tt_df[['TransactionID', target]]
    predictions = np.zeros(len(tt_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:', fold_)
        start = time.time()
        tr_idx = np.load('data/holdout%d_tr.npy'%fold_)
        val_idx = np.load('data/holdout%d_val.npy'%fold_)
        tr_x, tr_y = X.iloc[tr_idx, :], y[tr_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
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
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
    tt_df['prediction'] = predictions
    return tt_df

# %%

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        start = time.time()
        # folds = list(StratifiedKFold(n_splits=self.n_splits,
        #                              shuffle=True,
        #                              random_state=SEED).split(X, y))
        folds = list(KFold(n_splits=self.n_splits,
                           shuffle=True,
                           random_state=SEED).split(X, y))        
        print("folds_elapsed_time:{0}".format(time.time()-start) + "[sec]")

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            start = time.time()
            print('Fold:%d starts at' % i, datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                print("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train,
                        eval_set=[(X_holdout, y_holdout)],
                        eval_metric='auc',
                        early_stopping_rounds=100,
                        verbose=-1,
                        )
                cross_score = cross_val_score(clf, X_train, y_train,
                                              cv=3, scoring='roc_auc')
                print("cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:, 1]       

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            S_test[:, i] = S_test_i.mean(axis=1)
        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res
# %%
# Vars


# SEED = 42
# seed_everything(SEED)
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
    'TransactionID', 'TransactionDT',  # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'uid', 'uid2', 'uid3',             # Our new client uID -> very noisy data
    'bank_type',                     # Victims bank could differ by time
    'DT', 'DT_M', 'DT_W', 'DT_D',       # Temporary Variables
    'DT_hour', 'DT_day_week', 'DT_day',
    'DT_D_total', 'DT_W_total', 'DT_M_total',
    'id_30', 'id_31', 'id_33',
    'shifted_uid_TransactionAmt', 'shifted_uid2_TransactionAmt',  # 追加
    'shifted_uid3_TransactionAmt', 'shifted_opposite_uid_TransactionAmt',  # 追加
    'shifted_opposite_uid2_TransactionAmt', 'shifted_opposite_uid3_TransactionAmt',  # 追加
    'isTrain'  # 追加
]
features_columns = [col for col in list(train_df) if col not in rm_cols]


# %%


########################### Model params
SEED = 42

lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.005,
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
lgb_params2 = {
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate': 0.009,
                    'num_leaves': 2**7,
                    'max_depth': 13,
                    'colsample_bytree': 0.9,
                    'n_estimators':20000,
                    'min_child_samples': 79,
                    "subsample": 0.9,
                    "subsample_freq": 3,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3,
                    'ranodm_state': SEED+27,
                }
lgb_params3 = {
                    'boosting_type': 'gbdt',
                    'num_leaves': 2**6,             
                    'max_depth': 16,
                    'learning_rate': 0.03,
                    'is_unbalance': 'true',
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.01,
                    'min_child_weight': 3,
                    'SEED': SEED-27,
                    'n_jobs':-1
                }
# %%
lgb_params['learning_rate'] = 0.005
lgb_params['n_estimators'] = 1800
lgb_params['early_stopping_rounds'] = 100    
test_predictions = make_predictions_stacking(train_df, test_df, features_columns, TARGET, lgb_params, 5)


# %%
########################### Export
test_predictions['isFraud'] = test_predictions['prediction']
test_predictions[['TransactionID','isFraud']].to_csv('stacking/lgbm1_.csv', index=False)

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



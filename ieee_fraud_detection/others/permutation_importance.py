# %%[markdown]
# 参考にした カーネル[https://www.kaggle.com/artgor/eda-and-models]


# %%
import numpy as np
import scipy as sp
import pandas as pd
import os
from numba import jit

import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook  # プログレスバーを出す
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR # νを使って使うサポートベクターの数を制御している。
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
from collections import defaultdict
import gc #ガベージコレクションをするためのモジュール。
# ガベージコレクション→不要になったメモリ領域を自動的に解放する機能
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# permutation importanceはmodelにfitし終わったら計算するもの。ある特徴だけ順番をシャッフルしてモデルの精度が
# 落ちるかチェック。落ちれば落ちるほど、現実のデータとの乖離が分かり、重要であるという指標。

# import shap # それぞれの特徴変数がその予測にどのような影響を与えたかを算出するもの(要因分析にも使えるのでは)
from IPython.display import HTML
import json
import altair as alt #可視化ツールの一つ、簡単らしい
from altair.vega import v5

import networkx as nx #グラフ分析用のライブラリ

# alt.renderers.enable('notebook')

# %%
def reduce_mem_usage(df, verbose=True):
    # メモリ容量を減らす関数
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
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
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def id_split(dataframe):
    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]

    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]
    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]

    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]
    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]

    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0].astype(float)
    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1].astype(float)

    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]
    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()

    return dataframe

def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    splits = folds.split(X) if splits is None else splits
    n_splits = folds.n_splits if splits is None else n_folds
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                     'mse': {'lgb_metric_name': 'mse',
                             'catboost_metric_name': 'MSE',
                             'sklearn_scoring_function': metrics.mean_squared_error}
                    }
    result_dict = {}
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(splits):
        if verbose:
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000,
                              evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns),
                                   ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1,)

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1):
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))


    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                       loss_function=Logloss)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if averaging == 'usual':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':

            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            fig = plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols

    return result_dict

# %%
# transaction的な情報
train_transaction = pd.read_csv('data/train_transaction.csv')
# transactionについて補足的な情報
train_identity = pd.read_csv('data/train_identity.csv')
# transaction的な情報
test_transaction = pd.read_csv('data/test_transaction.csv')
# transactionについて補足的な情報
test_identity = pd.read_csv('data/test_identity.csv')
sub = pd.read_csv('data/sample_submission.csv')
# combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

del train_identity, train_transaction, test_identity, test_transaction

# %%
# 定数の列を探す
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1] #[]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1] #['V107']
one_value_cols == one_value_cols_test

# %%
# feature enginiering、大事そうなカテゴリ変数を基準に統計量算出
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')

# %%
# feature enginiering、大事そうなカテゴリ変数を基準に統計量算出
train_TA_s_card1 = train.groupby(['card1'])['TransactionAmt'].apply(sp.stats.skew).rename('TransactionAmt_to_skew_card1')
train_TA_s_card4 = train.groupby(['card4'])['TransactionAmt'].apply(sp.stats.skew).rename('TransactionAmt_to_skew_card4')
train_TA_k_card1 = train.groupby(['card1'])['TransactionAmt'].apply(sp.stats.kurtosis).rename('TransactionAmt_to_kurtosis_card1')
train_TA_k_card4 = train.groupby(['card4'])['TransactionAmt'].apply(sp.stats.kurtosis).rename('TransactionAmt_to_kurtosis_card4')
tmp_train = pd.merge(train, train_TA_s_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_TA_s_card4, on='card4', how='left')
tmp_train = pd.merge(tmp_train, train_TA_k_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_TA_k_card4, on='card4', how='left')
del train_TA_s_card1, train_TA_s_card4, train_TA_k_card1, train_TA_k_card4

test_TA_s_card1 = test.groupby(['card1'])['TransactionAmt'].apply(sp.stats.skew).rename('TransactionAmt_to_skew_card1')
test_TA_s_card4 = test.groupby(['card4'])['TransactionAmt'].apply(sp.stats.skew).rename('TransactionAmt_to_skew_card4')
test_TA_k_card1 = test.groupby(['card1'])['TransactionAmt'].apply(sp.stats.kurtosis).rename('TransactionAmt_to_kurtosis_card1')
test_TA_k_card4 = test.groupby(['card4'])['TransactionAmt'].apply(sp.stats.kurtosis).rename('TransactionAmt_to_kurtosis_card4')
tmp_test = pd.merge(test, test_TA_s_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_TA_s_card4, on='card4', how='left')
tmp_test = pd.merge(tmp_test, test_TA_k_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_TA_k_card4, on='card4', how='left')
del test_TA_s_card1, test_TA_s_card4, test_TA_k_card1, test_TA_k_card4

train_id_02_s_card1 = train.groupby(['card1'])['id_02'].apply(sp.stats.skew).rename('id_02_to_skew_card1')
train_id_02_s_card4 = train.groupby(['card4'])['id_02'].apply(sp.stats.skew).rename('id_02_to_skew_card4')
train_id_02_k_card1 = train.groupby(['card1'])['id_02'].apply(sp.stats.kurtosis).rename('id_02_to_kurtosis_card1')
train_id_02_k_card4 = train.groupby(['card4'])['id_02'].apply(sp.stats.kurtosis).rename('id_02_to_kurtosis_card4')
tmp_train = pd.merge(tmp_train, train_id_02_s_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_id_02_s_card4, on='card4', how='left')
tmp_train = pd.merge(tmp_train, train_id_02_k_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_id_02_k_card4, on='card4', how='left')
del train_id_02_s_card1, train_id_02_s_card4, train_id_02_k_card1, train_id_02_k_card4

test_id_02_s_card1 = test.groupby(['card1'])['id_02'].apply(sp.stats.skew).rename('id_02_to_skew_card1')
test_id_02_s_card4 = test.groupby(['card4'])['id_02'].apply(sp.stats.skew).rename('id_02_to_skew_card4')
test_id_02_k_card1 = test.groupby(['card1'])['id_02'].apply(sp.stats.kurtosis).rename('id_02_to_kurtosis_card1')
test_id_02_k_card4 = test.groupby(['card4'])['id_02'].apply(sp.stats.kurtosis).rename('id_02_to_kurtosis_card4')
tmp_test = pd.merge(tmp_test, test_id_02_s_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_id_02_s_card4, on='card4', how='left')
tmp_test = pd.merge(tmp_test, test_id_02_k_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_id_02_k_card4, on='card4', how='left')
del test_id_02_s_card1, test_id_02_s_card4, test_id_02_k_card1, test_id_02_k_card4

train_D15_s_card1 = train.groupby(['card1'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_card1')
train_D15_s_card4 = train.groupby(['card4'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_card4')
train_D15_k_card1 = train.groupby(['card1'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_card1')
train_D15_k_card4 = train.groupby(['card4'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_card4')
tmp_train = pd.merge(tmp_train, train_D15_s_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_D15_s_card4, on='card4', how='left')
tmp_train = pd.merge(tmp_train, train_D15_k_card1, on='card1', how='left')
tmp_train = pd.merge(tmp_train, train_D15_k_card4, on='card4', how='left')
del train_D15_s_card1, train_D15_s_card4, train_D15_k_card1, train_D15_k_card4

test_D15_s_card1 = test.groupby(['card1'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_card1')
test_D15_s_card4 = test.groupby(['card4'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_card4')
test_D15_k_card1 = test.groupby(['card1'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_card1')
test_D15_k_card4 = test.groupby(['card4'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_card4')
tmp_test = pd.merge(tmp_test, test_D15_s_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_D15_s_card4, on='card4', how='left')
tmp_test = pd.merge(tmp_test, test_D15_k_card1, on='card1', how='left')
tmp_test = pd.merge(tmp_test, test_D15_k_card4, on='card4', how='left')
del test_D15_s_card1, test_D15_s_card4, test_D15_k_card1, test_D15_k_card4

train_D15_s_addr1 = train.groupby(['addr1'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_addr1')
train_D15_s_addr2 = train.groupby(['addr2'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_addr2')
train_D15_k_addr1 = train.groupby(['addr1'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_addr1')
train_D15_k_addr2 = train.groupby(['addr2'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_addr2')
tmp_train = pd.merge(tmp_train, train_D15_s_addr1, on='addr1', how='left')
tmp_train = pd.merge(tmp_train, train_D15_s_addr2, on='addr2', how='left')
tmp_train = pd.merge(tmp_train, train_D15_k_addr1, on='addr1', how='left')
tmp_train = pd.merge(tmp_train, train_D15_k_addr2, on='addr2', how='left')
del train_D15_s_addr1, train_D15_s_addr2, train_D15_k_addr1, train_D15_k_addr2

test_D15_s_addr1 = test.groupby(['addr1'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_addr1')
test_D15_s_addr2 = test.groupby(['addr2'])['D15'].apply(sp.stats.skew).rename('D15_to_skew_addr2')
test_D15_k_addr1 = test.groupby(['addr1'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_addr1')
test_D15_k_addr2 = test.groupby(['addr2'])['D15'].apply(sp.stats.kurtosis).rename('D15_to_kurtosis_addr2')
tmp_test = pd.merge(tmp_test, test_D15_s_addr1, on='addr1', how='left')
tmp_test = pd.merge(tmp_test, test_D15_s_addr2, on='addr2', how='left')
tmp_test = pd.merge(tmp_test, test_D15_k_addr1, on='addr1', how='left')
tmp_test = pd.merge(tmp_test, test_D15_k_addr2, on='addr2', how='left')
del test_D15_s_addr1, test_D15_s_addr2, test_D15_k_addr1, test_D15_k_addr2

# %%
train = tmp_train.copy()
test = tmp_test.copy()
del tmp_train, tmp_test

# %%
train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)

# %%

train = id_split(train)
test = id_split(test)

# %%
many_null_cols = [col for col in train.columns if train[col].isnull().sum()/train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum()/test.shape[0] > 0.9]

# %%
# ほぼ全てのユーザーが共通の値を持っている変数(=特徴が出づらい変数)
big_top_value_cols = [col for col in train.columns
                     if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns
                          if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove('isFraud')
len(cols_to_drop)

# %%
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# %%
# カテゴリ変数なら欠損値を新たなカテゴリに

cate_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
             'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31',
             'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo',
             'ProductCD', 'card4', 'card6', 'M4','P_emaildomain', 'R_emaildomain', 'card1', 'card2',
             'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
             'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2',
             'R_emaildomain_3', 'device_name', 'OS_id_30', 'version_id_30', 'browser_id_31', 'version_id_31']

categorical = []
print(len(cate_cols))
for col in cate_cols:
    if col in train.columns:
        categorical.append(col)
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(train[col].astype(str).values)
        test[col] = le.transform(test[col].astype(str).values)
print(len(categorical))


# %%
# 連続値変数の欠損値を補完(もし欠損の数を特徴に入れるならこれより前でやる)、lgbmでpimp計算するからとりあえずスルー

# train.fillna(-999, inplace=True)
# test.fillna(-999, inplace=True)

# %%
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'],
                                            axis = 1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
del train
test = test[['TransactionDT', 'TransactionID']]

# %%
gc.collect()

# %%

def permuted(df):
    """特定のカラムをシャッフルしたデータフレームを返す"""
    for column_name in df.columns:
        permuted_df = df.copy()
        permuted_df[column_name] = np.random.permutation(permuted_df[column_name])
        yield column_name, permuted_df


def pimp(clf, X, y, n_fold, eval_func='auc', cv=None, columns=None):
    """PIMP (Permutation IMPortance) を計算する"""
    base_scores = []
    permuted_scores = defaultdict(list)
    columns = X.columns if columns is None else columns

    if cv is None:
        # 今回は時系列だからこの分け方で
        folds = TimeSeriesSplit(n_splits=n_fold)
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=eval_auc,
                verbose=10000, early_stopping_rounds=200)
        # まずは何もシャッフルしていないときのスコアを計算する
        y_pred_valid = clf.predict_proba(X_valid)[:, 1]
        if eval_func == 'auc':
            base_score = fast_auc(y_valid, y_pred_valid)
            base_scores.append(base_score)

        # 特定のカラムをシャッフルした状態で推論したときのスコアを計算する
        permuted_X_valid_gen = permuted(X_valid)
        for column_name, permuted_X_valid in permuted_X_valid_gen:
            y_pred_permuted = clf.predict_proba(permuted_X_valid)[:, 1]
            if eval_func == 'auc':
                permuted_score = fast_auc(y_valid, y_pred_permuted)
                permuted_scores[column_name].append(permuted_score)

    # 基本のスコアとシャッフルしたときのスコアを返す
    np_base_score = np.array(base_scores)
    dict_permuted_score = {name: np.array(scores) for name, scores in permuted_scores.items()}
    return np_base_score, dict_permuted_score

def score_difference_statistics(base, permuted):
    """シャッフルしたときのスコアに関する統計量 (平均・標準偏差) を返す"""
    mean_base_score = base.mean()
    for column_name, scores in permuted.items():
        score_differences = scores - mean_base_score
        yield column_name, score_differences.mean(), score_differences.std()

# %%

# permutation importanceの計算を実行(もしかしたらtrainとtestで比較する必要があるかも→
# 必要なしtestの正解ラベルわからん)
# いじるのはlearning_rateだけ、上げれば早く終わる。(デフォルトは0.03)

n_fold = 2
params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.3,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
         }



model = lgb.LGBMClassifier(**params, n_estimators=50000, n_jobs=-1)
base_score, permuted_scores = pimp(clf=model, X=X, y=y, n_fold=n_fold, eval_func='auc', cv=None, columns=None)

diff_stats = list(score_difference_statistics(base_score, permuted_scores))

# カラム名、ベーススコアとの差、95% 信頼区間を取り出す
sorted_diff_stats = sorted(diff_stats, key=lambda x: x[1])
column_names = [name for name, _, _ in sorted_diff_stats]
diff_means = [diff_mean for _, diff_mean, _ in sorted_diff_stats]
diff_stds_95 = [diff_std * 1.96 for _, _, diff_std in sorted_diff_stats]


# %%

# シャッフルしてスコアあげているやつは除外するべきでは...
# 正のものだけ取り出すようにする。
column_names = np.array(column_names)
diff_means = np.array(diff_means)

print(column_names[diff_means < 0])

# %%
def pimp_specified_cate(X, y, n_fold, categorical, eval_func='auc', cv=None, columns=None):
    """PIMP (Permutation IMPortance) を計算する"""
    base_scores = []
    permuted_scores = defaultdict(list)
    columns = X.columns if columns is None else columns

    params = {'num_leaves': 256,
              'min_child_samples': 79,
              'objective': 'binary',
              'max_depth': 13,
              'learning_rate': 0.3,
              "boosting_type": "gbdt",
              "subsample_freq": 3,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'auc',
              "verbosity": -1,
              'reg_alpha': 0.3,
              'reg_lambda': 0.3,
              'colsample_bytree': 0.9
              }

    if cv is None:
        # 今回は時系列だからこの分け方で
        folds = TimeSeriesSplit(n_splits=n_fold)
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            lgtrain = lgb.Dataset(X[columns][train_index], 
                                  label=y[train_index], 
                                  categorical_feature=categorical, 
                                  feature_name='auto')
            X_valid = X[columns][valid_index]
            y_valid = y[valid_index]
            lgvalid = lgb.Dataset(X[columns][valid_index], 
                                  label=y[valid_index], 
                                  categorical_feature=categorical, 
                                  feature_name='auto')

        else:
            lgtrain = lgb.Dataset(X[columns].iloc[train_index], 
                                  label=y.iloc[train_index], 
                                  categorical_feature=categorical, 
                                  feature_name='auto')
            X_valid = X[columns].iloc[valid_index]
            y_valid = y.iloc[valid_index]
            lgvalid = lgb.Dataset(X[columns].iloc[valid_index], 
                                  label=y.iloc[valid_index], 
                                  categorical_feature=categorical, 
                                  feature_name='auto')
        
        gbm = lgb.train(params, 
                        lgtrain,
                        valid_sets = lgvalid,
                        num_boost_round=50000,
                        verbose_eval=10000,
                        early_stopping_rounds=200
                        )
        # evals_resultのparamを入れれば、おそらくvalidの結果は取り出せる

        # まずは何もシャッフルしていないときのスコアを計算する
        
        y_pred_valid = gbm.predict(X_valid)
        if eval_func == 'auc':
            base_score = fast_auc(y_valid, y_pred_valid)
            base_scores.append(base_score)

        # 特定のカラムをシャッフルした状態で推論したときのスコアを計算する
        permuted_X_valid_gen = permuted(X_valid)
        for column_name, permuted_X_valid in permuted_X_valid_gen:
            y_pred_permuted = gbm.predict(permuted_X_valid)
            if eval_func == 'auc':
                permuted_score = fast_auc(y_valid, y_pred_permuted)
                permuted_scores[column_name].append(permuted_score)

    # 基本のスコアとシャッフルしたときのスコアを返す
    np_base_score = np.array(base_scores)
    dict_permuted_score = {name: np.array(scores) for name, scores in permuted_scores.items()}
    return np_base_score, dict_permuted_score


# %%
print('id_22' in categorical, 'id_22' in X.columns)

# %%
# lgb.Datasetを使う場合のpimpの求めかた
n_fold = 2

base_score, permuted_scores = pimp_specified_cate(X=X, y=y, n_fold=n_fold, 
                                                  categorical=categorical, 
                                                  eval_func='auc', cv=None, columns=None)

diff_stats = list(score_difference_statistics(base_score, permuted_scores))

# カラム名、ベーススコアとの差、95% 信頼区間を取り出す
sorted_diff_stats = sorted(diff_stats, key=lambda x: x[1])
column_names = [name for name, _, _ in sorted_diff_stats]
diff_means = [diff_mean for _, diff_mean, _ in sorted_diff_stats]
diff_stds_95 = [diff_std * 1.96 for _, _, diff_std in sorted_diff_stats]


# %%
# 最後の学習
# categorical変数をラベルエンコードしている
n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)

params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds,
                                             model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                             verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual')

# %%
sub['isFraud'] = result_dict_lgb['prediction']
sub.to_csv('submission.csv', index=False)

# %%[markdown]
# 入れ替えた結果スコアが下がる、つまりpermutation_importanceが高いやつ
# 学習率が0.03、特徴は上記、カテゴリ変数指定なし
# ['C1' 'C13' 'M4' 'card6' 'C14' 'D2' 'card1' 'C6'
#  'TransactionAmt_to_kurtosis_card1' 'addr1' 'C11' 'TransactionAmt' 'V70'
#  'card2' 'TransactionAmt_to_skew_card1' 'M5' 'M6' 'C2' 'card5' 'D4' 'V294'
#  'C9' 'C5' 'TransactionAmt_to_mean_card4' 'TransactionAmt_to_std_card4'
#  'ProductCD' 'C8' 'V258' 'card3' 'V54' 'P_emaildomain'
#  'TransactionAmt_to_std_card1' 'dist1' 'D11' 'D3' 'D15_to_mean_addr1'
#  'V62' 'D15_to_mean_card4' 'D1' 'V53' 'D15_to_std_card1'
#  'D15_to_mean_card1' 'TransactionAmt_to_mean_card1' 'P_emaildomain_2' 'D8'
#  'V283' 'D10' 'V317' 'V12' 'D5' 'V312' 'V308' 'screen_height' 'V61' 'V131'
#  'id_06' 'M3' 'V82' 'R_emaildomain_2' 'TransactionAmt_to_skew_card4'
#  'id_31' 'C10' 'V30' 'V310' 'V87' 'D15_to_mean_addr2' 'V49' 'V38' 'D15'
#  'V126' 'V45' 'M7' 'id_20' 'V165' 'D15_to_std_card4' 'V56' 'M9'
#  'D15_to_std_addr1' 'V7' 'V78' 'V261' 'V127' 'V35' 'V289' 'V75' 'V128'
#  'V262' 'card4' 'V189' 'V200' 'V187' 'V2' 'D14' 'V243' 'V71' 'V13' 'V91'
#  'V37' 'id_30' 'C12' 'V92' 'V96' 'V83' 'V277' 'V20' 'V267' 'V172' 'V160'
#  'V36' 'V39' 'D6' 'version_id_30' 'V76' 'V3' 'V64' 'V6' 'V47' 'V287'
#  'screen_width' 'V221' 'V178' 'V74' 'V291' 'D12' 'V81' 'V69' 'M8' 'V90'
#  'C7' 'V99' 'browser_id_31' 'V336' 'D13' 'id_04' 'V19' 'id_09' 'V234'
#  'V29' 'V97' 'V40' 'V303' 'V166' 'V274' 'V152' 'V164' 'DeviceInfo' 'V33'
#  'V51' 'V207' 'V94' 'V144' 'V85' 'V72' 'M2' 'V282' 'V280' 'V63' 'C4'
#  'V292' 'V46' 'V50' 'V326' 'id_36' 'V57' 'V285' 'V253'
#  'id_02_to_mean_card1' 'V239' 'V139' 'V245'
#  'TransactionAmt_to_kurtosis_card4' 'V337' 'V141' 'V95' 'V170' 'V4' 'V181'
#  'V306' 'V192' 'V264' 'V73' 'V60' 'V214' 'V142' 'V288' 'addr2' 'V219'
#  'V191' 'D9' 'V185' 'V257' 'V230' 'V148' 'id_02_to_mean_card4' 'V10'
#  'V130' 'V84' 'V225' 'V183' 'V218' 'V193' 'V237' 'V18' 'V229' 'V273' 'M1'
#  'V145' 'V179' 'V176' 'V188' 'V9' 'V22' 'id_12' 'id_34' 'V246' 'V233'
#  'V153' 'V21' 'V328' 'id_10' 'id_35' 'DeviceType' 'V196' 'V146' 'V333'
#  'V174' 'V302' 'V147' 'V248' 'V256' 'id_02_to_std_card4' 'V154' 'V16'
#  'V304' 'V236' 'V44' 'V247' 'V1' 'V41' 'V194' 'V195' 'V240' 'V241' 'V269'
#  'V325' 'id_29']
# この時のスコアが0.920702

# ---
# 学習率が0.3、特徴は上記、カテゴリ変数指定なし
# この時のスコアが0.898538
# ['C13' 'TransactionAmt_to_skew_card1' 'M4' 'C1' 'C5'
# 'TransactionAmt_to_kurtosis_card1' 'TransactionAmt' 'card1' 'C14' 'addr1'
# 'D2' 'card2' 'card6' 'V70' 'TransactionAmt_to_std_card4' 'C11' 'D4'
#  'card5' 'V294' 'C2' 'TransactionAmt_to_mean_card4' 'M5' 'ProductCD'
#  'dist1' 'D15_to_mean_addr2' 'M6' 'D1' 'P_emaildomain' 'D15_to_mean_addr1'
#  'V62' 'V283' 'C6' 'card3' 'D15_to_mean_card4' 'C8' 'V54' 'D15'
#  'D15_to_mean_card1' 'TransactionAmt_to_std_card1' 'D5' 'D8' 'V131' 'V312'
#  'id_38' 'D3' 'V82' 'M7' 'M3' 'V30' 'R_emaildomain' 'V49' 'D14' 'V91'
#  'D11' 'V11' 'V313' 'TransactionAmt_to_mean_card1' 'V13' 'V45'
#  'D15_to_std_card1' 'C10' 'V308' 'V165' 'C9' 'id_31' 'V307' 'V12' 'V75'
#  'V83' 'V53' 'V244' 'D15_to_std_card4' 'V36' 'M8' 'V20' 'V94'
#  'TransactionAmt_to_skew_card4' 'id_02_to_mean_card4' 'V19' 'V258' 'V149'
#  'V85' 'id_17' 'V280' 'screen_height' 'V126' 'card4' 'C4'
#  'P_emaildomain_2' 'V78' 'V37' 'V5' 'V317' 'id_19' 'V17' 'V285' 'id_01'
#  'V314' 'id_02_to_mean_card1' 'M2' 'V274' 'V262' 'V61' 'id_09' 'V127'
#  'V287' 'V170' 'D9' 'V187' 'id_13' 'C7' 'V63' 'V92' 'V44' 'V292' 'V242'
#  'V38' 'V79' 'V74' 'version_id_31' 'V58' 'browser_id_31' 'V303'
#  'R_emaildomain_2' 'screen_width' 'V96' 'D12' 'C12' 'V52' 'V76' 'V335'
#  'V267' 'R_emaildomain_1' 'id_05' 'V276' 'id_02_to_std_card1' 'V211'
#  'V243' 'V264' 'V200' 'V266' 'V156' 'V34' 'V271' 'id_30' 'V277' 'V35'
#  'V181' 'D15_to_std_addr1' 'V3' 'id_33' 'V73' 'V239' 'id_04' 'V224' 'V251'
#  'M9' 'V161' 'id_14' 'V43' 'V204' 'V87' 'V278' 'V72' 'V304' 'V9' 'V80'
#  'id_11' 'V245' 'V288' 'V210' 'V40' 'V289' 'V207' 'V166' 'V47' 'V337' 'V8'
#  'V230' 'V56' 'V147' 'V257' 'id_34' 'V142' 'V253' 'V51' 'V18' 'V329'
#  'V223' 'V140' 'V81' 'V143' 'addr2' 'V151' 'V146' 'DeviceType' 'V180'
#  'V59' 'V21' 'V50' 'V69' 'V192' 'id_32' 'V97' 'OS_id_30' 'V178' 'id_16'
#  'V155' 'V272' 'V141' 'V183' 'V148' 'id_10' 'V222' 'id_12' 'V273' 'V175'
#  'V324' 'V218' 'V252' 'V219' 'V150' 'V302' 'V130' 'V154' 'V331' 'V57'
#  'V196' 'V322' 'V153' 'V16' 'V255' 'V334' 'id_36' 'V197' 'V193']

# %%
gbm


# %%
tmp
# %%

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
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, kurtosis


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
# Model(cv method→groupkfold)
# def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    
#     folds = GroupKFold(n_splits=NFOLDS)

#     X, y = tr_df[features_columns], tr_df[target]    
#     P, P_y = tt_df[features_columns], tt_df[target]  
#     split_groups = tr_df['DT_M']

#     tt_df = tt_df[['TransactionID', target]]    
#     predictions = np.zeros(len(tt_df))
#     oof = np.zeros(len(tr_df))
    
#     for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
#         print('Fold:', fold_)
#         tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#         vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
            
#         print(len(tr_x), len(vl_x))
#         tr_data = lgb.Dataset(tr_x, label=tr_y)
#         if LOCAL_TEST:
#             # こっちはtestデータを予測している
#             vl_data = lgb.Dataset(P, label=P_y)
#         else:
#             # こっちはvalidationデータを予測している
#             vl_data = lgb.Dataset(vl_x, label=vl_y)

#         estimator = lgb.train(
#             lgb_params,
#             tr_data,
#             valid_sets = [tr_data, vl_data],
#             verbose_eval = 200,
#         )   
        
#         pp_p = estimator.predict(P)
#         predictions += pp_p/NFOLDS
        
#         oof_preds = estimator.predict(vl_x)
#         oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

#         if LOCAL_TEST:
#             feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),
#                                                   X.columns)), columns=['Value', 'Feature'])
#             print(feature_imp)
        
#         del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
#         gc.collect()
        
#     tt_df['prediction'] = predictions
#     # OOF→out-of-fold、いわゆるcvでのvalに回っているデータのこと(foldの外)
#     print('OOF AUC:', metrics.roc_auc_score(y, oof))
#     if LOCAL_TEST:
#         print('Holdout AUC:', metrics.roc_auc_score(tt_df[TARGET], tt_df['prediction']))
    
#     return tt_df


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
train_df = pd.read_pickle('data/train_transaction.pkl')

if LOCAL_TEST:
    
    # Convert TransactionDT to "Month" time-period. 
    # We will also drop penultimate block 
    # to "simulate" test set values difference
    # 擬似テストデータを作っていろいろやっている感じ
    train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
    test_df = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)
    train_df = train_df[train_df['DT_M'] < (train_df['DT_M'].max()-1)].reset_index(drop=True)
    
    train_identity = pd.read_pickle('data/train_identity.pkl')
    test_identity = train_identity[train_identity['TransactionID'].isin(
                                    test_df['TransactionID'])].reset_index(drop=True)
    train_identity = train_identity[train_identity['TransactionID'].isin(
                                    train_df['TransactionID'])].reset_index(drop=True)
    del train_df['DT_M'], test_df['DT_M']
    
else:
    test_df = pd.read_pickle('data/test_transaction.pkl')
    train_identity = pd.read_pickle('data/train_identity.pkl')
    test_identity = pd.read_pickle('data/test_identity.pkl')
    
base_columns = list(train_df) + list(train_identity)
print('Shape control:', train_df.shape, test_df.shape)

# %%
# D9 and TransactionDT
# Let's add temporary "time variables" for aggregations
# and add normal "time variables"

# Also, seems that D9 column is an hour
# and it is the same as df['DT'].dt.hour
# TransactionDTから時間のデータをカラムに追加
# D9についてはNaNを0、その他を1としている。
for df in [train_df, test_df]:
    # Temporary
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear
    df['DT_hour'] = df['DT'].dt.hour
    df['DT_day_week'] = df['DT'].dt.dayofweek
    df['DT_day'] = df['DT'].dt.day
    # D9 column
    df['D9'] = np.where(df['D9'].isna(), 0, 1)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #401
# %%
# Reset values for "noise" card1
i_cols = ['card1']

for col in i_cols:
    valid_card = pd.concat([train_df[[col]], test_df[[col]]])
    valid_card = valid_card[col].value_counts()
    valid_card = valid_card[valid_card > 2]
    valid_card = list(valid_card.index)

    train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
    test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

    train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
    test_df[col] = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #401

# %%
# M columns (except M4)
# All these columns are binary encoded 1/0
# We can have some features from it
i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

for df in [train_df, test_df]:
    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #403
# %%
# ProductCD and M4 Target mean
# ここのtarget_meanの実装をいじってみる
# 参考→https://www.rco.recruit.co.jp/career/engineer/blog/kaggle_talkingdata_basic/
# ここに効きそうなカテゴリデータを入れてみる(1,2個)
for col in ['ProductCD', 'M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                             columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean'] = test_df[col].map(temp_dict)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #405

# %%
########################### TransactionAmt

# Let's add some kind of client uID based on cardID ad addr columns
# The value will be very specific for each client so we need to remove it
# from final feature. But we can use it for aggregations.
train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)
test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)

train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

# Check if the Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with a model we are telling to trust or not to these values   
train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check'] = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #409

# %%
# fraudかどうかで平均を取り、その差分を特徴にする(ver5)
temp_df = train_df.groupby('isFraud')['TransactionAmt'].agg('mean').reset_index()
train_df['diff_is_not_fraud'] = np.abs(train_df['TransactionAmt']-temp_df.iloc[0, 1])
train_df['diff_is_fraud'] = np.abs(train_df['TransactionAmt']-temp_df.iloc[1, 1])
test_df['diff_is_not_fraud'] = np.abs(test_df['TransactionAmt']-temp_df.iloc[0, 1])
test_df['diff_is_fraud'] = np.abs(test_df['TransactionAmt']-temp_df.iloc[1, 1])


# %%
# Merge Identity columns
temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left') #43
del temp_df['TransactionID']
train_df = pd.concat([train_df, temp_df], axis=1)
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df, temp_df], axis=1)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #451
# %%
# For our model current TransactionAmt is a noise
# https://www.kaggle.com/kyakovlev/ieee-check-noise
# (even if features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations
# Transactionamt以外にも、id_02, D15をまとめてみる
i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        for agg_col in ['TransactionAmt', 'id_02', 'D15', 'diff_is_not_fraud', 'diff_is_fraud']:
            new_col_name = '_'.join([col, agg_col, agg_type])
            temp_df = pd.concat([train_df[[col, agg_col]],
                                test_df[[col, agg_col]]])

            #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])[agg_col].agg([agg_type]).reset_index().rename(
                                                    columns={agg_type: new_col_name})
            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()
            train_df[new_col_name] = train_df[col].map(temp_df)
            test_df[new_col_name] = test_df[col].map(temp_df)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #493
# %%
# ver7(skewを追加)
i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']

for col in i_cols:
    for agg_type in ['skew']:
        for agg_col in ['TransactionAmt', 'id_02', 'D15', 'diff_is_not_fraud', 'diff_is_fraud']:
            new_col_name = '_'.join([col, agg_col, agg_type])
            temp_df = pd.concat([train_df[[col, agg_col]],
                                test_df[[col, agg_col]]])

            #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])[agg_col].agg([agg_type]).reset_index().rename(
                                                    columns={agg_type: new_col_name})
            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()
            train_df[new_col_name] = train_df[col].map(temp_df)
            test_df[new_col_name] = test_df[col].map(temp_df)

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #514
# %%
# median系を追加(84特徴が増える)
i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']
for col in i_cols:
    for agg_col in ['TransactionAmt', 'id_02', 'D15']:
        for q in [0.25, 0.5, 0.75]:
            new_col_name = '_'.join([col, agg_col, 'quantile', str(q)])
            temp_df = pd.concat([train_df[[col, agg_col]],
                                 test_df[[col, agg_col]]])
            temp_df = temp_df.groupby([col])[agg_col].quantile(q).reset_index().rename(columns={agg_col: new_col_name})
            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()
            train_df[new_col_name] = train_df[col].map(temp_df)
            test_df[new_col_name] = test_df[col].map(temp_df)

        col_quantile1 = '_'.join([col, agg_col, 'quantile', str(0.25)])
        col_quantile3 = '_'.join([col, agg_col, 'quantile', str(0.75)])
        new_col_name = '_'.join([col, agg_col, 'iqr'])
        train_df[new_col_name] = train_df[col_quantile3] - train_df[col_quantile1]
        test_df[new_col_name] = test_df[col_quantile3] - test_df[col_quantile1]

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #514+84

# %%
# カテゴリでまとめた変数の差/比を特徴に
cols = ['card1_TransactionAmt_mean', 'card2_TransactionAmt_mean', 'card3_TransactionAmt_mean',
        'card5_TransactionAmt_mean', 'uid_TransactionAmt_mean', 'uid2_TransactionAmt_mean',
        'uid3_TransactionAmt_mean']

for col in cols:
    new_col_name1 = 'diff_' + col
    new_col_name2 = 'ratio_' + col
    train_df[new_col_name1] = train_df['TransactionAmt'] - train_df[col]
    train_df[new_col_name2] = train_df['TransactionAmt'] / train_df[col]
    test_df[new_col_name1] = test_df['TransactionAmt'] - test_df[col]
    test_df[new_col_name2] = test_df['TransactionAmt'] / test_df[col]

# %%
# def _distance(a, b):
#     return np.linalg.norm(b - a)


# def _get_feat(data, X_train, y_train, class_index, k_index):
#     inclass_X = X_train[y_train == class_index]
#     distances = np.array([_distance(a, data) for a in inclass_X])
#     sorted_distances_index = np.argsort(distances)
#     nearest_index = list(sorted_distances_index[0: (k_index + 1)])
#     dist = np.sum(distances[nearest_index])
#     return dist


# def knnExtract(X, y, k=1, folds=5):
#     CLASS_NUM = len(set(y))
#     res = np.empty((len(X), CLASS_NUM * k))
#     kf = KFold(n_splits=folds,  shuffle=True)

#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train = y[train_index]

#         features = np.empty([0, len(X_test)])

#         for class_index in range(CLASS_NUM):
#             for k_index in range(k):
#                 feat = np.array([np.apply_along_axis(
#                     _get_feat, 1,
#                     X_test, X_train, y_train,
#                     class_index, k_index
#                 )])
#                 features = np.append(features, feat, axis=0)
#         res[test_index] = features.T

#     return res

# %%

# %%
# 参考にあったcard1とcard2でlogをとる作業を
train_df['card1'] = np.log1p(train_df['card1'])
test_df['card1'] = np.log1p(test_df['card1'])
train_df['card2'] = np.log1p(train_df['card2'])
test_df['card2'] = np.log1p(test_df['card2'])

# Small "hack" to transform distribution
# (doesn't affect auc much, but I like it more)
# please see how distribution transformation can boost your score
# (not our case but related)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])
# %%
########################### 'P_emaildomain' - 'R_emaildomain'
p = 'P_emaildomain'
r = 'R_emaildomain'
uknown = 'email_not_provided'

for df in [train_df, test_df]:
    df[p] = df[p].fillna(uknown)
    df[r] = df[r].fillna(uknown)
    
    # Check if P_emaildomain matches R_emaildomain
    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #517

## Local test doesn't show any boost here, 
## but I think it's a good option for model stability

## Also, we will do frequency encoding later

# %%
########################### Device info
for df in [train_df, test_df]:
    # Device info
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['DeviceInfo_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]
    df.loc[df['DeviceInfo_device'].str.contains('SM', na=False), 'DeviceInfo_device'] = 'Samsung'
    df.loc[df['DeviceInfo_device'].str.contains('SAMSUNG', na=False), 'DeviceInfo_device'] = 'Samsung'
    df.loc[df['DeviceInfo_device'].str.contains('GT-', na=False), 'DeviceInfo_device'] = 'Samsung'
    df.loc[df['DeviceInfo_device'].str.contains('Moto G', na=False), 'DeviceInfo_device'] = 'Motorola'
    df.loc[df['DeviceInfo_device'].str.contains('Moto', na=False), 'DeviceInfo_device'] = 'Motorola'
    df.loc[df['DeviceInfo_device'].str.contains('moto', na=False), 'DeviceInfo_device'] = 'Motorola'
    df.loc[df['DeviceInfo_device'].str.contains('LG-', na=False), 'DeviceInfo_device'] = 'LG'
    df.loc[df['DeviceInfo_device'].str.contains('rv:', na=False), 'DeviceInfo_device'] = 'RV'
    df.loc[df['DeviceInfo_device'].str.contains('HUAWEI', na=False), 'DeviceInfo_device'] = 'Huawei'
    df.loc[df['DeviceInfo_device'].str.contains('ALE-', na=False), 'DeviceInfo_device'] = 'Huawei'
    df.loc[df['DeviceInfo_device'].str.contains('-L', na=False), 'DeviceInfo_device'] = 'Huawei'
    df.loc[df['DeviceInfo_device'].str.contains('Blade', na=False), 'DeviceInfo_device'] = 'ZTE'
    df.loc[df['DeviceInfo_device'].str.contains('BLADE', na=False), 'DeviceInfo_device'] = 'ZTE'
    df.loc[df['DeviceInfo_device'].str.contains('Linux', na=False), 'DeviceInfo_device'] = 'Linux'
    df.loc[df['DeviceInfo_device'].str.contains('XT', na=False), 'DeviceInfo_device'] = 'Sony'
    df.loc[df['DeviceInfo_device'].str.contains('HTC', na=False), 'DeviceInfo_device'] = 'HTC'
    df.loc[df['DeviceInfo_device'].str.contains('ASUS', na=False), 'DeviceInfo_device'] = 'Asus'
    df.loc[df.DeviceInfo_device.isin(df.DeviceInfo_device.value_counts()[df.DeviceInfo_device.value_counts() < 200].index), 'DeviceInfo_device'] = "Others"

    # Device info 2
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
    
    # Browser
    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #522




# %%
# Freq encoding
i_cols = ['card1', 'card2', 'card3', 'card5',
          'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
          'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
          'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
          'addr1', 'addr2',
          'dist1', 'dist2',
          'P_emaildomain', 'R_emaildomain',
          'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
          'id_30', 'id_30_device', 'id_30_version',
          'id_31_device',
          'id_33',
          'uid', 'uid2', 'uid3'
          ]

for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc'] = test_df[col].map(fq_encode)


for col in ['DT_M', 'DT_W', 'DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total'] = test_df[col].map(fq_encode)
        

periods = ['DT_M', 'DT_W', 'DT_D']
i_cols = ['uid']
for period in periods:
    for col in i_cols:
        new_column = col + '_' + period
            
        temp_df = pd.concat([train_df[[col, period]], test_df[[col, period]]])
        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
        fq_encode = temp_df[new_column].value_counts().to_dict()
            
        train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
        test_df[new_column] = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
        
        train_df[new_column] /= train_df[period+'_total']
        test_df[new_column] /= test_df[period+'_total']

print('train:', train_df.shape[1], 'test:', test_df.shape[1]) #568

# %%
########################### Encode Str columns
# For all such columns (probably not)
# we already did frequency encoding (numeric feature)
# so we will use astype('category') here
for col in list(train_df):
    if train_df[col].dtype == 'O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col] = test_df[col].fillna('unseen_before_label')
        
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')



# %%
# uid1, 2, 3を使って直後のデータとの差分とかそういうの見る感じ
# - uidが出た直後(直前)の日付の差分or時間の差分(なるべくばらつきが出るようなもの))
# - TransactionAmtとか日付とかここを検討


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
    test_predictions[['TransactionID','isFraud']].to_csv('kernel1_remodel8/submission8.csv', index=False)

# %%
## LOCAL_TEST = True
# auc→0.9215720401092782(ver7+ratioとかdiffを追加)(今までのmax→0.9211773468405661)
# auc→0.9211773468405661(ver4)
# feat_imp
# 664   1239    diff_uid3_TransactionAmt_mean
# 665   1246   diff_card5_TransactionAmt_mean
# 666   1267                    uid3_D15_mean
# 667   1269  ratio_card2_TransactionAmt_mean
# 668   1298                         uid_DT_M
# 669   1319                               D4
# 670   1330            uid3_D15_quantile_0.5
# 671   1333                  uid3_id_02_mean
# 672   1355   diff_card2_TransactionAmt_mean
# 673   1369         uid3_id_02_quantile_0.75
# 674   1375   ratio_uid3_TransactionAmt_mean
# 675   1377  ratio_card5_TransactionAmt_mean
# 676   1439             P_emaildomain_fq_enc
# 677   1449                               D2
# 678   1488                              C13
# 679   1492                      uid3_fq_enc
# 680   1496          uid3_id_02_quantile_0.5
# 681   1501                   uid3_id_02_iqr
# 682   1507                     addr1_fq_enc
# 683   1578                            id_02
# 684   1585                            dist1
# 685   1727                     uid3_D15_std
# 686   1737                            addr1
# 687   1793         uid3_id_02_quantile_0.25
# 688   1878                   uid3_id_02_std
# 689   1894                              D15
# 690   1907                         uid_DT_W
# 691   1998                            card1
# 692   2069                  uid3_id_02_skew
# 693   2418                         uid_DT_D

# 
## LOCAL_TEST = False
# max auc(lb)→0.9485(feature_eliminateした場合)
# max auc(cvの)→0.976896
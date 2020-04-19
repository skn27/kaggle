# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again

# %%
# General imports
import numpy as np
import pandas as pd
import gc
import random
import os
import warnings
import datetime

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
# 最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 50)

# %%
#  Helpers
# Seeder
# :seed to make all processes deterministic     # type: int


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool


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
# transaction的な情報
train_df = pd.read_csv('data/train_transaction.csv')
# transaction的な情報
test_df = pd.read_csv('data/test_transaction.csv')
test_df['isFraud'] = 0

# transactionについて補足的な情報
train_identity = pd.read_csv('data/train_identity.csv')

# transactionについて補足的な情報
test_identity = pd.read_csv('data/test_identity.csv')

sub = pd.read_csv('data/sample_submission.csv')
# %%
# Base Minification

train_df = reduce_mem_usage(train_df)
test_df  = reduce_mem_usage(test_df)

train_identity = reduce_mem_usage(train_identity)
test_identity  = reduce_mem_usage(test_identity)

# %%
# card4, card6, ProductCD
# Converting Strings to ints(or floats if nan in column) using frequency encoding
# We will be able to use these columns as category or as numerical feature
# カテゴリ変数を出現頻度でencode

for col in ['card4', 'card6', 'ProductCD']:
    print('Encoding', col)
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    col_encoded = temp_df[col].value_counts().to_dict()   
    train_df[col] = train_df[col].map(col_encoded)
    test_df[col]  = test_df[col].map(col_encoded)
    print(col_encoded)

# %%

# M columns
# Converting Strings to ints(or floats if nan in column)

# この中は全部T or F or missingvalだからあらかじめこうやって置換
for col in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']:
    train_df[col] = train_df[col].map({'T': 1, 'F': 0})
    test_df[col] = test_df[col].map({'T': 1, 'F': 0})

# M4はM0, M1, M2, missingvalだから別にencode(frequencyでencode)
for col in ['M4']:
    print('Encoding', col)
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    col_encoded = temp_df[col].value_counts().to_dict()   
    train_df[col] = train_df[col].map(col_encoded)
    test_df[col] = test_df[col].map(col_encoded)
    print(col_encoded)

# %%
test_identity[['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
       'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
       'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']].describe(include='all')

# %%
test_identity[['DeviceType', 'DeviceInfo']].describe(include='all')
# %%
tmp_train = train_identity.copy()
tmp_test = test_identity.copy()

# %%


# %%
# Identity columns

# id_12-id_38でnuniqueがNaNじゃないものを扱っているっぽい
# NaNかどうかの調べかた
def minify_identity_df(df):

    df['id_12'] = df['id_12'].map({'Found': 1, 'NotFound': 0})
    df['id_15'] = df['id_15'].map({'New': 2, 'Found': 1, 'Unknown': 0})
    df['id_16'] = df['id_16'].map({'Found':1, 'NotFound':0})

    df['id_23'] = df['id_23'].map({'IP_PROXY:TRANSPARENT':3, 'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1})

    df['id_27'] = df['id_27'].map({'Found':1, 'NotFound':0})
    df['id_28'] = df['id_28'].map({'New':2, 'Found':1})

    df['id_29'] = df['id_29'].map({'Found':1, 'NotFound':0})

    df['id_35'] = df['id_35'].map({'T':1, 'F':0})
    df['id_36'] = df['id_36'].map({'T':1, 'F':0})
    df['id_37'] = df['id_37'].map({'T':1, 'F':0})
    df['id_38'] = df['id_38'].map({'T':1, 'F':0})

    df['id_34'] = df['id_34'].fillna(':-2')
    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
    df['id_34'] = np.where(df['id_34']==-2, np.nan, df['id_34'])
    
    df['id_33'] = df['id_33'].fillna('0x0')
    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
    df['id_33'] = np.where(df['id_33'] == '0x0', np.nan, df['id_33'])

    df['DeviceType'].map({'desktop':1, 'mobile':0})
    return df




    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    dataframe['had_id'] = 1
    gc.collect()

    return dataframe
# %%

tmp_train = minify_identity_df(tmp_train)
tmp_test = minify_identity_df(tmp_test)

tmp_train['id_34'].value_counts()
# %%

for col in ['id_33']:
    train_identity[col] = train_identity[col].fillna('unseen_before_label')
    test_identity[col]  = test_identity[col].fillna('unseen_before_label')
    
    le = LabelEncoder()
    le.fit(list(train_identity[col])+list(test_identity[col]))
    train_identity[col] = le.transform(train_identity[col])
    test_identity[col]  = le.transform(test_identity[col])

# %%
train_identity['id_34'].value_counts()
#%%
########################### Final Minification
#################################################################################

train_df = reduce_mem_usage(train_df)
test_df  = reduce_mem_usage(test_df)

train_identity = reduce_mem_usage(train_identity)
test_identity  = reduce_mem_usage(test_identity)

#%%
########################### Export
#################################################################################

train_df.to_pickle('data/train_transaction.pkl')
test_df.to_pickle('data/test_transaction.pkl')

train_identity.to_pickle('data/train_identity.pkl')
test_identity.to_pickle('data/test_identity.pkl')

# %%
tmp = train_df['D9'].fillna(-1)
import seaborn as sns
sns.distplot(tmp)
# %%
# これの改良→ドメイン分けたり、デバイスの種類分けたりの考察をしたい。

# カテゴリデータ一覧
# --transaction--
# ProductCD
# card1 - card6
# addr1, addr2
# P_emaildomain
# R_emaildomain
# M1 - M9
# --identity--
# DeviceType
# DeviceInfo
# id_12 - id_38

# まだエンコードしてないカテゴリデータ
# --transaction--
# card1, 3, 5(エンコードせず、新しいIDを作るのに使っている)
# addr1, 2
# P_emaildomein (もう一つで扱われている)
# R_emaildomein (もう一つで扱われている)
# --identity--
# DevideInfo
# id_13, 14, 17-22, 24, 26, 32

# 次にやること→まだえんこーどしていないカテゴリデータの扱いについて調べる。
# 調べて、もしこっちに処理がまとめられそうならまとめてからpickle、
# あとはid_splitをチェックしてもう少し改良を
# あらかた改良が終わったら、negative-down-samplingしてからpickleする。
# 参考→https://www.slideshare.net/TakanoriHayashi3/talkingdata-adtracking-fraud-detection-challenge-1st-place-solution

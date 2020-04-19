# https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again/data

# %%

# General imports
import numpy as np
import pandas as pd
import gc
import random
import os
import warnings
import datetime
import keras
import random
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')
# 最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 50)

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


# SEED = 42
# seed_everything(SEED)
LOCAL_TEST = False
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
train_df = pd.merge(train_df, train_identity, how='left', on='TransactionID')
test_df = pd.merge(test_df, test_identity, how='left', on='TransactionID')
del train_identity, test_identity
# %%
useful_features = list(train_df.iloc[:, 3:55].columns) + ['DeviceInfo', 'id_30', 'id_31']

y = train_df.sort_values('TransactionDT')['isFraud']
X = train_df.sort_values('TransactionDT')[useful_features]
X_test = test_df[useful_features]

# %%
########################### 'P_emaildomain' - 'R_emaildomain'
p = 'P_emaildomain'
r = 'R_emaildomain'
uknown = 'email_not_provided'

for df in [X, X_test]:
    df[p] = df[p].fillna(uknown)
    df[r] = df[r].fillna(uknown)
    
    # Check if P_emaildomain matches R_emaildomain
    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

# %%
########################### Device info
for df in [X, X_test]:
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


# %%
X = X.drop(['id_30', 'id_31'], axis=1)
X_test = X_test.drop(['id_30', 'id_31'], axis=1)

# %%
X.isnull().sum()
# %%
categorical_features = [
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'P_emaildomain',
    'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
    'id_30_device', 'id_30_version', 'id_31_device',
    'P_emaildomain_prefix', 'R_emaildomain_prefix',
]

continuous_features = list(filter(lambda x: x not in categorical_features, X))

# %%
continuous_features
# %%
class ContinuousFeatureConverter:
    def __init__(self, name, feature, log_transform):
        self.name = name
        self.skew = feature.skew()
        self.log_transform = log_transform
        
    def transform(self, feature):
        if self.skew > 1:
            feature = self.log_transform(feature)
        
        mean = feature.mean()
        std = feature.std()
        return (feature - mean)/(std + 1e-6)

# %%
feature = X['TransactionAmt']
log = lambda x: np.log10(x + 1 - min(0, x.min()))
converter = ContinuousFeatureConverter(f, feature, log)
# %%
feature_converters = {}
continuous_train = pd.DataFrame()
continuous_test = pd.DataFrame()

for f in continuous_features:
    feature = X[f]
    feature_test = X_test[f]
    log = lambda x: np.log10(x + 1 - min(0, x.min()))
    converter = ContinuousFeatureConverter(f, feature, log)
    print(f)
    feature_converters[f] = converter    
    continuous_train[f] = log(feature)
    continuous_test[f] = log(feature_test)
# %%
continuous_train['isna_sum'] = continuous_train.isna().sum(axis=1)
continuous_test['isna_sum'] = continuous_test.isna().sum(axis=1)

continuous_train['isna_sum'] = (continuous_train['isna_sum'] - continuous_train['isna_sum'].mean())/continuous_train['isna_sum'].std()
continuous_test['isna_sum'] = (continuous_test['isna_sum'] - continuous_test['isna_sum'].mean())/continuous_test['isna_sum'].std()

# %%
isna_columns = []
for column in continuous_features:
    isna = continuous_train[column].isna()
    if isna.mean() > 0.:
        continuous_train[column + '_isna'] = isna.astype(int)
        continuous_test[column + '_isna'] = continuous_test[column].isna().astype(int)
        isna_columns.append(column)
# %%

continuous_train = continuous_train.fillna(continuous_train.mean(skipna=True))
continuous_test = continuous_test.fillna(continuous_test.mean(skipna=True))


# %%
def categorical_encode(df_train, df_test, categorical_features, n_values=50):
    df_train = df_train[categorical_features].astype(str)
    df_test = df_test[categorical_features].astype(str)
    
    categories = []
    for column in categorical_features:
        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])
        values2use = categories[-1]
        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')
        df_test[column] = df_test[column].apply(lambda x: x if x in values2use else 'Other')
        
    
    ohe = OneHotEncoder(categories=categories)
    ohe.fit(pd.concat([df_train, df_test]))
    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)
    df_test = pd.DataFrame(ohe.transform(df_test).toarray()).astype(np.float16)
    return df_train, df_test
# %%
train_categorical, test_categorical = categorical_encode(X, X_test, categorical_features)

# %%
X = pd.concat([continuous_train, train_categorical], axis=1)
del continuous_train, train_categorical
X_test = pd.concat([continuous_test, test_categorical], axis=1)
del continuous_test, test_categorical

# %%
split_ind = int(X.shape[0]*0.8)

X_tr = X.iloc[:split_ind]
X_val = X.iloc[split_ind:]

y_tr = y.iloc[:split_ind]
y_val = y.iloc[split_ind:]

del X

# %%
SEED = 27
seed_everything(SEED)

# %%
# Compatible with tensorflow backend
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
get_custom_objects().update({'focal_loss_fn': focal_loss()})

# %%
def create_model(loss_fn):
    inps = Input(shape=(X_tr.shape[1],))
    x = Dense(512, activation=custom_gelu)(inps)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation=custom_gelu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Nadam(),
        loss=[loss_fn]
    )
    #model.summary()
    return model

# %%
model_focal = create_model('focal_loss_fn')
model_bce = create_model('binary_crossentropy')
es_cb_focal = EarlyStopping(monitor='focal_loss_fn', patience=0, verbose=1, mode='auto')
es_cb_bce = EarlyStopping(monitor='binary_crossentropy', patience=0, verbose=1, mode='auto')
# %%
model_bce.fit(
    X_tr, y_tr, epochs=50, batch_size=2048, validation_data=(X_val, y_val), verbose=True, 
    callbacks=[roc_callback(training_data=(X_val, y_tr), validation_data=(X_val, y_val)), es_cb_focal]
)
model_focal.fit(
    X_tr, y_tr, epochs=50, batch_size=2048, validation_data=(X_val, y_val), verbose=True, 
    callbacks=[roc_callback(training_data=(X_val, y_tr), validation_data=(X_val, y_val)), es_cb_bce]
)
# %%
val_preds_bce = model_bce.predict(X_val).flatten()
val_preds_focal = model_focal.predict(X_val).flatten()
# %%
print('BCE preds: ', roc_auc_score(y_val, val_preds_bce))
print('Focal preds: ',roc_auc_score(y_val, val_preds_focal))
print('Correlation matrix: ')
print(np.corrcoef(val_preds_bce, val_preds_focal))
print("Spearman's correlation: ", spearmanr(val_preds_bce, val_preds_focal).correlation)
print('Averaging: ', roc_auc_score(y_val, val_preds_bce + val_preds_focal))
print('Rank averaging: ', roc_auc_score(y_val, rankdata(val_preds_bce, method='dense') + rankdata(val_preds_focal, method='dense')))

# %%
# fine-tuning
model_bce.fit(X_val, y_val, epochs=10, batch_size=2048, verbose=True)
model_focal.fit(X_val, y_val, epochs=10, batch_size=2048, verbose=True)
# %%
sub = pd.read_csv('data/sample_submission.csv')
sub['bce_predict'] = model_bce.predict(X_test).flatten()
sub['focal_predict'] = model_bce.predict(X_test).flatten()
sub.to_csv('stacking/submission_nn.csv', index=False)






#%%

#%% [markdown]
# # モデル作成のポイント
# - 前月からの売り上げ予測として考える(2015年10月から2015年11月の売り上げをイメージ)
# - 月ごとの売り上げdfを作成することを考える
# - [lag operator](https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9)を使って
# lagの値を取得してNaNをゼロで埋め、値を0-20の範囲にクリップする。(差分というよりは単純に前月の売り上げを特徴にすればいい)
# - 必ずターゲットの値を0-20の値に収めること
# - gradient boost treeモデルに突っ込む
# - アイテム/ショップペアラグとは別に、あなたは合計ショップまたは合計アイテム売上のラグ値を加えることを試みることができます（それは本質的に平均エンコーディングです）。 
# - ショップ(アイテムID)でグループバイしてlagの合計を計算する。それを各ショップ(アイテムID)に対応したencodeで作った特徴になる。
#%% [markdown]
#  # 目標
#  - 2015年11月に各ショップで各アイテムがいくつ売れるかを予測する。
#  - 11月の日毎に幾つ売れるか予測してから、それを合算した結果をまとめるのもあり。#  
#  ---
#  
#  目的変数:item_cent_day
#  
#  説明変数:date, date_block_num, shop_id, item_id, item_price
#%% [markdown]
# # メモ
# 
# - 説明変数にするために日付を曜日に変換、weekdayかholidayかにするのはあり。
# - 特徴を何にするか→曜日、平日or休日、date_block_num、item_id、item_category_id
# - 時系列予測みたいなのはかなりきつい(お店x品物分のモデル構築は不可能だし(データ少ない)、汎用的なのは無理)
# - date_block_numはテストだと新しくなるけど、大丈夫なのか
# - データは日毎ではあるけど、モデルの予測は月ごとである必要がある→こういう場合なるべく情報を持つにはどうすればいいのか
#%% [markdown]
# # EDA
# 
# *やりたいこと*
# - 可視化(各ショップのアイテムがどれだけ売れるかを棒グラフで)
#%% [markdown]
#  ## 必要なライブラリの読み込み

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["font.family"] = "IPAexGothic"
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import pickle
from itertools import product
import lightgbm as lgb

#%% [markdown]
#  ## データの読み込み

#%%
path = '/Users/soki/Documents/kaggle/coursera/data/'
# [item_category_name, item_category_id]
item_categories = pd.read_csv(path + 'item_categories.csv')
# [item_name, item_id, item_category_id]
items = pd.read_csv(path + 'items.csv')

# [date, date_block_num, shop_id, item_id, item_price, item_cnt_day]
sales_train = pd.read_csv(path + 'sales_train.csv.gz')
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
# [shop_name, shop_id]
shops = pd.read_csv(path + 'shops.csv')
test = pd.read_csv(path + 'test.csv.gz')
#%%
sales_train.head(10)
#%%
test.head(10)
#%%
len(items)

#%% [markdown]
# ## 欠損値の確認
#%%
print(item_categories.isnull().any())
print(items.isnull().any())
print(sales_train.isnull().any())
print(shops.isnull().any())
print(test.isnull().any())
#%% [markdown]
# ## 外れ値の確認

#%%
sns.boxplot(sales_train['item_cnt_day'])
#%%
sns.boxplot(sales_train['item_price'])

#%%
# 外れ値の行を削除
sales_train = sales_train[sales_train.item_cnt_day < 1001]
sales_train = sales_train[sales_train.item_price < 100000]

#%%
# date_block_num=4, shop_id = 32, item_id = 2973でitem_priceが負の値になっているものがある。
# これを中央値に置き換える
median = sales_train[(sales_train.date_block_num == 4)&(sales_train.shop_id == 32)&
					 (sales_train.item_id==2973&(sales_train.item_price > 0))].item_price.median()
sales_train.loc[sales_train.item_price < 0, 'item_price'] = median

#%%
shops['shop_name']
# shop_idが0と57、1と58、10と11が重複(それぞれ後ろの数字に揃える)
#%%
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

#%%
print('train:', len(sales_train))
print('test:', len(test))
#%% [markdown]
#  ## データの可視化(店舗ごと) 

#%%
# データの変形
mat = []
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
	sales = sales_train[sales_train.date_block_num == i]
	mat.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), 
						   dtype='int16'))
matrix = pd.DataFrame(np.vstack(mat), columns=cols)
matrix[cols[0]] = matrix[cols[0]].astype(np.int8)
matrix[cols[1]] = matrix[cols[1]].astype(np.int8)
matrix[cols[2]] = matrix[cols[2]].astype(np.int16)
matrix.sort_values(cols, inplace=True)

#%%
matrix.head()
#%%
sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']
#%%
# groupbyで作ったグループに対してaggを読んでkeyの列にvalueの計算をする。
# できるのはdataframe、sumしたものがカラムになっている。
group = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

# ここで大元のデータとくっつける
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
							.fillna(0)
							.clip(0, 20)
							.astype(np.float16))
#%%
matrix.head()
#%%
# testデータとtrainデータをくっつける準備。まとめて処理をするため(ラグ特徴量をとるのに便利)
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

#%%
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)
#%%
# item_category_idを追加
matrix = pd.merge(matrix, items[['item_id', 'item_category_id']], on='item_id', how='left')

#%%
matrix.head()
#%%
# columnは量を増やすことが可能
# ずらした分ともとのdfをマージするという考え方は同じ、それをうまく関数化

def lag_feature(df, lags, col):
	"""make lag feature
	
	Arguments:
		df {pd.DataFrame} -- Dataframe which want to add lag feature to 
		lags {list} -- number of lags
		col {str} -- columns of making lag_feature
	"""
	for i in lags:
		tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
		shifted = tmp.copy()
		shifted.columns = ['date_block_num', 'shop_id', 'item_id', col+'_lag_'+str(i)]
		shifted['date_block_num'] += i
		df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
	return df

#%%
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], 'item_cnt_month')
#%%
# mean_encodeしたあと、lagが有効なものについてはlagをとったもので特徴を作り、そのままのやつは消してしまう。
mean_encoded_feature_list = [['date_block_num'], ['date_block_num', 'item_id'], 
						['date_block_num', 'shop_id'], ['date_block_num', 'item_category_id'],
						['date_block_num', 'shop_id', 'item_category_id']]


group = matrix.groupby(mean_encoded_feature_list[0]).agg({'item_cnt_month':['mean']})
col = '_'.join(mean_encoded_feature_list[0]) + '_mean'
group.columns = [col]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=mean_encoded_feature_list[0], how='left')
matrix[col] = matrix[col].astype(np.float16)
matrix = lag_feature(matrix, [1], col)
matrix.drop([col], axis=1, inplace=True)

#%%
group = matrix.groupby(mean_encoded_feature_list[1]).agg({'item_cnt_month':['mean']})
col = '_'.join(mean_encoded_feature_list[1]) + '_mean'
group.columns = [col]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=mean_encoded_feature_list[1], how='left')
matrix[col] = matrix[col].astype(np.float16)
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], col)
matrix.drop([col], axis=1, inplace=True)

#%%
group = matrix.groupby(mean_encoded_feature_list[2]).agg({'item_cnt_month':['mean']})
col = '_'.join(mean_encoded_feature_list[2]) + '_mean'
group.columns = [col]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=mean_encoded_feature_list[2], how='left')
matrix[col] = matrix[col].astype(np.float16)
matrix = lag_feature(matrix, [1], col)
matrix.drop([col], axis=1, inplace=True)
#%%
group = matrix.groupby(mean_encoded_feature_list[3]).agg({'item_cnt_month':['mean']})
col = '_'.join(mean_encoded_feature_list[3]) + '_mean'
group.columns = [col]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=mean_encoded_feature_list[3], how='left')
matrix[col] = matrix[col].astype(np.float16)
matrix = lag_feature(matrix, [1], col)
matrix.drop([col], axis=1, inplace=True)

#%%
group = matrix.groupby(mean_encoded_feature_list[4]).agg({'item_cnt_month':['mean']})
col = '_'.join(mean_encoded_feature_list[4]) + '_mean'
group.columns = [col]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=mean_encoded_feature_list[4], how='left')
matrix[col] = matrix[col].astype(np.float16)
matrix = lag_feature(matrix, [1], col)
matrix.drop([col], axis=1, inplace=True)
#%%
# special features
matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

#%%
len(matrix[matrix.date_block_num == 34])
#%%
# 各ショップでそのitemが最後に売れてからどれだけ立っているかを特徴量にしている
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         

#%%
# ショップ関係なしにitemが最後に売れてからどれだけ立っているかを特徴量にしている。
cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num 
#%%
len(matrix[matrix.date_block_num == 34])
#%%
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

#%%
# 12以上のラグを取っているため前の情報は捨てる
matrix = matrix[matrix.date_block_num > 11]
# matrix内のnanを全部入れ替える
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df
matrix = fill_na(matrix)

#%%
len(matrix[matrix.date_block_num == 34])

#%%
matrix.to_csv('tr_df.csv', index = False)
del matrix
del cache
del group
del items
del shops
del sales_train
#%%
data = pd.read_csv('/Users/soki/Documents/kaggle/tr_df.csv')
#%%
min(data['date_block_num'])
#%%
holydays_dict = {12:11, 13:4, 14:7, 15:4, 16:7, 17:7, 18:4, 19:5, 20:4, 21:4, 
                 22:7, 23:4, 24:11, 25:5, 26:6, 27:4, 28:9, 29:5, 30:4, 31:5, 
                 32:4, 33:4, 34:6}
data['holydays'] = data['date_block_num'].map(holydays_dict).astype(np.int8)
data['holydays_percentage'] = (data['holydays'] / data['days']).astype(np.float16)
#%%
data.head()
#%%
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
#%%
len(X_test)
#%%

model_xgb = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model_xgb.fit(
    X_train, 
    Y_train, 
    eval_metric='rmse', 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

Y_test_xgb = model_xgb.predict(X_test).clip(0, 20)
#%%
Y_test_xgb

#%%
#%%
# rf(欠損値を含む行を除外)
X_train_rf = X_train.fillna(-999)
X_test_rf = X_test.fillna(-999)

model_rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=7,
    max_features='auto',
    min_samples_leaf=2,
    random_state=1
)

model_rf.fit(X_train_rf, Y_train)

#%%
Y_test_rf = model_rf.predict(X_test_rf).clip(0, 20)
#%%
#%%
# LightGBMでやってみる
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid)
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.2,
    'feature_fraction': 0.9,
    'verbose': 0

}

model_lgbm = lgb.train(lgbm_params, lgb_train, num_boost_round=100, 
                    valid_sets=lgb_eval, early_stopping_rounds=10)

#%%
Y_test_lgbm = model_lgbm.predict(X_test, num_iteration=model_lgbm.best_iteration)

#%%
Y_test_ensemble = np.mean([Y_test_xgb, Y_test_lgbm, Y_test_rf], axis=0)
submission = pd.DataFrame({
    'ID': test.index, 
    'item_cnt_month': Y_test_ensemble
})
submission.to_csv('lgbm_submission_add_feat_ensemble.csv', index=False)
#%% [markdown]
# 明日やること→欠損値をどう処理すべきかの考察、lightbgmで実装

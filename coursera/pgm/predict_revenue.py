#%% [markdown]
# # モデル作成のポイント
# - 前月からの売り上げ予測として考える(2015年10月から2015年11月の売り上げをイメージ)
# - 月ごとの売り上げdfを作成することを考える
# - [lag operator](https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9)を使って
# lagの値を取得してNaNをゼロで埋め、値を0-20の範囲にクリップする。
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

#%%
sales_train.head(10)


#%%
test.head(10)

#%%
len(items)

#%%
print('train min/max date: %s/%s'%(min(sales_train['date']), 
				   max(sales_train['date'])))

#%% [markdown]
# ## 欠損値の確認

#%%
print(item_categories.isnull().any())
print(items.isnull().any())
print(sales_train.isnull().any())
print(shops.isnull().any())
print(test.isnull().any())

#%% [markdown]
#  ## データの可視化(店舗ごと) 

#%%
# 月ごとに、店舗ごとに、商品ごとにデータを分類
sales_train['revenue'] = sales_train['item_price']*sales_train['item_cnt_day']
new_df = sales_train.drop(['item_price'], axis=1)
new_df.head(10)
grouped = new_df.groupby(['date_block_num', 'shop_id', 'item_id'])
sorted_df = grouped.sum().reset_index()
df0 = sorted_df[sorted_df['date_block_num'] == 0]
df1 = sorted_df[sorted_df['date_block_num'] == 1]

#%%
df1.head(10)

#%%
left = df0[df0.shop_id == 0]
right = items[['item_id', 'item_category_id']]
pd.merge(left, right, on='item_id', how='right').sort_values('item_id')

#%%
df_c = pd.merge(df1[df1.shop_id == 0], right, on='item_id', how='right').sort_values('item_id')
df_c[df_c.item_id == 30]


#%%
print(min(shops['shop_id']), max(shops['shop_id']))


#%%
df_list = []
for i in range(34):
    for j in range(60):
	tmp_df = sorted_df[sorted_df['date_block_num'] == i]
	tmp_df = tmp_df[tmp_df['shop_id'] == j]
	df_c = pd.merge(tmp_df, right, on='item_id', how='right').sort_values('item_id')
	df_c = df_c.fillna({'date_block_num':i, 'shop_id':j, 'item_cnt_day':0, 'revenue':0})
	df_c['date_block_num'] = df_c['date_block_num'].astype(np.int32)
	df_c['shop_id'] = df_c['shop_id'].astype(np.int32)
	df_list.append(df_c)

#%%
df_c_concat = pd.concat(df_list, sort=True)
# df_c_concat.to_csv('df_concat.csv') 
#%%
df_c_concat.head(100)
# df_c_concat = pd.read_csv('/Users/soki/Documents/kaggle/coursera/pgm/df_concat.csv')
#%%
df_diff = df_c_concat.copy()
df_diff = df_diff.drop('revenue', axis=1)
#%%
df_diff0 = df_diff[df_diff.shop_id == 0]
df_diff0.head(100)
#%%
df_diff0_sorted = df_diff0.sort_values(['date_block_num', 'item_id'])

#%%
df_diff0_sorted.head(100)

#%%
df_diff0_sorted['diff_item_cnt_day'] = df_diff0_sorted['item_cnt_day'].diff(22170)

#%%
df_diff0_sorted.head(30000)
#%%

#%%
df_diff = df_c_concat.copy()
df_diff = df_diff.drop('revenue', axis=1)
df_diff_list = []

for i in range(60):
    df_diff_tmp = df_diff[df_diff.shop_id == i].sort_values(['date_block_num', 'item_id'])
    df_diff_tmp['diff_item_cnt_day'] = df_diff_tmp['item_cnt_day'].diff(22170)
    df_diff_list.append(df_diff_tmp)

df_train = pd.concat(df_diff_list, sort=True)
#%%
df_train = df_train.fillna(0)

#%%
df_train.head(10)
#%%
df_train.to_csv('training_data.csv')
#%% [markdown]
# # やらなきゃいけないこと
# 
# - モデル構築
# - 学習させる
# - 予測させる
# - 前月の個数と比べて予測結果を返す

# ----
# - mean-encoding使う
# - 全部が0のデータを消す
# - 
#%%
train = pd.read_csv('/Users/soki/Documents/kaggle/training_data.csv')
train = train.drop('Unnamed: 0', axis=1)
#%%
# 'diff_item_cnt_day'と'item_cnt_day' が同時に0の行を削除
df_tr = train[(train['diff_item_cnt_day'] != 0)|(train['item_cnt_day'] != 0)]

#%%
# ショップ(アイテムID)でグループバイしてlagの合計を計算する。
# それを各ショップ(アイテムID)に対応したencodeで作った特徴になる。
df_tr2 = df_tr.copy()
df_tr2['shop_id_sum_diff'] = 0
df_tr2['item_id_sum_diff'] = 0
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_ind, val_ind in kf.split(df_tr2):
	X_tr, X_val = df_tr2.iloc[tr_ind], df_tr2.iloc[val_ind]
	for col in ['item_id', 'shop_id']:
		sums = X_val[col].map(X_tr.groupby(col).diff_item_cnt_day.sum())
		if col == 'shop_id':
			df_tr2.shop_id_sum_diff.iloc[val_ind] = sums
		else:
			df_tr2.item_id_sum_diff.iloc[val_ind] = sums

#%%
print(df_tr2['shop_id_sum_diff'].isnull().value_counts())
print(df_tr2['item_id_sum_diff'].isnull().value_counts())

#%%
df_tr2.fillna(np.sum(df_tr['diff_item_cnt_day']), inplace=True)

#%%
# 普通のmeanencoding
df_tr2 = df_tr.copy()
df_tr2['shop_id_mean'] = 0
df_tr2['item_id_mean'] = 0
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_ind, val_ind in kf.split(df_tr2):
	X_tr, X_val = df_tr2.iloc[tr_ind], df_tr2.iloc[val_ind]
	for col in ['item_id', 'shop_id']:
		means = X_val[col].map(X_tr.groupby(col).item_cnt_day.mean())
		if col == 'shop_id':
			df_tr2.shop_id_mean.iloc[val_ind] = means
		else:
			df_tr2.item_id_mean.iloc[val_ind] = means

df_tr2.fillna(np.mean(df_tr['item_cnt_day']), inplace=True)
#%%
df_tr2.head()
#%%
predictors = [x for x in df_tr2.columns if x not in ['item_cnt_day']]
target = 'item_cnt_day'
predictors
#%% 
def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
	#Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain['diff_item_cnt_day'])
	with open('xgb_model.pickle', mode='wb') as fp:
		pickle.dump(alg, fp)
	#Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])

	if performCV:
		cv_score = cross_val_score(alg, dtrain[predictors], dtrain['item_cnt_day'], cv=cv_folds)

	#Print model report:
	print("Model Report")
	print("Accuracy : {:.4f}".format(r2_score(dtrain['item_cnt_day'].values, dtrain_predictions)))
	
	if performCV:
		print("CV Score : Mean - {:.6f} | Std - {:.6f} | Min - {:.6f} | Max - {:.6f}".format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

#%%
## xgboostのグリッドサーチ
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
			'objective':['reg:linear'],
			'learning_rate': [.03, 0.05, .07], #so called `eta` value
			'max_depth': [5, 6, 7],
			'min_child_weight': [4],
			'silent': [1],
			'subsample': [0.7],
			'colsample_bytree': [0.7],
			'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
					parameters,
					cv = 5,
					verbose=True)

xgb_grid.fit(df_tr2[predictors], df_tr2[target])
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

#%%
xgb_grid.best_estimator_

#%%

#%%
rmr = RandomForestRegressor(random_state=0)
modelfit(rmr, df_tr2, predictors)
#%%
xgb1 = XGBRegressor()
xgb1.fit(df_tr2[predictors], df_tr2[target])
#%%
with open('xgb_model.pickle', mode='wb') as fp:
		pickle.dump(xgb1, fp)
#%%
# テストデータを変形する。
path = '/Users/soki/Documents/kaggle/coursera/data/'
test = pd.read_csv(path + 'test.csv.gz')
test_df = test.copy()

#%%
#d_b_nを追加
# [ID, shop_id, item_id]
test_df['date_block_num'] = 34
#%%
#item_category_idを追加
right = items[['item_id', 'item_category_id']]
test_df = pd.merge(test_df, right, on='item_id', how='right').dropna().sort_values('ID')

#%%
'''
sum_diffの追加(shop, item)
'''
shop_diff_df = df_tr2[['shop_id', 'shop_id_sum_diff']]
test_df = test_df.assign(shop_id_sum_diff = test_df['shop_id'].map(shop_diff_df.groupby('shop_id').shop_id_sum_diff.mean()
))
#%%
item_diff_df = df_tr2[['item_id', 'item_id_sum_diff']]
test_df = test_df.assign(item_id_sum_diff = test_df['item_id'].map(item_diff_df.groupby('item_id').item_id_sum_diff.mean()
))
#%%
test_df.fillna(test_df['item_id_sum_diff'].mean(), inplace=True) 
#%%
'''
meanの追加(shop, item)
'''
shop_mean_df = df_tr2[['shop_id', 'shop_id_mean']].drop_duplicates()
test_df = test_df.assign(shop_id_mean = test_df['shop_id'].map(shop_mean_df.groupby('shop_id').shop_id_mean.mean()
))

item_mean_df = df_tr2[['item_id', 'item_id_mean']].drop_duplicates()
test_df = test_df.assign(item_id_mean = test_df['item_id'].map(item_mean_df.groupby('item_id').item_id_mean.mean()
))
#%%
test_df.fillna({'shop_id_mean':shop_mean_df['shop_id_mean'].mean(), 'item_id_mean':item_mean_df['item_id_mean'].mean()}, 
				inplace=True)

#%%

test_df.head()

#%%
with open('xgb_model.pickle', mode='rb') as fp:
	xgb1 = pickle.load(fp)

#%%
pred = xgb1.predict(test_df.loc[:, predictors])

#%%
test_df['pred'] = pred
test_df.head()
#%%
df_tr2_33 = df_tr2[df_tr2['date_block_num']==33][['item_cnt_day', 'item_id', 'shop_id']]
#%%
test_pred_merge = pd.merge(df_tr2_33, test_df[['ID', 'shop_id', 'item_id', 'pred_diff']], on=['shop_id', 'item_id'], how='right')

#%%
test_pred_merge.isnull().any()
#%%
test_pred_merge = test_pred_merge.fillna(0).sort_values('ID')
#%%
test_pred_merge.head()

#%%
test_pred_merge['item_cnt_month'] = test_pred_merge['item_cnt_day'] + test_pred_merge['pred_diff']

#%%
submit_df = test_pred_merge[['ID', 'item_cnt_month']]

#%%
submit_df2 = submit_df.copy()
submit_df2['ID'] = submit_df2['ID'].astype(np.int32)
submit_df2['item_cnt_month'] = submit_df2['item_cnt_month'].clip(0, 20)
#%%
submit_df2.head()
#%%
submit_df2.to_csv('submission_rfr_mean.csv', index=False)

#%% [markdown]
# 単純に前月の売り下げをそのまま記入。
#%%
df_tr_33 = df_tr[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']]
#%%
df_tr_33 = df_tr_33[df_tr_33['date_block_num'] == 32]

#%%
df_tr_33.head()

#%%
df_tr_previous = df_tr_33.groupby(['shop_id', 'item_id'], as_index=False).item_cnt_day.sum()
#%%
df_tr_previous.columns
#%%
# テストデータを変形する。
path = '/Users/soki/Documents/kaggle/coursera/data/'
test = pd.read_csv(path + 'test.csv.gz')
test_df = test.copy()

#%%
test_pred_merge = pd.merge(df_tr_previous, test_df, on=['shop_id', 'item_id'], how='right')

#%%
test_pred_merge.fillna(0, inplace=True)
test_pred_merge.head(10)
#%%
test_pred_merge = test_pred_merge[['ID', 'item_cnt_day']].sort_values('ID')
test_pred_merge['ID'] = test_pred_merge['ID'].astype(np.int32)
test_pred_merge['item_cnt_month'] = test_pred_merge['item_cnt_day'].clip(0, 20)
test_pred_merge.drop('item_cnt_day', axis=1, inplace=True)
#%%
test_pred_merge.to_csv('submission_previous.csv', index=False)

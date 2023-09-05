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
# [ID, shop_id, item_id]
test = pd.read_csv(path + 'test.csv.gz')

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

#%% [markdown]
#  ## データの可視化(店舗ごと) 

#%%
# 月ごとに、店舗ごとに、商品ごとにデータを分類
df_tr = sales_train.copy()
df_tr = df_tr.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
df_tr['item_cnt_month'] = df_tr['item_cnt_day']
df_tr.drop('item_cnt_day', axis = 1, inplace=True)
#%%
df_tr.head()
#%%
# item_category_idを追加
left = df_tr
right = items[['item_id', 'item_category_id']]
df_tr = pd.merge(left, right, on='item_id', how='left').sort_values(['date_block_num', 'shop_id', 'item_id'])

#%% [markdown]
# 単純に前月の売り下げをそのまま記入。
#%%
df_tr33 = df_tr[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']]
df_tr33 = df_tr33[df_tr33['date_block_num'] == 33]

#%%
df_tr33.head()
#%%
# テストデータを変形する。
path = '/Users/soki/Documents/kaggle/coursera/data/'
test = pd.read_csv(path + 'test.csv.gz')
test_df = test.copy()

#%%
# 予測結果をテストデータと紐付け
test_pred_merge = pd.merge(df_tr33, test_df, on=['shop_id', 'item_id'], how='right')
test_pred_merge.fillna(0, inplace=True)
test_pred_merge = test_pred_merge[['ID', 'item_cnt_month']].sort_values('ID')
test_pred_merge['ID'] = test_pred_merge['ID'].astype(np.int32)
test_pred_merge['item_cnt_month'] = test_pred_merge['item_cnt_month'].clip(0, 20)
#%%
test_pred_merge.isnull().any()

#%%
test_pred_merge.to_csv('submission_previous_ver2.csv', index=False)

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
# 
# 
# 
#%%
df_tr.head()
#%%
df_list = []
df_tr0 = df_tr[df_tr.date_block_num == 0]
df_tr0['item_cnt_prev_month'] = 0
df_list.append(df_tr0)
for i in range(1, 34):
	df_tmp = df_tr[df_tr['date_block_num'] == i]
	df_prev = df_tr[df_tr['date_block_num'] == i-1][['shop_id', 'item_id', 'item_cnt_month']]
	df_prev.rename(columns={'item_cnt_month':'item_cnt_prev_month'}, inplace=True)
	df_merge = pd.merge(df_tmp, df_prev, on=['shop_id', 'item_id'], how='left')
	df_list.append(df_merge)
#%%
df_tr = pd.concat(df_list, axis=0)
df_tr.fillna(0, inplace=True)
df_tr['item_cnt_prev_month'] = df_tr['item_cnt_prev_month'].clip(0, 20)
#%%
df_tr.to_csv('train_data_ver2.csv', index=None)

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
df_tr2['shop_id_mean'] = 0 # これはshop_idというカテゴリ変数をエンコーディングしたってこと
df_tr2['item_id_mean'] = 0 # これはitem_idというカテゴリ変数をエンコーディングしたってこと
df_tr2['item_category_id_mean'] = 0
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_ind, val_ind in kf.split(df_tr2):
	X_tr, X_val = df_tr2.iloc[tr_ind], df_tr2.iloc[val_ind]
	for col in ['item_id', 'shop_id', 'item_category_id']:
		means = X_val[col].map(X_tr.groupby(col).item_cnt_month.mean())
		if col == 'shop_id':
			df_tr2.shop_id_mean.iloc[val_ind] = means
		elif col == 'item_id':
			df_tr2.item_id_mean.iloc[val_ind] = means
		else:
			df_tr2.item_category_id_mean.iloc[val_ind] = means

df_tr2.fillna(np.mean(df_tr['item_cnt_month']), inplace=True)
#%%
df_tr2.head()
#%%
predictors = [x for x in df_tr2.columns 
				if x not in ['item_cnt_month', 'shop_id', 'item_id', 'item_category_id']]
target = 'item_cnt_month'
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
meanの追加(shop, item, item_category_id)
'''
shop_mean_df = df_tr2[['shop_id', 'shop_id_mean']].drop_duplicates()
test_df = test_df.assign(shop_id_mean = test_df['shop_id'].map(
	shop_mean_df.groupby('shop_id').shop_id_mean.mean()
))

item_mean_df = df_tr2[['item_id', 'item_id_mean']].drop_duplicates()
test_df = test_df.assign(item_id_mean = test_df['item_id'].map(
	item_mean_df.groupby('item_id').item_id_mean.mean()
))

item_category_mean_df = df_tr2[['item_category_id', 'item_category_id_mean']].drop_duplicates()
test_df = test_df.assign(item_category_id_mean = test_df['item_category_id'].map(
	item_category_mean_df.groupby('item_category_id').item_category_id_mean.mean()
))
#%%
test_df.fillna({'shop_id_mean':df_tr2['shop_id_mean'].mean(), 
				'item_id_mean':df_tr2['item_id_mean'].mean(), 
				'item_category_id_mean':df_tr2['item_category_id_mean'].mean()},
				inplace=True)
#%%
test_df.head()
#%%
"""[item_cnt_prev_monthの追加]
"""
df_tr33 = df_tr2[df_tr2.date_block_num == 33][['shop_id', 'item_id', 'item_cnt_month']]
df_tr33.rename(columns={'item_cnt_month':'item_cnt_prev_month'}, inplace=True)
test_df = pd.merge(test_df, df_tr33, on=['shop_id', 'item_id'], how='left')
test_df.fillna(0, inplace=True)
test_df['item_cnt_prev_month'] = test_df['item_cnt_prev_month'].clip(0, 20)
#%%
#%%
with open('xgb_model.pickle', mode='rb') as fp:
	xgb1 = pickle.load(fp)

#%%
pred = xgb1.predict(test_df.loc[:, predictors])

#%%
test_df['item_cnt_month'] = pred.clip(0, 20)
test_df.head()

#%%
submit_df = test_df[['ID', 'item_cnt_month']]
submit_df['ID'] = submit_df['ID'].astype(np.int32)
#%%
submit_df.head()
#%%
submit_df.to_csv('submission_xgb_mean_enc.csv', index=False)

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
test_pred_merge.head(10)にch
#%%
test_pred_merge = test_pred_merge[['ID', 'item_cnt_day']].sort_values('ID')
test_pred_merge['ID'] = test_pred_merge['ID'].astype(np.int32)
test_pred_merge['item_cnt_month'] = test_pred_merge['item_cnt_day'].clip(0, 20)
test_pred_merge.drop('item_cnt_day', axis=1, inplace=True)
#%%
test_pred_merge.to_csv('submission_previous.csv', index=False)

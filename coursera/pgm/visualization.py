#%% [markdown]
# # モデル作成のポイント
# - 前月からの売り上げ予測として考える(2015年10月から2015年11月の売り上げをイメージ)
# - 月ごとの売り上げdfを作成することを考える
# - [lag operator](https://medium.com/@NatalieOlivo/use-pandas-to-lag-your-timeseries-data-in-order-to-examine-causal-relationships-f8186451b3a9)を使って
# lagの値を取得してNaNをゼロで埋め、値を0-20の範囲にクリップする。
# - 必ずターゲットの値を0-20の値に収めること
#%% [markdown]
# # 目標、月ごとに各ショップの各アイテムがいくつ売れるかを予測する。

#%% [markdown]
# ## 必要なライブラリの読み込み

#%%
import numpy as np
import pandas as pd

#%% [markdown]
# ## データの読み込み

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
print(min(sales_train['date']), max(sales_train['date']))
sales_train.tail()
#%% [markdown]
# ## データの可視化(店舗ごと)
grouped_shop = sales_train.groupby('shop_id')

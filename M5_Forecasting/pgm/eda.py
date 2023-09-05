# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os 
if not os.getcwd() == '/Users/soki/Documents/kaggle/M5_Forecasting/':
    os.chdir('/Users/soki/Documents/kaggle/M5_Forecasting/')

# %%
calender = pd.read_csv('data/calendar.csv')
sales_train_val = pd.read_csv('data/sales_train_validation.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')
sell_prices = pd.read_csv('data/sell_prices.csv')


# %%
sales_train_val.head()

# %%

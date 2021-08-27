# -*- ecoding: utf-8 -*-
# @ModuleName: 
# @Function: 
# @Author: RankFan
# @Time: 2021/8/27 15:33

# ref: https://github.com/rmpbastos/data_science/blob/master/_0014_Boost_your_Data_Analysis_with_Pandas.ipynb

import pandas as pd
import requests
import json

PATH = 'https://raw.githubusercontent.com/rmpbastos/data_sets/main/kaggle_housing/house_df.csv'
df = pd.read_csv(PATH)
type(df)  # pandas.core.frame.DataFrame

df.head()  # head 5
df.tail()  # tail 5
df.shape  # (1460, 16)

df.info()  # summary of df
df.describe()  # 描述性统计

df['Neighborhood'].value_counts()  # count

# DataFrame index
df.set_index('Id', inplace=True)
df.index

df = pd.read_csv(PATH, index_col='Id')  # second method

# rows and columns

df.columns

df['LotArea'].head()
type(df['LotArea'])  # pandas.core.series.Series

df.rename(columns={'BedroomAbvGr': 'Bedroom'}, inplace=True)  # rename columns

df_copy = df.copy()  # copy dataframe
df_copy['Sold'] = 'N'  # add column(s)
df_copy.tail()

data_to_append = {'LotArea': [9500, 15000],
                  'Steet': ['Pave', 'Gravel'],
                  'Neighborhood': ['Downtown', 'Downtown'],
                  'HouseStyle': ['2Story', '1Story'],
                  'YearBuilt': [2021, 2019],
                  'CentralAir': ['Y', 'N'],
                  'Bedroom': [5, 4],
                  'Fireplaces': [1, 0],
                  'GarageType': ['Attchd', 'Attchd'],
                  'GarageYrBlt': [2021, 2019],
                  'GarageArea': [300, 250],
                  'PoolArea': [0, 0],
                  'PoolQC': ['G', 'G'],
                  'Fence': ['G', 'G'],
                  'SalePrice': [250000, 195000],
                  'Sold': ['Y', 'Y']}

df_to_append = pd.DataFrame(data_to_append)  # dict to dataframe
df_copy = df_copy.append(df_to_append, ignore_index=True)  # add row(s)
df_copy.tail()

df_copy.drop(labels=1461, axis=0, inplace=True)  # remove row(s) ; axis = 0
df_copy.drop(labels='Fence', axis=1, inplace=True)  # remove column(s) ; axis = 1

# loc is used to access rows and columns by label/index or based on a boolean array
df.loc[1000]  # the 1000th row; index = 1000
df.loc[1000, ['LotArea', 'SalePrice']]  # index = 1000; columns = ['LotArea', 'SalePrice']
df.loc[df['SalePrice'] >= 600000]  # df['SalePrice'] >= 600000 is condion; return boolen

# iloc is used to select data based on their integer location or based on a boolean array as well
df.iloc[0, 0]  # 1st row; 1st column
df.iloc[10, :]  # 10th column
df.iloc[:, -1]  # the last colums
df.iloc[8:12, 2:5]

df.isnull()  # detecting the missing values
df.isnull().sum()  # the sum of missing values per column
df.isnull().sum() / df.shape[0]  # ratio

# ratio > 0
for column in df.columns:
    if df[column].isnull().sum() > 0:
        print(column, ': {:.2%}'.format(df[column].isnull().sum() / df[column].shape[0]))

df_toremove = df.copy()  # copy to drop
df_toremove.drop(labels=['PoolQC'], axis=1, inplace=True)  # drop column(s)
df_toremove.dropna(subset=['GarageType'], axis=0, inplace=True)  # drop rows

df_tofill = df.copy()  # copy to fill the null
df_tofill['Fence'].fillna(value='NoFence', inplace=True)  # fiil all in the column['Fence']

garage_median = df_tofill['GarageYrBlt'].median()  # fill the median
df_tofill.fillna({'GarageYrBlt': garage_median}, inplace=True)

df['SalePrice'].plot(kind='hist');  # Histograms
df.plot(x='SalePrice', y='YearBuilt', kind='scatter')  # scatter

df.to_csv(r'./Python_经济知识综合/My_DataFrame.csv')  # save by the relative path
df.to_csv('C:/Users/username/Documents/My_DataFrame.csv')  # absolute path

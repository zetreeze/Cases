import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import catboost
import math
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
pd.options.mode.chained_assignment = None

df_items_cat = pd.read_csv('item_categories.csv')
df_items = pd.read_csv('items.csv')
df_sales_train = pd.read_csv('sales_train.csv')
df_shops = pd.read_csv('shops.csv')


df = df_sales_train.merge(df_shops, how='inner', left_on='shop_id', right_index=True)
df = df.drop('shop_id_x', axis=1)
df = df.drop('shop_id_y', axis=1)
df = df.merge(df_items, how='inner', left_on='item_id', right_index=True)
df = df.drop('item_id_x', axis=1)
df = df.drop('item_id_y', axis=1)
df = df.merge(df_items_cat, how='inner', left_on='item_category_id', right_index=True)
df = df.drop('item_category_id_x', axis=1)
df = df.drop('item_category_id_y', axis=1)

#print(df.shape[0],'rows x', df.shape[1], 'columns')

df_transp = df.T
#print(df.head().T)

#print(df.describe())
df['datetime'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
#print(df)


dtminmax =df['datetime'].agg(['min', 'max'])
#print(dtminmax)

df_plat = df[df['item_price'] > 0]


df_plat['Month'] = df_plat['datetime'].dt.month
df_plat['Year'] = df_plat['datetime'].dt.year
sort_by_months = df_plat.sort_values('Month', ascending=True)
#print(sort_by_months)

df_id_only = df_plat[['item_category_id', 'item_id', 'shop_id']]
#print(df_id_only)

#топ 10 продаваемых товаров
items = sort_by_months.groupby('item_id').agg(kolvo =('item_id','count'))
top10_items = items.sort_values('kolvo', ascending=False).head(10)

pls = pd.DataFrame()
pls['ID_item'] = top10_items.index
pls['kolich'] = top10_items.values 

#fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
#ax.vlines(x=pls.index, ymin=0, ymax=pls.kolich, color='firebrick', alpha=0.7, linewidth=20)

#for i, kolich in enumerate(pls.kolich):
#    ax.text(i, kolich+250, round(kolich, 1), horizontalalignment='center')

#ax.set_title('10 самых часто заказываемых товаров', fontdict={'size':22})
#ax.set(ylabel='Количество заказов с товаром', ylim=(0, 33000))
#ax.set(xlabel='ID товара')
#plt.xticks(pls.index, pls.ID_item, rotation=60, horizontalalignment='right', fontsize=12)
#plt.show()

priceplt = df_plat.groupby('item_price').agg(kolvo =('item_price','count'))

plot = pd.DataFrame()
plot['Item_price'] = priceplt.index
plot['quantity'] = priceplt.values 

#plt.subplot(1,2,1)
#plt.hist(plot['Item_price'], bins=10, color='red', range=(0,30000))
#plt.ylabel('Частота')
#plt.xlabel('Цена')
#plt.title('Гистограмма продаж')

#plt.subplot(1,2,2)
#plt.boxplot(plot['quantity'])
#plt.ylabel('Количество')
#plt.title('Анализ выбросов в количестве продаж')
#plt.show()


dynamic = sort_by_months.groupby('date_block_num')['item_cnt_day'].agg(mitem_c = 'count')
plot2 = pd.DataFrame()
plot2['Month'] = dynamic.index
plot2['quantity'] = dynamic.values 

#plt.plot('Month', 'quantity', data=plot2)
#plt.xlabel('Последовательный номер месяца')
#plt.ylabel('Количество проданного товара')
#plt.title('Динамика продаж')
#plt.show()

tovars = sort_by_months.groupby('item_category_id')['item_cnt_day'].agg(item_category_ct = 'sum')
tovars_names = sort_by_months.groupby('item_category_id')['item_category_name'].agg(name = 'unique')

df_tovars = tovars.merge(tovars_names, how='inner', left_on='item_category_id', right_index=True)

pd_tovars = pd.DataFrame()
pd_tovars['Количество'] = df_tovars['item_category_ct'].values
pd_tovars['Name'] = df_tovars['name'].values
pd_tovars['ID'] = df_tovars.index

pd_tovars = pd_tovars.sort_values('Количество', ascending=False).head(15)
pd_tovars['index'] =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
pd_tovars = pd_tovars.set_index('index')


#fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
#ax.vlines(x=pd_tovars.index, ymin=0, ymax=pd_tovars.Количество, color='blue', alpha=0.5, linewidth=20)

#for i, Количество in enumerate(pd_tovars.Количество):
#    ax.text(i, Количество+2500, round(Количество, 1), horizontalalignment='center')

#ax.set_title('10 самых продаваемых категорий по количеству продаж', fontdict={'size':22})
#ax.set(ylabel='Количество', ylim=(0, 700000))
#ax.set(xlabel='ID')
#plt.xticks(pd_tovars.index, pd_tovars.Name, rotation=60, horizontalalignment='right', fontsize=12)
#plt.show()


magazins = sort_by_months.groupby('shop_id')['item_cnt_day'].agg(shop_sells = 'sum')
magazins_names = sort_by_months.groupby('shop_id')['shop_name'].agg(shop_names = 'unique')

df_magazins = magazins.merge(magazins_names, how='inner', left_on='shop_id', right_index=True)

pd_magazins = pd.DataFrame()
pd_magazins['Количество'] = df_magazins['shop_sells'].values
pd_magazins['shopNname'] = df_magazins['shop_names'].values
pd_magazins['ID'] = df_magazins.index

pd_magazins = pd_magazins.sort_values('Количество', ascending=False).head(15)
pd_magazins['index'] =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
pd_magazins = pd_magazins.set_index('index')


#fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
#ax.vlines(x=pd_magazins.index, ymin=0, ymax=pd_magazins.Количество, color='blue', alpha=0.5, linewidth=20)

#for i, Количество in enumerate(pd_magazins.Количество):
#    ax.text(i, Количество+2500, round(Количество, 1), horizontalalignment='center')

#ax.set_title('10 самых популярных магазинов', fontdict={'size':22})
#ax.set(ylabel='Количество', ylim=(0, 350000))
#ax.set(xlabel='ID')
#plt.xticks(pd_magazins.index, pd_magazins.shopNname, rotation=60, horizontalalignment='right', fontsize=12)
#plt.show()

#plt.subplot(1,2,1)
#sns.boxplot(sort_by_months['item_price'])

#plt.subplot(1,2,2)
#sns.boxplot(sort_by_months['item_cnt_day'])
#plt.show()

outliersprc = sort_by_months
outlierscnt = sort_by_months

Q1price = outliersprc['item_price'].quantile(0.25)
Q3price = outliersprc['item_price'].quantile(0.75)
IQRprice = Q3price - Q1price
lower = Q1price - 1.5*IQRprice
upper = Q3price + 1.5*IQRprice

upper_array = np.where(outliersprc['item_price'] >= upper)[0]
lower_array = np.where(outliersprc['item_price'] <= lower)[0]

#print(upper_array, lower_array)

outliersprc.drop(index=upper_array, inplace=True, axis =1)
outliersprc.drop(index=lower_array, inplace=True, axis =1)


#sns.boxplot(outliersprc['item_price'])
#plt.show()

outlierscnt = outlierscnt[outlierscnt['item_cnt_day'] < 8]
outlierscnt = outlierscnt[outlierscnt['item_cnt_day'] > 0]

#sns.boxplot(outlierscnt['item_cnt_day'])
#plt.show()


forpriceavg = outliersprc.drop_duplicates(['item_id','shop_name','Year'])

priceavg = forpriceavg.groupby(['Year'])['item_price'].agg(avg = 'mean')
#print(priceavg)



#динамика продаж товаров

cnt_items_month1 = outlierscnt.groupby('date_block_num')['item_cnt_day'].agg(sum ='sum')
#print(cnt_items_month1)

date_df_md = pd.DataFrame()
date_df_md['value1'] = cnt_items_month1['sum'].values
date_df_md.value1 = date_df_md.value1.astype(int)
dates = pd.date_range(start='2013-01-01', end='2015-10-31', freq='MS')
dates_for_pd = pd.DataFrame(dates, columns=['date'])
#dates_for_pd['date'] = dates_for_pd['date'].dt.strftime('%Y-%m-%d')
date_df_md['date'] = dates_for_pd['date']
date_df_md.set_index('date', inplace=True)
date_df_md.index.freq = "MS"

#model = ExponentialSmoothing(endog = date_df_md.value1).fit()
#predictions = model.forecast(steps = 4)
#print(predictions)
#метод экспоненциального сглаживания в данном случае не подходит для прогнозированпия, так как он строится по среднему значению за все месяца и не учитывает время.

#model = ExponentialSmoothing(endog = date_df_md.value1, trend= "add", seasonal="add", seasonal_periods= 12).fit()
#predictions = model.forecast(steps = 4)
#print(predictions)
#метод линейного тренда Холта также не подходит для прогноза, не смотря на то что он учитывает сезонность, 
#т.к. Модель Холта предполагает, что тренд и уровень сезонных колебаний изменяются с постоянной скоростью

start_index = len(date_df_md) - 12
df_training = date_df_md.iloc[:start_index]
df_test = date_df_md.iloc[start_index:]

model = SARIMAX(df_training['value1'],order=(2,0,0), seasonal_order=(1,1,0,12))
results = model.fit()
#print(results.summary())

start = len(df_training)
end = len(df_training) + len(df_test) - 1
predictions_sar = results.predict(start=start, end=end, dynamic=False, type='level')

mse = mean_squared_error(df_test['value1'], predictions_sar)
#print(f'SARIMA(2,0,0)(1,1,0,12) MSE Error: {mse:11.2})')
print(math.sqrt(mse))

model = SARIMAX(date_df_md['value1'],order=(2,0,0), seasonal_order=(1,1,0,12))
results = model.fit()
fcast = results.predict(len(date_df_md), len(date_df_md)+4, type='levels')


fcastvalues = fcast.values
fcastdates = fcast.index
dffcast = pd.DataFrame({'value1': fcastvalues})
dffcast.value1 = dffcast.value1.astype(int)
datesfcast = pd.date_range(start='2015-11-01', end='2016-03-21', freq='MS')
dates_for_pd = pd.DataFrame(datesfcast, columns=['date'])
dffcast['date'] = dates_for_pd['date']
dffcast.set_index('date', inplace=True)
dffcast.index.freq = "MS"

forecast_df = pd.concat([date_df_md, dffcast])

train_fcast = forecast_df.loc[forecast_df.index <= '2015-11-01']
test_fcast = forecast_df.loc[forecast_df.index >= '2015-11-01']

fig, ax = plt.subplots(figsize=(15, 5))
train_fcast.plot(ax=ax, label='Значения продаж', title='Динамика продаж с учетом прогноза')
test_fcast.plot(ax=ax, label='Прогноз')
ax.axvline('2015-11-01', color='black', ls='--')
ax.legend(['Значения продаж', 'Прогноз продаж'])
plt.show()

#SARIMA подходит для прогнозирования, так как эта модель учитывет сезонность, а также учитывает СКО в прогнозировании.
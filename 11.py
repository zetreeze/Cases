import pandas as pd
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster import hierarchy

df = pd.read_csv('autotrader_xml_sitemaps.csv', low_memory=False)
df['path'] = df['loc'].apply(lambda x: urlparse(x)[2])
df.to_dict(orient='list')
dfghj = pd.DataFrame()
dfghj = df['path'].str.split(pat='/',expand=True)
dfghj.columns = ['delete','1st','2nd','3rd','4th','5th','6th','7th']
dfghj = dfghj.drop('delete', axis=1)
dfghj.fillna('End',inplace=True)

#Визуализация будет выглядеть так, потому что ничего другого найти я не смог. Все древовидные диаграммы старнные, чисел нет, и т.д.
dfuniq = dfghj.groupby('1st')['2nd'].nunique().reset_index()
dfwhat = dfghj.groupby('1st')['2nd'].agg(secondpath ='unique').reset_index()
dfuniq.columns = ['Раздел','Количество разветвлений']
#print(dfuniq,'\n', dfwhat)


lastmod = pd.DataFrame()
lastmod['date'] = df['lastmod']
update = lastmod.merge(dfghj['1st'], left_index=True, right_index=True)

#Частота обновлений по отдельным разделам
chastupdate = update.groupby('1st')['date'].agg(count = 'count').reset_index()
chastupdate['count'] = chastupdate['count'] / 24
chastupdate.rename(columns={'count':'Частота обновлений в сутки'}, inplace=True)
#print(chastupdate)

#Какой контент присутствует в разных разделах
content = dfghj.loc[((dfghj['1st'])=='content')]
uniqcont = content.groupby('1st')['2nd'].agg(secondpath ='unique').reset_index()
#print(uniqcont.values)

#Рекомендации по контенту: можно дополнительно выкладывать новости, например с уменьшением цены на различные транспортные средства.
#Или например делать короткие видео с обзором поступивших транспортных средств на продажу
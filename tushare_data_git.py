import numpy as np
import pandas as pd
import tushare as ts
from icecream import ic
import datetime as dt

# config
pro = ts.pro_api('your api key')


# def get_stock_data(start, end):
#     ticker_list = []
#     ticker_df = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
#     data = pd.DataFrame()
#     for i in range(0,len(ticker_df[:]),10):
#         ic(i)
#         ticker_list = ticker_df[ticker_df['list_date'] > start]['ts_code'][i:i+10]
#         ticker_list = ','.join(ticker_list)
#         daily_df = pro.query('daily',ts_code=ticker_list,start_date=start,end_date=end)
#         data = pd.concat([data,daily_df],axis=0)
#     return data

def get_stock_data(start, end):
    ticker_list = ['603986.SH','603501.SH','603288.SH','603259.SH','601995.SH','601899.SH','601888.SH','601857.SH','601818.SH','601688.SH','601668.SH','601628.SH','601601.SH','601398.SH','601336.SH','601318.SH','601288.SH','601211.SH','601166.SH','601138.SH','601088.SH','601066.SH','601012.SH','600918.SH','600893.SH','600887.SH','600837.SH','600809.SH','600745.SH','600703.SH','600690.SH','600588.SH','600585.SH','600570.SH','600547.SH','600519.SH','600438.SH','600309.SH','600276.SH','600196.SH','600104.SH','600050.SH','600048.SH','600036.SH','600031.SH','600030.SH','600028.SH','600016.SH','600009.SH','600000.SH']
    ticker_list = ','.join(ticker_list)
    data = pd.DataFrame()
    daily_df = pro.query('daily',ts_code=ticker_list,start_date=start,end_date=end,fields='ts_code,trade_date,close')
    data = pd.concat([data,daily_df],axis=0)
    return data

df = pd.DataFrame()
for j in [str("%.2d" % i) for i in range(17,21)]:
    for i in [str("%.2d" % i) for i in range(1,13)]:
        start = '20'+j+i+'01'
        end = '20'+j+i+'31'
        df = pd.concat([df,get_stock_data(start,end)],axis=0)
df.columns = ['stock', 'date', 'close']
df = df.pivot_table(index=['date'], columns='stock', values='close')
df.index = df.index.map(lambda x:dt.datetime.strptime(str(x),'%Y%m%d'))


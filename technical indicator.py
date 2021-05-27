import numpy as np
import pandas as pd
import datetime as dt
from itertools import product
import yfinance as yf
from pandas_datareader import data as pdr
import time
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si

start_time = time.time()

# get data
yf.pdr_override()
indexes = '000001.SZ 002142.SZ 002807.SZ 002839.SZ 600000.SS 600015.SS 600016.SS 600036.SS 600908.SS 600919.SS ' \
          '600926.SS 601009.SS 601128.SS 601166.SS 601169.SS 601229.SS 601288.SS 601328.SS 601398.SS 601818.SS ' \
          '601939.SS 601988.SS 601997.SS 601998.SS 603323.SS 000001.SS'
indexes = indexes.split()  # this must be a list
data = pdr.get_data_yahoo(indexes, start="2010-09-01", end="2019-12-01", interval='1d')["Adj Close"]
# 10 years of data is enough to cover big economic cycles
df = pd.DataFrame(data=data)
df.fillna(method='ffill', inplace=True)
df.dropna(inplace=True, axis=1)  # drop column

# define parameters
sma1 = range(0, 41, 4)
sma2 = range(0, 281, 40)
rsi_period = [7, 14, 28]
rsi_oversold = [5, 10, 15, 20, 25]
rsi_overbought = [75, 80, 85, 90, 95]
stock_mapping_dict = {
    '000001.SZ': '平安银行',
    '002142.SZ': '宁波银行',
    '002807.SZ': '江阴银行',
    '002839.SZ': '张家港行',
    '600000.SS': '浦发银行',
    '600015.SS': '华夏银行',
    '600016.SS': '民生银行',
    '600036.SS': '招商银行',
    '600908.SS': '无锡银行',
    '600919.SS': '江苏银行',
    '600926.SS': '杭州银行',
    '601009.SS': '南京银行',
    '601128.SS': '常熟银行',
    '601166.SS': '兴业银行',
    '601169.SS': '北京银行',
    '601229.SS': '上海银行',
    '601288.SS': '农业银行',
    '601328.SS': '交通银行',
    '601398.SS': '工商银行',
    '601818.SS': '光大银行',
    '601939.SS': '建设银行',
    '601988.SS': '中国银行',
    '601997.SS': '贵阳银行',
    '601998.SS': '中信银行',
    '603323.SS': '吴江银行',
    '000001.SS': '上证综合指数'
}

# main logic for each asset and each technical indicators
all_results = pd.DataFrame()
for asset in df.columns:
    results = pd.DataFrame()
    # df[asset+'_return'] = np.log(df[asset] / df[asset].shift(1))
    df[asset + '_return'] = np.log(df[asset]).diff().shift(-1)
    # df.dropna(inplace=True) # when sum or mean, na will be treated as zero

    for SMA1, SMA2 in product(sma1, sma2):
        df['SMA1'] = df[asset].rolling(SMA1).mean()
        df['SMA2'] = df[asset].rolling(SMA2).mean()
        df['Position'] = np.where(df['SMA1'] > df['SMA2'], 1, -1)
        # df['Strategy'] = df['Position'].shift(1) * df[asset+'_return']
        df['Strategy'] = df['Position'] * df[asset + '_return']
        # perf = np.exp(df[[asset+'_return', 'Strategy']].sum())
        perf = np.exp(df[[asset + '_return', 'Strategy']].sum())
        results = results.append(pd.DataFrame(
            {'ASSET': asset, 'INDICATOR': 'MA', 'SMA1': SMA1, 'SMA2': SMA2,
             'MARKET': perf[asset + '_return'],
             'STRATEGY': perf['Strategy'], 'ALPHA': perf['Strategy'] - perf[asset + '_return']},
            index=[0]), ignore_index=True)

    for period, ob, os in product(rsi_period, rsi_overbought, rsi_oversold):
        df['Up'] = np.maximum(df[asset].diff(), 0)
        df['Down'] = np.maximum(-df[asset].diff(), 0)
        df['RS'] = df['Up'].rolling(period).mean() / df['Down'].rolling(period).mean()
        df['RSI'] = 100 - 100 / (1 + df['RS'])
        df['Position'] = np.where(df['RSI'] < os, 1, np.where(df['RSI'] > ob, -1, 0))
        df['Strategy'] = df['Position'] * df[asset + '_return']
        perf = np.exp(df[[asset + '_return', 'Strategy']].sum())
        results = results.append(pd.DataFrame(
            {'ASSET': asset, 'INDICATOR': 'RSI', 'PERIOD': period, 'OVERBOUGHT': ob, 'OVERSOLD': os,
             'MARKET': perf[asset + '_return'],
             'STRATEGY': perf['Strategy'], 'ALPHA': perf['Strategy'] - perf[asset + '_return']},
            index=[0]), ignore_index=True)

    all_results = all_results.append(results.loc[results['INDICATOR'] == 'MA'].sort_values('ALPHA', ascending=False).iloc[0], ignore_index=True)
    all_results = all_results.append(results.loc[results['INDICATOR'] == 'RSI'].sort_values('ALPHA', ascending=False).iloc[0], ignore_index=True)

# final touches
all_results.replace({'ASSET': stock_mapping_dict}, inplace=True)
all_results.sort_values(['INDICATOR', 'ALPHA'], ascending=False, inplace=True)
print(all_results)

# Pick the best stock
df['SMA1'] = df['601169.SS'].rolling(24).mean()
df['SMA2'] = df['601169.SS'].rolling(40).mean()
df['Position_MA'] = np.where(df['SMA1'] > df['SMA2'], 1, -1)
df['Up'] = np.maximum(df['601169.SS'].diff(), 0)
df['Down'] = np.maximum(-df['601169.SS'].diff(), 0)
df['RS'] = df['Up'].rolling(14).mean() / df['Down'].rolling(14).mean()
df['RSI'] = 100 - 100 / (1 + df['RS'])
df['Position_RSI'] = np.where(df['RSI'] < 20, 1, np.where(df['RSI'] > 75, -1, 0))
df['Position_SUM'] = df['Position_RSI'] + df['Position_MA']
df.replace({'Position_SUM': {2: 1, -2: -1}}, inplace=True)
df['Strategy'] = df['Position_SUM'] * df['601169.SS_return']
print(df[['601169.SS_return', 'Strategy']].sum())
plt.plot(df.index, df['601169.SS_return'].cumsum(), alpha=0.7)
plt.plot(df.index, df['Strategy'].cumsum(), alpha=0.7)
plt.plot(df.index, df['000001.SS_return'].cumsum(), alpha=0.7)
plt.legend(['Beijing Bank', 'ma+rsi strategy', '000001.SS'])
plt.title('in sample backtesting')
plt.grid()
plt.show()

'''
# below takes roughly 6 mins
fundamental = pd.DataFrame(columns=['stock', 'marketCap(B yuan)', 'heldPercentInstitutions', 'heldPercentInsiders'])
for i in indexes[:]:
    try:
        fundamental = fundamental.append(
            {'stock': i,
             'marketCap(B yuan)': int(yf.Ticker(i).info['marketCap']/(10**9)),
             'heldPercentInstitutions': yf.Ticker(i).info['heldPercentInstitutions'],
             'heldPercentInsiders': yf.Ticker(i).info['heldPercentInsiders']
             }, ignore_index=True
        )
    except:
        pass
fundamental.replace({'stock': stock_mapping_dict}, inplace=True)
fundamental.sort_values(['marketCap(B yuan)'], ascending=False, inplace=True)
print(fundamental)
'''

# test data set
data2 = pdr.get_data_yahoo(indexes, start="2010-09-01", interval='1d')["Adj Close"]
test = pd.DataFrame(data=data2)
test.fillna(method='ffill', inplace=True)
test.dropna(inplace=True, axis=1)  # drop column
for asset in test.columns:
    results = pd.DataFrame()
    test[asset + '_return'] = np.log(test[asset]).diff().shift(-1)
test['SMA1'] = test['601169.SS'].rolling(24).mean()
test['SMA2'] = test['601169.SS'].rolling(40).mean()
test['Position_MA'] = np.where(test['SMA1'] > test['SMA2'], 1, -1)
test['Up'] = np.maximum(test['601169.SS'].diff(), 0)
test['Down'] = np.maximum(-test['601169.SS'].diff(), 0)
test['RS'] = test['Up'].rolling(14).mean() / test['Down'].rolling(14).mean()
test['RSI'] = 100 - 100 / (1 + test['RS'])
test['Position_RSI'] = np.where(test['RSI'] < 20, 1, np.where(test['RSI'] > 75, -1, 0))
test['Position_SUM'] = test['Position_RSI'] + test['Position_MA']
test.replace({'Position_SUM': {2: 1, -2: -1}}, inplace=True)
test['Strategy'] = test['Position_SUM'] * test['601169.SS_return']
print(test[['601169.SS_return', 'Strategy']].sum())
plt.plot(test.index, test['601169.SS_return'].cumsum(), alpha=0.7)
plt.plot(test.index, test['Strategy'].cumsum(), alpha=0.7)
plt.plot(test.index, test['000001.SS_return'].cumsum(), alpha=0.7)
plt.legend(['Beijing Bank', 'ma+rsi strategy', '000001.SS'])
plt.grid()
plt.axvline(x=dt.datetime.strptime("12/1/2019", "%m/%d/%Y"), color='red')
plt.title('out sample backtesting after red line')
plt.show()

print("time elapsed: {:.2f}m".format((time.time() - start_time) / 60))

import numpy as np
import pandas as pd
from icecream import ic
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import time
import math
from scipy import stats
import warnings
import tushare_data

warnings.filterwarnings('ignore')
start_time = time.time()
ic.enable()
ic.configureOutput(includeContext=True)

def cointegration_test(y, x):
    # Step 1: regress on variable on the other
    ols_result = sm.OLS(y, x).fit()
    # Step 2: obtain the residual (ols_resuld.resid)
    # Step 3: apply Augmented Dickey-Fuller test to see whether
    #        the residual is unit root
    return adfuller(ols_result.resid)

def find_good_pairs(df):
    #df = df.fillna(method='ffill')
    #df = df.fillna(method='bfill')
    dim = df.shape[1]  # number of columns
    #pvalue_matrix = np.ones((dim, dim))
    #correl_matrix = np.zeros((dim, dim))
    keys = df.keys()  # index object of df columns
    good_pairs = []
    short = []
    long = []
    for i in range(dim):
        for j in range(i + 1, dim):
            try:
                stock1 = df[keys[i]]  # first stock
                stock2 = df[keys[j]]  # second stock
                # correlation is about magnitude in short time
                correl = np.corrcoef(stock1,stock2)[0,1]
                # cointegration is about possibility if stationary over long time
                #pvalue = coint(stock1, stock2)[1]
                pvalue = cointegration_test(stock1,stock2)[1]
                #pvalue_matrix[i, j] = pvalue
                #correl_matrix[i, j] = correl
                if pvalue < coint_param and correl > corr_param:
                    good_pairs.append((keys[i], keys[j]))
                    diff = stock1-stock2
                    rmean = diff.rolling(rmeanwindow).mean()[-1]
                    #rmean = diff[-2]
                    std = np.std(diff[-rmeanwindow:])
                    if diff[-1] > rmean+2*std:# and diff[-1] < rmean+3*std:
                        print(f'long {keys[i]}, short {keys[j]}, corr is {correl}, coint is {pvalue}')
                        if style == 'reversal':
                            #if stock1[-1] < stock1[-2]:
                            short.append(keys[i])
                            #if stock2[-1] > stock2[-2]:
                            long.append(keys[j])
                        if style == 'trend':
                            if (stock1[-1]-stock1[-5])/stock1[-5] < buyhighlimit:
                                long.append(keys[i])
                            short.append(keys[j])
                    elif diff[-1] < rmean-2*std:# and diff[-1] > rmean-3*std:
                        print(f'long {keys[j]}, short {keys[i]}, corr is {correl}, coint is {pvalue}')
                        if style == 'reversal':
                            #if stock2[-1] < stock2[-2]:
                            short.append(keys[j])
                            #if stock1[-1] > stock1[-2]:
                            long.append(keys[i])
                        if style == 'trend':
                            if (stock2[-1] - stock2[-5]) / stock2[-5] < buyhighlimit:
                                long.append(keys[j])
                            short.append(keys[i])
            except:
                pass
    return good_pairs, set(short), set(long)


# today_pos = get_position(stock_list, yesterday, close, yesterday_amt, 100)
def get_position(stock_list_buy,stock_list_sell, yesterday, close, yes_amount):
    single_day_close = close[close.index == yesterday].transpose()
    single_day_close['position'] = yes_amount / max(len([stock_list_buy]+[stock_list_sell]),1)/turnoveradj / single_day_close[yesterday]
    single_day_close.loc[single_day_close.index.isin(stock_list_buy), 'position'] = single_day_close.loc[single_day_close.index.isin(stock_list_buy), 'position']*1
    single_day_close.loc[single_day_close.index.isin(stock_list_sell), 'position'] = single_day_close.loc[single_day_close.index.isin(stock_list_sell), 'position']*(-1)
    single_day_close.loc[~single_day_close.index.isin(list(stock_list_buy)+list(stock_list_sell)), 'position'] = 0
    # in case of inf
    single_day_close.replace([np.inf, -np.inf], np.nan, inplace=True)
    single_day_close['position'] = single_day_close['position'].fillna(0)
    # # position round at 200 when ticker start with 688
    # single_day_close.loc[single_day_close.index.map(lambda x: '688' == str(x)[:3]), 'position'] \
    #     = (single_day_close.loc[single_day_close.index.map(lambda x: '688' == str(x)[:3]), 'position'] / 200).astype(int) * 200
    # # position round at 200 when ticker start with 300
    # single_day_close.loc[single_day_close.index.map(lambda x: '300' == str(x)[:3]), 'position'] \
    #     = (single_day_close.loc[single_day_close.index.map(lambda x: '300' == str(x)[:3]), 'position'] / 200).astype(int) * 200
    # position round at 100
    single_day_close['position'] = np.floor(single_day_close['position'] / 100)* 100
    return single_day_close['position']

# position_change, turnover_amt = change_position(yesterday_pos, today_pos, close, date, vwap)
def change_position(today_pos, tomorrow_pos, close, date, vwap):
    sub = pd.DataFrame(tomorrow_pos - today_pos)
    sub.columns = ['position']
    sub['sell'] = sub['position']
    sub['buy'] = sub['position']
    # change sign make sure both are positive
    sub.loc[sub['buy'] <= 0, 'buy'] = 0
    sub.loc[sub['sell'] >= 0, 'sell'] = 0
    sub['sell'] = -sub['sell']
    single_day_vwap = vwap[vwap.index == date].transpose()
    turnover_amt = (sub['buy'] * single_day_vwap[date] + sub['sell'] * single_day_vwap[date]).sum()
    return sub[['buy', 'sell']], turnover_amt

# today_rtn, amount = get_return(yesterday_amt, today_pos, date, today, close, vwap)
def get_return(yesterday_amt, today_pos, yesterday, date, close, vwap):
    # using close
    #    pre_close = close[close.index==yesterday].transpose()
    #    close_single = close[close.index==date].transpose()
    #    daily_profit = close_single[date]-pre_close[yesterday]
    # using vwap
    vwap_yesterday = vwap[vwap.index == yesterday].transpose()
    vwap_today = vwap[vwap.index == date].transpose()
    if yesterday == date:  # last day using close to sell
        vwap_today = close[close.index == date].transpose()
    daily_profit = vwap_today[date] - vwap_yesterday[yesterday]
    profit = (daily_profit * today_pos).sum()
    rtn = profit / yesterday_amt
    return rtn, profit + yesterday_amt

# functions define
############################################################
# main

close = pd.read_csv('close_price.csv', index_col=0)
close = close.loc[:,close.columns.isin(['603986.SH','603501.SH','603288.SH','603259.SH','601995.SH','601899.SH','601888.SH','601857.SH','601818.SH','601688.SH','601668.SH','601628.SH','601601.SH','601398.SH','601336.SH','601318.SH','601288.SH','601211.SH','601166.SH','601138.SH','601088.SH','601066.SH','601012.SH','600918.SH','600893.SH','600887.SH','600837.SH','600809.SH','600745.SH','600703.SH','600690.SH','600588.SH','600585.SH','600570.SH','600547.SH','600519.SH','600438.SH','600309.SH','600276.SH','600196.SH','600104.SH','600050.SH','600048.SH','600036.SH','600031.SH','600030.SH','600028.SH','600016.SH','600009.SH','600000.SH'])]
close.index = close.index.map(lambda x:dt.datetime.strptime(x,'%Y-%m-%d'))
close = tushare_data.df
close = close[:] # for backtesting speed
close = close.dropna(axis=1)

vwap = pd.read_csv('vwap_price.csv', index_col=0)
vwap = vwap.loc[:,vwap.columns.isin(['603986.SH','603501.SH','603288.SH','603259.SH','601995.SH','601899.SH','601888.SH','601857.SH','601818.SH','601688.SH','601668.SH','601628.SH','601601.SH','601398.SH','601336.SH','601318.SH','601288.SH','601211.SH','601166.SH','601138.SH','601088.SH','601066.SH','601012.SH','600918.SH','600893.SH','600887.SH','600837.SH','600809.SH','600745.SH','600703.SH','600690.SH','600588.SH','600585.SH','600570.SH','600547.SH','600519.SH','600438.SH','600309.SH','600276.SH','600196.SH','600104.SH','600050.SH','600048.SH','600036.SH','600031.SH','600030.SH','600028.SH','600016.SH','600009.SH','600000.SH'])]
vwap.index = vwap.index.map(lambda x:dt.datetime.strptime(x,'%Y-%m-%d'))
vwap = vwap[:] # for backtesting speed


date_list = list(close.index)[1:]
init_pos = close.transpose()[date_list[0]]
init_pos.values[:] = 0
today_pos = init_pos
position_change_list_buy = []
position_change_list_sell = []
rtn_list = []
amount_list = []
turnover_amt_list = []

print('########## config ############')
amount = 1.0e6
cointwindow = 200
rmeanwindow = 20
rebalance = 3 # every signal is evaluated for x days return
turnoveradj = 1 * rebalance
coint_param = 0.05
corr_param = 0.6
buyhighlimit = 0.2
style = 'trend'
print(f'amount:{amount} cointwindow:{cointwindow} rmeanwindow:{rmeanwindow} rebalance:{rebalance} turnoveradj:{turnoveradj} coint_param:{coint_param} corr_param:{corr_param} style:{style} buyhighlimit:{buyhighlimit}')


### main loop ###
ic('start main loop')
for i, date in enumerate(date_list[:]):
    if i >= cointwindow:
        print(f'do trading after cointwindow {cointwindow} + {i-cointwindow} days')
        # yesterday's position and amount
        yesterday_pos = today_pos
        yesterday_amt = amount
        yesterday = date_list[i] # use today close to establish position
        try:
            tomorrow = date_list[i + rebalance] # use vwap diff of tomorrow - today to get return
        except:  # last day
            tomorrow = date
        # print('yesterday is ',yesterday,' and today is ',date)
        # use yesterday's signal to pick stock

        #if i % rebalance == 0:
        good_pairs, short, long = find_good_pairs(close.iloc[i-cointwindow:i])
        print(f'long stock {long} and short stock {short}')
        print("time elapsed for finding pairs: {:.2f}m".format((time.time() - start_time) / 60))
        today_pos = get_position(long,short, yesterday, close, yesterday_amt)
        #today_rtn, amount = get_return(yesterday_amt, today_pos, date, tomorrow, close, vwap)
        today_rtn, amount = get_return(yesterday_amt, today_pos, date, tomorrow, close, close)

        if math.isnan(today_rtn) and math.isnan(amount):
            print('something went wrong!')
            today_rtn = 0
            amount = yesterday_amt
            today_pos = yesterday_pos
            turnover_amt = 0

        rtn_list.append(today_rtn)
        amount_list.append(amount)
        print(date, f'todays + {rebalance} return % is ', today_rtn*100, 'equity worth of ', amount)
        #position_change, turnover_amt = change_position(yesterday_pos, today_pos, close, date, vwap)
        position_change, turnover_amt = change_position(yesterday_pos, today_pos, close, date, close)
        position_change_list_buy.append(position_change['buy'])
        position_change_list_sell.append(position_change['sell'])
        turnover_amt_list.append(turnover_amt)

########################### records ###################################
buy_df = pd.concat(position_change_list_buy, axis=1)
buy_df.columns = date_list[cointwindow:]
sell_df = pd.concat(position_change_list_sell, axis=1)
sell_df.columns = date_list[cointwindow:]
pnl_record = pd.DataFrame({'return': rtn_list,
                           'exposure': amount_list,
                           'turnover_amount': turnover_amt_list})
pnl_record.index = date_list[cointwindow:]

print('buy record here: \n', buy_df)
print('sell record here: \n', sell_df)
print('pnl record here: \n', pnl_record)
# pnl_record.to_csv('pnl_record.csv')
############################ results ###################################
#2019
pnl_2019 = pnl_record[('2019-01-01'<=pnl_record.index)&(pnl_record.index<='2020-01-01')]
#收益率
return2019 = stats.gmean(pnl_2019['return']+1)
#return2019 = np.mean(pnl_2019['return']+1)
print('2019平均每日收益% is :',(return2019-1)*100)
#胜率
winpercentage2019 = (pnl_2019['return']>0).sum()/len(pnl_2019[pnl_2019['return']!=0])
print('2019胜率% is :',winpercentage2019*100)
#换手率 (includes selling previous and buying new)
turnover_rate2019 = (pnl_2019['turnover_amount']/pnl_2019['exposure']).mean()
print('2019换手% is :',turnover_rate2019*100)
print(f"样本中下单比例%: {pnl_2019[pnl_2019['return']!=0].shape[0]/max(len(pnl_2019),1)*100}")
print(f"最大回撤%: {(pnl_2019/pnl_2019.rolling(cointwindow,min_periods=1).max()-1).rolling(cointwindow,min_periods=1).min()['exposure'].min()*100}")
print(f"最大当日跌幅%: {pnl_2019['return'].min()*100}")
print(f"总收益%: {(pnl_2019['exposure'].tail(1).values[0]/float(amount)-1)*100}")

# 2020
pnl_2020 = pnl_record[('2020-01-01' <= pnl_record.index) & (pnl_record.index <= '2021-01-01')]
# 收益率
return2020 = stats.gmean(pnl_2020['return']+1)
#return2020 = np.mean(pnl_2020['return'])
print('2020平均每日收益% is :', (return2020-1)*100)
# 胜率
winpercentage2020 = (pnl_2020['return'] > 0).sum() / len(pnl_2020[pnl_2020['return']!=0])
print('2020胜率% is :', winpercentage2020*100)
#换手率 (includes selling previous and buying new)
# this is total buy sell amount / total money in account
turnover_rate2020 = (pnl_2020['turnover_amount'] / pnl_2020['exposure']).mean()
print('2020换手% is :', turnover_rate2020*100)
print(f"样本中下单比例%: {pnl_2020[pnl_2020['return']!=0].shape[0]/max(len(pnl_2020),1)*100}")
print(f"最大回撤%: {(pnl_2020/pnl_2020.rolling(cointwindow,min_periods=1).max()-1).rolling(cointwindow,min_periods=1).min()['exposure'].min()*100}")
print(f"最大当日跌幅%: {pnl_2020['return'].min()*100}")
print(f"总收益%: {(pnl_2020['exposure'].tail(1).values[0]/float(amount)-1)*100}")

# main
##############################################################
# graphical analysis

# aa = close['600893.SH']
# bb = close['600196.SH']
# diff = aa - bb
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()
# ax1.plot(aa,label='aa',color='orange')
# ax2.plot(bb,label='bb')
# plt.xticks(rotation=25)
# ax1.legend(loc=1)
# ax2.legend(loc=2)
# plt.grid()
# plt.title('cointegration')
# plt.show()
#
# fig = plt.figure()
# plt.plot(diff,label='diff')
# plt.plot(diff.rolling(rmeanwindow).mean(),label=f'rolling {rmeanwindow}')
# plt.xticks(rotation=25)
# plt.legend()
# plt.grid()
# plt.title('signal generate')
# plt.show()

# plt.figure()
# plot_acf(diff,lags=30)
# plt.show()
# plot_pacf(diff,lags=30)
# plt.show()

## stock = '3188.HK'
## indexdata = yf.download(stock, '2020-11-01', '2021-01-01', group_by='ticker')
## indexdata['return'] = indexdata['Adj Close'].pct_change(1).shift(-1)
## plt.plot((1+indexdata['return']).cumprod()-1,label='000300')
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()
# ax1.plot(pnl_record['exposure'],label='strategy',color='orange') # already compounding due to reinvestment capital
# #ax2.plot(close[close.index>=pnl_record.index[0]]['600519.SH'],label='maotao')
# ax2.plot(close[close.index>=pnl_record.index[0]]['601318.SH'],label='pingan')
# plt.grid()
# ax1.legend(loc=1)
# ax2.legend(loc=2)
# plt.xlabel('date')
# plt.ylabel('return')
# plt.xticks(rotation=25)
# plt.title('Backtest')
# plt.show()

fig = plt.figure()
plt.plot(pnl_record['exposure'],label='strategy') # already compounding due to reinvestment capital
plt.plot(close[close.index>=pnl_record.index[0]]['600519.SH']*(1000000/700),label='maotai')
plt.plot(close[close.index>=pnl_record.index[0]]['601318.SH']*(1000000/65),label='pingan')
plt.grid()
plt.legend()
plt.xlabel('date')
plt.ylabel('return')
plt.xticks(rotation=25)
plt.title('Backtest')
plt.show()

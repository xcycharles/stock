import numpy as np
import pandas as pd
from icecream import ic
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import time
import math
from scipy import stats
from scipy.stats import rankdata
from Alpha_101_GTJA_191.a101Alpha_code_1 import *
from tushare_data import *
import warnings
warnings.filterwarnings('ignore')
start_time = time.time()

res = {}
res_median = {}
for decay in [1,3,5,10]:
    print(f'decay is {decay}')
    print("time elapsed: {:.2f}m".format((time.time() - start_time) / 60))
    all_df = get_alpha(df,decay)
    res["res_{}".format(decay)] = all_df
    res_median["decay_{}".format(decay)] = all_df.filter(regex='alpha').median().dropna().sort_values(ascending=False)

ic_df = pd.DataFrame()
for i,j in res_median.items():
    ic_df[i]=j.sort_index
import numpy as np
import scipy as stats
import pandas as pd
import statsmodels.api as sm
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf
import time
from scipy.stats.mstats import winsorize
import itertools
import sklearn
start_time = time.time()

# initiating stuff

# It doesn't help a lot to write this in classes and OOP since most operations are related to data
final_result = pd.DataFrame([]) # return
final_result1 = pd.DataFrame([]) # asset pricing
final_result2 = pd.DataFrame([]) # risk premium
# downloading from yahoo could make the script timeout, so do it run at start
stock = '^GSPC'
index = yf.download(stock, '1983-12-01', '2017-07-01', group_by='ticker', interval='1mo')
index.loc[:, 'return'] = np.log(index['Adj Close']).diff().shift(-1)
index = index[:-1]

# the main loop

#for f in ['feps','fep','disp','pead','nsi','accrual','noa','ag','ia']:
for f in ['feps','noa','ag']:

    # Step 1: prepare data

    df = pd.read_stata('dataset19842018.dta')
    df_return = pd.read_stata('famafrench19842018.dta')
    df[f].dropna(axis=0,how='any', inplace=True) # check by df.isnull().sum()
    df['mv_eq'].dropna(axis=0,how='any', inplace=True) # market value
    df['siccd'].dropna(axis=0,how='any', inplace=True) # industry
    df['size_quantile'] = df.groupby(by=['yearmonth'])['mv_eq'].transform( \
        lambda x: pd.qcut(x, q=[0,.2,.4,.6,.8,1],labels=[0,1,2,3,4])) # groupby period must match forward return period to check size and factor periodically
    df[f+'_quantile'] = df.groupby(by=['yearmonth','size_quantile'])[f].transform( \
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop'))
    df[f+'_quantile'].dropna(axis=0,how='any', inplace=True) # due to dropping duplicated bin edges creats dups
    df['industry_'+f+'_quantile'] = df.groupby(by=['yearmonth','naics'])[f].transform( \
        lambda x: pd.qcut(x, q=[0,.2,.4,.6,.8,1], labels=False, duplicates='drop'))
    df['industry_'+f+'_quantile'].dropna(axis=0,how='any', inplace=True) # due to dropping duplicated bin edges creats dups

    # Step 2: inspect factor

    # make sure it is not garbage in garbage out
    plt.figure(figsize=(10,5))
    plt.hist(winsorize(df.groupby(by=['naics'])[f].aggregate([np.mean]).iloc[:,0],limits=[0.1,0.1]),bins=100,alpha=0.3)
    plt.hist(winsorize(df.groupby(by=['naics'])[f].aggregate([np.var]).iloc[:,0],limits=[0.1,0.1]),bins=100,alpha=0.3)
    plt.title(f'{f} factor inspection across industry')
    plt.legend([f+' mean',f+' variance'])
    plt.grid()
    plt.show()


    # Step 3: all time series long short return analysis
    # with size impact on factor

    longshort = pd.DataFrame([],index=[f+' LS mean %',f+' tstat'])
    for i in range(0,5):
        size_factor_year = df.groupby(by=['yearmonth','size_quantile',f+'_quantile'])['ret_f0f1'].mean().reset_index()
        low = size_factor_year.loc[(size_factor_year['size_quantile'] == i) & (size_factor_year[f+'_quantile'] == 0), :]
        high = size_factor_year.loc[(size_factor_year['size_quantile'] == i) & (size_factor_year[f+'_quantile'] == 4), :]
        result = pd.merge(low, high, on='yearmonth', suffixes=('_low', '_high'))
        result.dropna(axis=0, how='any', inplace=True) # not every month we have factor return "result.isnull().sum()"
        m = (result['ret_f0f1_high'] - result['ret_f0f1_low']).mean()
        t = (result['ret_f0f1_high'] - result['ret_f0f1_low']).mean() / \
        ((result['ret_f0f1_high'] - result['ret_f0f1_low']).std() / np.sqrt(len(result)))
        longshort['size '+str(i)]=[m*100, t]

    # with industry impact on factor

    size_factor_year = df.groupby(by=['yearmonth', 'industry_'+f+'_quantile'])['ret_f0f1'].mean().reset_index()
    low = size_factor_year.loc[size_factor_year['industry_'+f+'_quantile'] == 0, :]
    high = size_factor_year.loc[size_factor_year['industry_'+f+'_quantile'] == 4, :]
    result = pd.merge(low, high, on='yearmonth', suffixes=('_low', '_high'))
    result.dropna(axis=0, how='any', inplace=True)  # not every month we have factor return "result.isnull().sum()"
    m = (result['ret_f0f1_high'] - result['ret_f0f1_low']).mean()
    t = (result['ret_f0f1_high'] - result['ret_f0f1_low']).mean() / \
        ((result['ret_f0f1_high'] - result['ret_f0f1_low']).std() / np.sqrt(len(result)))
    longshort['by industry'] = [m * 100, t]

    # without size or industry
    size_factor_year = df.groupby(by=['yearmonth',f+'_quantile'])['ret_f0f1'].mean().reset_index()
    low = size_factor_year.loc[size_factor_year[f+'_quantile'] == 0, :]
    high = size_factor_year.loc[size_factor_year[f+'_quantile'] == 4, :]
    result = pd.merge(low, high, on='yearmonth', suffixes=('_low', '_high'))
    result.dropna(axis=0, how='any', inplace=True)  # not every month we have factor return "result.isnull().sum()"

    final_result = pd.concat([final_result, longshort], axis=0)

    # Step 4: Do risk adjustment in FF3F framework
    # convert factor values to pure factor returns (without size or industry) by using long short just like ff3f

    factor = pd.DataFrame({'yearmonth': result['yearmonth'],f: result['ret_f0f1_high'] - result['ret_f0f1_low']})
    df_return = pd.merge(df_return, factor, on='yearmonth')
    model = RollingOLS.from_formula(f'{f}~mktrf+smb+hml',data=df_return,window=12).fit() # 1 year rolling
    asset_pricing = pd.DataFrame([],index=[f+' exposure',f+' tstat'])
    asset_pricing['alpha'] = [np.mean(model.params['Intercept']),np.mean(model.tvalues['Intercept'])]
    asset_pricing['beta_mktrf'] = [np.mean(model.params['mktrf']),np.mean(model.tvalues['mktrf'])]
    asset_pricing['beta_smb'] = [np.mean(model.params['smb']),np.mean(model.tvalues['smb'])]
    asset_pricing['beta_hml'] = [np.mean(model.params['hml']),np.mean(model.tvalues['hml'])]
    final_result1 = pd.concat([final_result1, asset_pricing], axis=0)

    # Step 5: Use fama macbeth to analyze risk premium across time

    # preparation

    # get market returns for each asset
    df_macbeth = df_return
    for i in df['cusip'].unique()[:]:
        right = df.loc[df['cusip'] == str(i), ['yearmonth', 'ret_f0f1']]
        right.rename(columns={'ret_f0f1': str(i) + '_return'}, inplace=True)
        if len(right) == 420: # it is as trade of between number of assets (around 15 stocks) in 2nd regression and len(df_macbeth)
            df_macbeth = pd.merge(df_macbeth, right, how='inner', on='yearmonth') # note df_macbeth dates might not be continuous
    # change from series to array
    factors = df_macbeth[['mktrf', 'smb', 'hml', f]]
    riskfree = df_macbeth['rf']
    assets = df_macbeth.loc[:, df_macbeth.columns.str.contains('return')]
    # change from series to matrix, as will do matrix calculations later
    factors = np.mat(factors)
    riskfree = np.mat(riskfree)
    assets = np.mat(assets)
    # match the shapes
    T, K = factors.shape  # 432 months by factors
    T, N = assets.shape  # 432 months by assets
    riskfree.shape = T, 1  # change to 432by1 from 1by432
    excessReturns = assets - riskfree  # 432 months by assets

    # Time series regressions
    ts_reg = sm.OLS(excessReturns, sm.add_constant(factors)).fit()
    alpha = ts_reg.params[0]  # 1 row as alpha by assets
    beta = ts_reg.params[1:]  # factors by assets

    # Cross-section regression
    # do second pass macbeth regression between asset beta and asset mean returns in each time period as beta is not constant
    # risk premium has unit of return which is higher than risk free
    gamma = []
    for i in np.arange(start=0,stop=len(excessReturns)-12,step=1): # yearly rolling
        avgExcessReturns = np.mean(excessReturns[i:i+12], axis=0)
        # not need constant term for intercept
        cs_reg = sm.OLS(avgExcessReturns.T, beta.T).fit()
        gamma.append(cs_reg.params)
    gamma = np.array(gamma)
    rolling_prefix = [0]*12
    df_macbeth['gamma'] = np.append(rolling_prefix,gamma[:,-1])

    # collect result
    premium = pd.DataFrame([],index=[f'{f} risk premium %',f'{f} tstat'])
    premium[f] = [np.mean(gamma,axis=0)[3]*100,(np.mean(gamma,0)/(np.std(gamma,0)/np.sqrt(gamma.shape[0])))[3]]
    final_result2 = pd.concat([final_result2, premium], axis=0)


    # Step 6 plotting and visualize result

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.plot(pd.to_datetime(df_return['yearmonth'],format='%Y%m'),model.params['Intercept'])
    ax1.plot(pd.to_datetime(df_return['yearmonth'],format='%Y%m'),model.params['mktrf'],color='green',alpha=0.3)
    ax1.plot(pd.to_datetime(factor['yearmonth'],format='%Y%m'),factor[f],color='black',alpha=0.5)
    # factor LS return and intercept should be consistently above zero in both market up and down times (beta neutral)
    # no significant alpha means no need to introduce more factors to explain asset pricing
    ax2 = ax1.twinx()
    ax2.plot(index['Adj Close'],color='orange',alpha=0.8)
    # risk premium tell when buying stocks with higher factor exposure earns higher return
    start = dt.datetime.strptime(df_macbeth['yearmonth'][0].astype(int).astype(str),'%Y%m')
    end = dt.datetime.strptime(df_macbeth['yearmonth'].iloc[-1].astype(int).astype(str),'%Y%m')
    ax1.plot(pd.date_range(start,end,periods=gamma.shape[0]),gamma[:,3],color='purple')
    ax1.legend(['alpha','beta',f'{f} L-S return',f'{f} risk premium'],loc=4)
    ax2.legend(['S&P500'],loc=2)
    ax1.axhline(y=0,color='black',alpha=0.5)
    plt.title(f'{f} factor full analysis')
    plt.grid()
    plt.show()

    # Step 7 backtesting

    # Using factor exposure

    plt.plot(pd.to_datetime(df_return['yearmonth'],format='%Y%m'),np.cumsum(df_macbeth[f]))
    plt.plot(index['return'].cumsum())
    plt.grid()
    plt.legend([f'{f} return','S&P500 return'])
    plt.title(f'factor {f} normal backtesting')
    plt.show()
    # sm.formula.ols(formula='pead_demean~ia_demean-1',data=df_test2).fit().params[0] # find orthogonal coefficient -0.0438

    # Using factor 择时策略 when risk premium is positive

    plt.plot(pd.to_datetime(df_return.loc[df_macbeth['gamma']>0,'yearmonth'],format='%Y%m'), \
             np.cumsum(df_macbeth.loc[df_macbeth['gamma']>0,f]))
    plt.plot(index['return'].cumsum())
    plt.grid()
    plt.legend([f'{f} return','S&P500 return'])
    plt.title(f'factor {f} backtesting with timing on positive risk premium')
    plt.show()

    print(final_result)
    print(final_result1)
    print(final_result2)
    print("time elapsed: {:.2f}m".format((time.time() - start_time)/60))


# Load machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# add classification label and split-out validation dataset
Y = df.dropna().loc[:,'ret_f0f1'].apply(lambda row: 1 if row > 0 else 0)
Y = Y.astype('category')
X = df.dropna().loc[:,['feps','disp','fep','pead','nsi','accrual','noa','ag','ia']]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X,Y)
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
print ("Ridge model:", pretty_print_coefs(ridge.coef_))

# best subset selection
X = df.dropna().loc[:,['feps','disp','fep','pead','nsi','accrual','noa','ag','ia']]
Y = df.dropna().loc[:,'ret_f0f1']

def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(Y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - Y) ** 2).sum()
    return {"model":regr, "RSS":RSS}


def getBest(k):
    tic = time.time()
    results = []
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc - tic), "seconds.")
    # Return the best model, along with some other useful information about the model
    return best_model

# Could take quite awhile to complete...
models_best = pd.DataFrame(columns=["RSS", "model"])
tic = time.time()
for i in range(1,8):
    models_best.loc[i] = getBest(i)
toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
print(models_best.loc[3, "model"].summary())
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

# Set up a 2x2 grid so we can look at 4 plots at once
plt.subplot(2, 2, 1)

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector
plt.plot(models_best["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector

rsquared_adj = models_best.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
aic = models_best.apply(lambda row: row[1].aic, axis=1)

plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_best.apply(lambda row: row[1].bic, axis=1)

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('BIC')
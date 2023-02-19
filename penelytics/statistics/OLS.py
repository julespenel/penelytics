# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:42:03 2017

@author: Jules Penel
"""

print(__doc__)

# ~~> Import relevant packages
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import statsmodels.api as sm
import statsmodels.graphics as ttt


def f_engle_test(res, lags=None):
    sm.stats.diagnostic.het_arch(res)
    oTable = PrettyTable(['Lag', 'LM', 'LM - Pvalue', 'F-value', 'F-Pvalue'])
    if lags is None:
        oTable.add_row([1]+np.round(sm.stats.diagnostic.het_arch(res), 4).tolist())
    else:
        for i in range(1, lags+1):
            oTable.add_row([i]+np.round(sm.stats.diagnostic.het_arch(res, maxlag=i), 4).tolist())
    print('== Engle test for ARCH effect ==')
    print(oTable)


def f_arma(Y, AR=0, MA=0, X=None, X_names=None, dDates=None):
    a = sm.tsa.ARMA(Y,(AR, MA), exog=X, dates=dDates)#, oDataset[0][:,0], freq='d')
    if not X_names is None: a.exog_names = X_names
    b = a.fit()
    print('\n', '#'*78, '\n', sep='')
    print(b.summary())
    print('\n', '#'*78, '\n', sep='')
    return(b)


def f_arima(Y, AR=0, I=0, MA=0, X=None, X_names=None, dDates=None):
    a = sm.tsa.ARIMA(Y,(AR, MA, I), exog=X, dates=dDates)#, oDataset[0][:,0], freq='d')
    a.fit()
    if not X_names is None: a.exog_names = X_names
    b= a.fit(maxlag=10)
    print('\n', '#'*78, '\n', sep='')
    print(b.summary())
    print('\n', '#'*78, '\n', sep='')
    return(b)


def f_sarimax(Y, ARIMA=(1,0,1), SARIMA=(2,1,1,1), X=None, X_names=None, dDates=None):
    a = sm.tsa.SARIMAX(Y,exog=X, dates=dDates,order=ARIMA,seasonal_order=SARIMA, hamilton_representation=True, simple_differencing=True)
    if not X_names is None: a.exog_names = X_names
    b = a.fit()
    print(b.summary())
    b.plot_diagnostics(figsize=gSize)
    return(b)


def f_markov_switching(Y, reg=2, AR=2, X=None, X_names=None, dDates=None,
                     bGraph=True,
                     switching_ar=True, switching_trend=True,
                     switching_exog=False, switching_variance=False):
    a = sm.tsa.MarkovAutoregression(Y, reg, AR, exog=X, dates=dDates,
                                    switching_ar=switching_ar,
                                    switching_trend=switching_trend,
                                    switching_exog=switching_exog,
                                    switching_variance=switching_variance)
    if not X_names is None: a.exog_names = [a.exog_names[0]] + X_names
    b = a.fit()
    print(b.summary())
    if bGraph:
        pred = b.predict()
        pred[pred < 0] *= 0
        err = np.abs(np.sqrt(Y[AR:]) - np.sqrt(pred))
        avgErr = np.mean(err); print('Avg ERR:',avgErr)

        plt.figure(figsize=gSize)
        plt.suptitle('Model\'s Fitness', fontsize=16)
        plt.subplots_adjust(hspace=0.35)

        plt.subplot(321, title='Comparison')
        plt.plot_date(dDates[AR:], Y[AR:], label='Actual', fmt='-')
        plt.plot_date(dDates[AR:], pred, label='Predict', fmt='-', color='red', linewidth=2)
        plt.axhline(0, color="black", linestyle="--", alpha=0.8)
        plt.ylabel('Dependent Variable')
        plt.xlabel('Time')
        plt.legend()

        plt.subplot(323, title='Prediction Error')
        plt.plot_date(dDates[AR:], err, label='Absolute Error', fmt='-')
        plt.hlines(avgErr,dDates[AR], dDates[-1], color='green', label='Mean ('+str(round(avgErr,4))+')', linewidth=2)
        plt.ylabel('Absolute Error')
        plt.xlabel('Time')
#        plt.ylim(0,0.04)
        plt.legend()

        sm.qqplot(b.resid, fit=True, line='s', ax=plt.subplot(325, title='QQ-plot Residuals'))

        for i in range(reg):
            plt.subplot(reg, 2, (2*i)+2,
                        title='Smoothed Marginal Prob. Regime ' + str(i))
#            plt.plot_date(dDates[AR:], b.smoother_results.filtered_marginal_probabilities[i], fmt='-')
            plt.plot_date(dDates[AR:], b.smoothed_marginal_probabilities[:,i], fmt='-')
            plt.ylim(0,1.1)
            plt.ylabel('Probability')
            plt.xlabel('Time')
    return b


def f_augmented_dickey_fuller(X, sNames):
    """
    == Augmented Dickey Fuller Test ==
    """
    oTable = PrettyTable(['Variable', 't-stat', 'p-value', 'used lag', 'nb obs',
                          'Crit. Val.:', '1%', '5%', '10%'])
    sCrit = ['1%', '5%', '10%']
    for i in range(X.shape[1]):
        dummy = sm.tsa.stattools.adfuller(X[:,i])
        oTable.add_row([sNames[i]] + [round(dummy[z],4) for z in range(4)] +
                       [''] + [round(dummy[4][s],4) for s in sCrit])
    print(oTable)


# def shapley_owen(data: pd.DataFrame, Y: str, X: list):
#     size = len(X)
#
#     code = lambda x: bin(x)[2:].zfill(size)
#     check = []
#     for i in range(1, 2**size):
#         # check.append([x for x, y in zip(X, code(i)) if y == '1'])
#         check.append([X[x] for x in range(size) if 2**x & i == 2**x])
#
#     check = [code(x) for x in range(1, 2**size)]

def shapley_owen(data: pd.DataFrame, Y: str, X: list, intercept=True):
    """
    Use binary numbers to process all possible combinations
    """
    data = pd.DataFrame(np.random.uniform(0, 100, (1000, 5)), columns=['Haha', 'Hoho', 'Huhu', 'Hihi', 'Hehe'])
    Y = 'Haha'
    X = ['Hoho', 'Huhu', 'Hihi', 'Hehe']
    intercept=True


    # scipy.special.binom
    # X = list(range(5))
    if intercept:
        data = sm.add_constant(data)
        X.insert(0, 'const')

    size = len(X)

    code = lambda x: bin(x)[2:].zfill(size)
    dict_rsquares = {x: {f'level {y+1}': {'with':[], 'without':[]} for y in range(size)} for x in range(size)}
    rsquared_contrib = {x: 0 for x in range(size)}
    params = {}
    for i in range(1, 2**size):
        params[i] = {}
        cur = code(i)
        level = cur.count('1')
        selection = [x for x, y in zip(X, cur) if y == '1']
        weight_with = (1/size) / scipy.special.binom(size - 1, level - 1)
        if level != size:
            weight_without = (1/size) / scipy.special.binom(size - 1, level)
        else:
            weight_without = 0

        # params[i]['selection'] = [x for x, y in zip(X, cur) if y == '1']
        # params[i]['weight'] = (1/size) / scipy.special.binom(size - 1, level - 1)

        ols = sm.OLS(endog=data[Y], exog=data[selection], hasconst='const' in selection).fit()

        for z, val in enumerate(cur):
            if val == '1':  # <~~ The variable IS included
                rsquared_contrib[z] += ols.rsquared * weight_with
                dict_rsquares[z][f'level {level}']['with'].append((cur, ols.rsquared, weight_with))
            else:  # <~~ The variable IS NOT included
                rsquared_contrib[z] -= ols.rsquared * weight_without
                dict_rsquares[z][f'level {level+1}']['without'].append((cur, ols.rsquared, weight_without))

        ####### # ~~> Might need this later, but for now, not needed
        # for z, val in enumerate(cur):
        #     if val == '1':  # <~~ The variable IS included
        #         dict_rsquares[z][f'level {level}']['with'].append((cur, ols.rsquared, weight_with))
        #     else:  # <~~ The variable IS NOT included
        #         dict_rsquares[z][f'level {level+1}']['without'].append((cur, ols.rsquared, weight_without))

        # check.append([x for x, y in zip(X, cur) if y == '1'])
        # check.append([X[x] for x in range(size) if 2**x & i == 2**x])





    weights = {code(i): 1/(code(i).count('1') * size) for i in range(1, 2**size)}
    scipy.special.binom(size - 1, code(i).count('1') - 1)

    # scipy.special.binom()


    # check = [code(x) for x in range(1, 2**size)]


def option_2():
    X = list(range(5))
    size = len(X)

    # code = lambda x: bin(x)[2:].zfill(size)
    check = []
    for i in range(1, 2**size):
        # check.append([X[x] for x in range(size) if 2**x & i == 2**x])
        for x in range(size):
            cur = 2**x
            if cur & i == cur:
                check.append(X[x])


    # check = [code(x) for x in range(1, 2**size)]


def main():
    import timeit
    print('Option 1:', timeit.timeit(option_1))
    print('Option 2:', timeit.timeit(option_2))
    print('Option 1:', timeit.timeit(option_1))
    print('Option 2:', timeit.timeit(option_2))

import cProfile
cProfile.run('main()', r'C:\Users\Jules Penel\Desktop\python stuff\PENELYTICS\checkspeed.pstat')


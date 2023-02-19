# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:42:03 2017

@author: ga2ples

Objective: Get partial autocorrelations
"""

print(__doc__)

# ~~> Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import statsmodels.api as sm

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def f_autocorrelation(oDATA, nLags=20, display_graph=True, print_table=True):
    """
    Analysis of autocorrelation and partial autocorrelation in data
    Prints detail of numeric values and display a summary graph
    :param oDATA:
    :param nLags:
    :return: acf, pacf, PrettyTable object
    """
    # ~~> Autocorrelogram - Table
    acf = sm.tsa.acf(oDATA, nlags=nLags, qstat=True)#, missing='drop')
    pacf = sm.tsa.pacf(oDATA, nlags=nLags)
    oTable = PrettyTable(['Lag', 'PAC', '', 'AC', 'Ljunb Box Q Stat', 'p-value'])
    for i in range(1, pacf[1:].shape[0]):
        oTable.add_row([i, round(pacf[i], 3), '', round(acf[0][i], 3), round(acf[1][i], 3), round(acf[2][i], 3)])
    if print_table:
        print('== Autocorrelogram ==')
        print(oTable)

    if display_graph:
        # ~~> Autocorrelogram - Graph
        plt.figure(figsize=(10, 6))
        plt.suptitle('Autocorrelogram', fontsize=16)
        sm.graphics.tsa.plot_pacf(oDATA, plt.subplot(222), lags=nLags, zero=False, auto_ylims=True)
        sm.graphics.tsa.plot_acf(oDATA, plt.subplot(224), lags=nLags, zero=False, auto_ylims=True)

        # ~~> Ljung - Box Test
        nblag = min(len(oDATA) / 2 - 2, 40)  # ~~> Copied Stata's default parameters
        plt.subplot(121, title='Ljung-Box\'s White Noise test')
        plt.plot(sm.stats.acorr_ljungbox(oDATA, lags=nblag)['lb_pvalue'])
        plt.xlabel('Lags')
        plt.ylabel('P-value')
        plt.hlines(0.05, 0, nblag, color='red', linewidth=2)
        plt.ylim(0, max(plt.ylim()[1], 0.1))
        plt.annotate(
            'If below we reject H0(Absence of autocorrelation)',
            xy=(nblag, 0.06),
            horizontalalignment='right',
            color='red'
        )

    return acf, pacf, oTable

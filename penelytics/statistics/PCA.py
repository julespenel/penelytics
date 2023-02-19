# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:57:43 2017

@author: GA2PLES

Objective: Principal Component Analysis of our extended list of factors
"""
print(__doc__)

# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# import statsmodels.api as sm


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def fPCA(oDATA, sTitles, n, b_graph=True):
    gSize = (10, 6)

    # ~~> Standardise the data
    oDATA = (oDATA - np.mean(oDATA, 0)) / np.std(oDATA, 0)
    sTitles += [('PC' + str(i + 1)) for i in range(n)]

    if b_graph:
        # ~~> Get the values for the graph
        a, b = np.linalg.eig(np.cov(oDATA.T))
        c = [(sTitles[i], np.abs(a[i]), b[:, i]) for i in range(a.shape[0])]
        c.sort(key=lambda x: x[1], reverse=True)

        # ~~> Create the graph
        plt.figure(figsize=gSize)
        plt.suptitle('Principal Component Analysis', fontsize=16)
        ExpVar = [100 * x[1] / sum(a) for x in c]
        plt.bar(range(oDATA.shape[1]), ExpVar)
        plt.plot(range(oDATA.shape[1]), [sum(ExpVar[:x + 1]) for x in range(len(c))], c='r', marker='s')
        plt.axhline(90, color="black", linestyle="--", label='90%', alpha=0.8)
        plt.axhline(95, color="black", linestyle="--", label='95%', alpha=0.8)
        plt.axhline(99, color="black", linestyle="--", label='99%', alpha=0.8)
        plt.ylabel('Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylim(0, 100)

    # ~~> Use SK learn to have a standardised output
    from sklearn.decomposition import PCA
    tt = PCA(n_components=n)
    Transformed = tt.fit_transform(oDATA)

    if b_graph:
        # Pull out the table of correlations
        print('== Correlations ==')
        oTable = PrettyTable(['Variable'] + [('PC' + str(i + 1)) for i in range(n)])
        dblCorr = np.corrcoef(oDATA.T, y=Transformed.T)[:, -n:]
        for i in range(dblCorr.shape[0]):
            oTable.add_row([sTitles[i]] + np.round(dblCorr[i, :], 2).tolist())
        print(oTable)

    return Transformed
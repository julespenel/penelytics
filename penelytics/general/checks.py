from fancychart.templates import standardize_graph
import statsmodels.api as sm
from penelytics.config import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def outlier_check(data: pd.DataFrame, return_graph=True, save_graph=False, label=''):
    """
    Check for outlier using boxplots
    For identified outliers, the index value is displayed
    """
    if label:
        label = f' - {label}'

    # ~~> Generate figure and produce box plot
    fig = plt.figure()
    sns.boxplot(data=data, notch=True, boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5))

    # ~~> Get threshold for outliers
    boundaries = data.quantile((0.25, 0.75))
    iqr = boundaries.diff().iloc[-1]
    boundaries.iloc[0] -= 1.5 * iqr  # <~~ Lower bound
    boundaries.iloc[1] += 1.5 * iqr  # <~~ Upper bound
    outlier_filter = ((data < boundaries.iloc[0]) | (data > boundaries.iloc[1]))

    # ~~> Stripplot all the non-outliers
    sns.stripplot(data=data[~outlier_filter], color=".3", size=3, alpha=0.5)

    # ~~> For the outlier, add a label
    for x in data.columns:
        for index, value in data[[x]][outlier_filter[x]].dropna().iterrows():
            plt.text(x, value, f'  {index}', ha='left', va='center', fontsize=6.5)

    # ~~> Standardize
    standardize_graph(format='single-text-width')

    if save_graph:
        plt.savefig(f'{config.TEMP_PATH}\\outlier check{label}.png', dpi=300)

    # ~~> Either return or just close and clean garbage
    if return_graph:
        return fig
    else:
        plt.close()


def distribution_check(data: pd.DataFrame, return_graph=True, save_graph=False):
    # if label:
    #     label = f' - {label}'

    per_year = data.groupby(['Heating Year', 'City Code']).sum()['Raw HDD'].unstack(1)
    f_normality_test(per_year, linewidth=0.5)
    standardize_graph()

    # ~~> Outliers checks
    fig = data.boxplot(fontsize=6.5, notch=True, boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5))
    standardize_graph(format='single-text-width')

    if save_graph:
        plt.savefig(f'{config.TEMP_PATH}\\outlier check.png', dpi=300)

    # ~~> Either return or just close and clean garbage
    if return_graph:
        return fig
    else:
        plt.close()


def get_best_layout_multi_chart(x) -> tuple:
    """
    Based on the screen size, output the optimal layout for multi plotting

    :param x: Number
    :return: Tuple
    """

    max_size = 9
    screen_ratio = 1920 / 1080
    best = 9999
    out = None
    for i in range(1, max_size + 1):
        if 0 <= (i * (i // screen_ratio)) - x < best:
            out = (int(i // screen_ratio), i)
            best = (i * (i // screen_ratio)) - x
        if 0 <= (i * (1 + i // screen_ratio)) - x < best:
            out = (int(1 + i // screen_ratio), i)
            best = (i * (1 + i // screen_ratio)) - x

    if out is None:
        out = (int(1 + i // screen_ratio), i)

    return out


def statistic_description(data: pd.DataFrame, subplot=None, show_legend=True):
    graph_data = data[data.notnull()].values
    kde = sm.nonparametric.KDEUnivariate(graph_data)
    kde.fit(kernel='gau')

    if subplot is None:
        plt.figure()
        subplot = plt.subplot(111)
    subplot.hist(graph_data, density=True, rwidth=0.95, label='True distribution')
    subplot.plot(kde.support, kde.density, label='Kernel Density (gaussian) from samples')
    subplot.set_title(data.name)
    # subplot
    if show_legend:
        subplot.legend(loc='best')
    return subplot




def distribution_overview(data: pd.DataFrame):
    layout = get_best_layout_multi_chart(data.shape[1])
    fig, subplots = plt.subplots(*layout,
                                 sharex=False,
                                 sharey=False)
    subplots = subplots.flatten()

    for i in range(min(data.shape[1], layout[0] * layout[1])):
        subplots[i] = statistic_description(data.iloc[:, i], subplots[i], False)

    fig.show()


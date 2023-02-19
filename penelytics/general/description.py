import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd


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


data_all = pd.read_csv(r'C:\Users\Jules Penel\Desktop\Econometrics\time_series_60min_singleindex.csv')
statistic_description(data_all.iloc[:, 2:18])
print('DONE')
# statistic_description(data_slice)

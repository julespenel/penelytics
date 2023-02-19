"""
Good stuff but it is ugly and there are better solutions out there
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from config import CONF
MAX_SIZE = 15

def correlogram(data, graph_size=None):
    if isinstance(data, pd.DataFrame):
        data = data[data.columns[data.dtypes != "object"]]  # <~~ Filter out object which will raise an error
        values = data.values
        xheaders = data.columns
        yheaders = data.index
    elif isinstance(data, np.ndarray):
        values = data
        xheaders = [x for x in range(data.shape[1])]
        yheaders = [x for x in range(data.shape[0])]
    else:
        raise ValueError('data is not a dataframe or an array')

    # if graph_size is None:
        # graph_size = CONF.GRAPH_SIZE
    nrows, ncols = values.shape
    graph_size = values.shape[::-1]
    ycanvas = 1/nrows
    xcanvas = 1/ncols
    width = xcanvas
    height = ycanvas


    fig = plt.figure(figsize=graph_size)
    # ax = fig.add_axes([0, 0, 1, 1])
    margins = 0.1
    ax = fig.add_axes([margins, margins, 1 - margins*2, 1 - margins*2])
    ax.tick_params(
        axis='both',
        direction='out',
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
    )

    # ~~> Draw the grid
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1/ncols))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1/nrows))
    ax.tick_params(
        axis='both',
        which='minor',
        grid_alpha=1,
        grid_linestyle='-',
        gridOn='both',
    )

    # ~~> Write the labels
    ax.xaxis.set_ticks([xcanvas/2 + xcanvas * x for x in range(ncols)])
    ax.yaxis.set_ticks([ycanvas/2 + ycanvas * x for x in range(nrows)])
    ax.set_xticklabels(xheaders)
    ax.set_yticklabels(yheaders[::-1])

    # ax.yaxis.set_ticks([0.10 * x for x in range(1, 11)])
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    # ax.xaxis.set_major_locator(plt.IndexLocator(base=0.1, offset=0.05))
    # ax.yaxis.set_major_locator(plt.IndexLocator(base=0.1, offset=0.05))

    # ax.grid(True)

    for row in range(1, nrows+1):
        for col in range(1, ncols+1):
            multiplier = (abs(values[row-1, col-1])**0.5) * 0.9
            if values[row-1, col-1] < 0:
                color = 'red'
            else:
                color = 'green'
            # ax.add_patch(plt.Rectangle(
            #     (row * x - width,
            #      col * y - height),
            #     width,
            #     height,
            #     fill=False
            # ))
            ax.add_patch(plt.Rectangle(
                (col * xcanvas - width/2 - (width/2 * multiplier),
                 row * ycanvas - height/2 - (height/2 * multiplier)),
                width * multiplier,
                height * multiplier,
                fill=True,
                facecolor=color,
                # capstyle='projecting'
                # joinstyle='bevel',
                # hatch='/\\|-'
            ))


correlogram(pd.DataFrame(np.random.uniform(-1, 1, size=(4, 5)), columns=['a', 'b', 'c', 'd', 'e']))


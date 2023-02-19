from penelytics.config import config
import matplotlib.pyplot as plt
from fancychart.templates import standardize_graph
import pandas as pd
import os

class AutoAnalyser:
    """
    This intends to analyse a provided dataset to provide a rapid analysis covering a wide range of area
    - Identification of missing datapoints
    - Check for extrems
    - Check of data distribution types
    """
    def __init__(self, data):
        self.data = pd.DataFrame()
        self._load_data(data)

        scan = data.applymap(type).apply(lambda x: x.value_counts())

        if isinstance(self.data.index, pd.RangeIndex):
            pass

        if self.data.index.is_all_dates:
            for col in self.data.columns:
                identify_gaps(self.data[col])


    def _load_data(self, data):
        # ~~> Load data
        if isinstance(data, str):
            self.data = pd.read_csv(f'{data}', parse_dates=True)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy(True)
        else:
            raise TypeError(f'data variable passed was neither a DataFrame or a string path, {type(data)} isn\'t valid')
        return self


def identify_gaps(data, plot_graph=True):
    if plot_graph:
        data.plot()

    nans = data.iloc[:, 0].isna()
    if nans.any():
        data[nans].scatter()
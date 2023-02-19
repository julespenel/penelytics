import pandas as pd
import matplotlib.pyplot as plt
from fancychart.templates.standardize_graphs import standardize_graph

PATH = r'C:\Users\Jules Penel\Desktop\python stuff\Gas data\GIE AGSI\gasInStorage.csv'

data = pd.read_csv(PATH, index_col=0, parse_dates=True)


sl = data.iloc[:, -1]

a = data.applymap(type).apply(lambda x: x.value_counts())

def string_type(x: str):
    try:
        int(x)
        return 'int_str'
    except ValueError:
        pass

    try:
        float(x)
        return 'float_str'
    except ValueError:
        pass




    if x.isnumeric():
        return 'int_str'
    elif x.is():
        return 'numeric_str'



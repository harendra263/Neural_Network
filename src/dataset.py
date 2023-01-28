import pandas as pd
import numpy as np


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)



dataset = raw_dataset.copy()

print(dataset.isna().sum())

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({
    1: "USA", 2: "EUROPE", 3: "JAPAN"
})


dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep=' ')

print(dataset.tail())
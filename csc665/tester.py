import pandas as pd
import numpy as np

csv_df = pd.read_csv("Income.csv")
csv_df.sort_values(['Education'], inplace=True)
print(csv_df.head())
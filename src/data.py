"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import os
import pandas as pd

DIR_PATH = os.path.join(os.path.dirname(__file__), "../data/")
FILES = ['games.csv', 'mechanics.csv', 'themes.csv', 'subcategories.csv']

def load_data():

    data = {}
    for f in FILES:
        k = f.replace('.csv', '')
        v = pd.read_csv(DIR_PATH + f)
        data[k] = v
    return data

def merge_data(dfs):
    df = dfs['games']
    df = df.merge(dfs['mechanics'], on="BGGId", how="left")
    df = df.merge(dfs['themes'], on="BGGId", how="left")
    df = df.merge(dfs['subcategories'], on="BGGId", how="left")
    return df

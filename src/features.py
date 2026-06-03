"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import pandas as pd


# numeric features available directly from games.csv
NUMERIC_FEATURES = [
    "GameWeight",
    "MinPlayers",
    "MaxPlayers",
    "MfgPlaytime",
    "NumWant",
    "NumExpansions",
    "YearPublished",
]


def get_features(data, target):

    if target == "AvgRating":
        
        # drop rows missing the target
        df = data['games'].dropna(subset=[target])

        X = df[NUMERIC_FEATURES]
        X = X.dropna()

        # align y to the same rows as X
        y = df.loc[X.index, target]

    return X, y

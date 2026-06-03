"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler


# numeric features available directly from games.csv
LR_FEATURES = [
    "GameWeight",
    "MinPlayers",
    "MaxPlayers",
    "MfgPlaytime",
    "NumExpansions",
    "YearPublished",
]
LR_PREPROCESSING = ["dropna", "StandardScaler"]
LR_FEATURE_ENGINEERING = []
LR_CONFIG = [LR_FEATURES, LR_PREPROCESSING, LR_FEATURE_ENGINEERING]

def get_features(data, target, model_name):

    if model_name == "Linear Regression":
        
        # drop rows missing the target
        df = data['games'].dropna(subset=[target])

        X = df[LR_FEATURES]
        X = X.dropna()

        # align y to the same rows as X
        y = df.loc[X.index, target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    return X_scaled, y

"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import numpy as np

from sklearn.preprocessing import StandardScaler



# for logger.py
features = []
preprocessing = []
feature_engineering = []
CONFIG = [features, preprocessing, feature_engineering]


# numeric features available directly from games.csv
LR_FEATURES = [
    "GameWeight",
    "MinPlayers",
    "MaxPlayers",
    "MfgPlaytime",
    "NumExpansions",
    "YearPublished",
]

LOG_FEATURES = ["MfgPlaytime", "NumExpansions", "MaxPlayers"]

CAT_GENRES = [
    "Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family",
    "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens"
]

def get_features(data, target, model_name):

    if model_name == "Linear Regression":
        
        # drop rows missing the target
        df = data.dropna(subset=[target])
        
        # select the features
        binary_cols = _get_binary_columns(df)
        X = df[LR_FEATURES + binary_cols].copy()

        # set empty binary columns to 0 (assume missing)
        X[binary_cols] = X[binary_cols].fillna(0)
        features.extend(X.columns.tolist())

        # drop rows with missing values   
        preprocessing.append("dropna")
        X = X.dropna()

        # align y to the same rows as X
        y = df.loc[X.index, target]

        # feature engineering
        feature_engineering.append("log1p(" + ", ".join(LOG_FEATURES) + ")")
        for col in LOG_FEATURES:
            X[col] = np.log1p(X[col])

        # scale features
        preprocessing.append("StandardScaler")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


    return X_scaled, y


# all binary columns from mechanics, themes, subcategories are 0/1
# grab them dynamically after merge
def _get_binary_columns(df):
    binary_cols = []
    for col in df.columns:
        if df[col].dropna().isin([0, 1]).all():
            binary_cols.append(col)
    return binary_cols


def get_genre_features(df, genre_col):

    # filter to games in this genre
    genre_df = df[df[genre_col] == 1].copy()

    # drop rows missing target or predictor
    genre_df = genre_df.dropna(subset=["AvgRating", "GameWeight"])

    X = genre_df[["GameWeight"]]
    y = genre_df["AvgRating"]

    return X, y
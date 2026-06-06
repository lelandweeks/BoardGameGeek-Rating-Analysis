"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""


import numpy as np
import pandas as pd

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

def create_rating_label(df):

    # drop the missing rows and create a temp df
    df_temp = df[['NumOwned', 'AvgRating']].dropna().copy()
    
    # combine rating and ownership into single value
    df_temp['rating_class'] = df_temp['AvgRating'] / df_temp['NumOwned']

    # bin into 3 classes
    # underrated and overrated and then
    # middle_rated is neither underrated or overrated
    # (tried this first but realized wrong order:
    # labels=['underrated', 'middle_rated', 'overrated'])
    df_temp['rating_class'] = pd.qcut(
        df_temp['rating_class'], q=3, 
        labels=['overrated', 'middle_rated', 'underrated']
    )
    
    # add the rating_class column back to a new df
    df = df.copy()
    df['rating_class'] = df_temp['rating_class']
    return df

def get_class_features(df):

    # for the logger
    _reset_config()

    # drop rows with missing values   
    preprocessing.append("dropna")
    df = df.dropna(subset=['rating_class'])

    binary_cols = _get_binary_columns(df)
    X = df[LR_FEATURES + binary_cols].copy()
    X[binary_cols] = X[binary_cols].fillna(0)
    X = X.dropna()

    # for the logger
    features.extend(X.columns.tolist())

    y = df.loc[X.index, 'rating_class']

    # scale features
    # kept getting warning "lbfgs failed to converge"
    # after increaseing max_iter=10000, decided to scale features
    preprocessing.append("StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def get_features(data, target, model_name):

    # for the logger
    _reset_config()

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

    # for the logger
    _reset_config()

    # filter to games in this genre
    genre_df = df[df[genre_col] == 1].copy()

    # drop rows missing target or predictor
    genre_df = genre_df.dropna(subset=["AvgRating", "GameWeight"])

    X = genre_df[["GameWeight"]]
    y = genre_df["AvgRating"]

    return X, y

def _reset_config():
    features.clear()
    preprocessing.clear()
    feature_engineering.clear()
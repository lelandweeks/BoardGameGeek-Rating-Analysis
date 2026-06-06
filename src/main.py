"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import argparse

from data import load_data, merge_data
from features import get_features, CONFIG, CAT_GENRES, get_genre_features
from models import run_regression, run_lasso, run_ridge
from evaluate import evaluate_regression, get_regression_metrics
from logger import log_run

OUTPUT_DIR = 'output/'
LASSO_ALPHA = 0.001
RIDGE_ALPHA = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    choices=["q1", "q2", "q3", "all"],
                    default="all")
args = parser.parse_args()

# load the data
print("Loading data...")
dfs = load_data()
df = merge_data(dfs)

print("Data Loaded:")
for key in dfs.keys():
    print(f"  {key}")
print()

# run selected models
models = ["q1", "q2", "q3"]
if args.model != "all":
    models = [args.model]

if "q2" in models:
    question = "Q2: Can we predict a game's average rating based on its features?"
    model_name = "Linear Regression"
    target = "AvgRating"
    print(question)
    
    X, y = get_features(df, target, model_name)

    # run linear regression
    print("Running Linear Regression...")
    y_test, y_pred = run_regression(X, y)
    metrics = get_regression_metrics(y_test, y_pred)
    evaluate_regression(metrics)
    print()
    log_run(
        question = question,
        model_name = model_name,
        features = CONFIG[0],
        preprocessing = CONFIG[1],
        feature_engineering= CONFIG[2],
        hyperparameters = None,
        results = metrics
    )

    # run lasso regression
    print("Running Lasso Regression...")
    y_test, y_pred = run_lasso(X, y, LASSO_ALPHA)
    metrics = get_regression_metrics(y_test, y_pred)
    evaluate_regression(metrics)
    print()
    log_run(
        question = question,
        model_name = "Lasso",
        features = CONFIG[0],
        preprocessing = CONFIG[1],
        feature_engineering = CONFIG[2],
        hyperparameters = {"alpha": LASSO_ALPHA},
        results = metrics
    )

    # run ridge regression
    print("Running Ridge Regression...")
    y_test, y_pred = run_ridge(X, y, alpha=RIDGE_ALPHA)
    metrics = get_regression_metrics(y_test, y_pred)
    evaluate_regression(metrics)
    print()
    log_run(
        question = question,
        model_name = "Ridge",
        features = CONFIG[0],
        preprocessing = CONFIG[1],
        feature_engineering = CONFIG[2],
        hyperparameters = {"alpha": RIDGE_ALPHA},
        results = metrics
    )


if "q3" in models:
    question = "Q3: Does complexity predict rating within a genre?"
    model_name = "Linear Regression"
    target = "AvgRating"
    print(question)


    print("")

    for genre in CAT_GENRES:
        X, y = get_genre_features(df, genre)

        y_test, y_pred = run_regression(X, y)
        metrics = get_regression_metrics(y_test, y_pred)
        print(f"\n{genre}")
        evaluate_regression(metrics)
        print()
        log_run(
            question           = question,
            model_name         = model_name,
            features           = ["GameWeight"],
            preprocessing      = ["dropna"],
            feature_engineering= CONFIG[2],
            hyperparameters    = {"genre": genre},
            results            = metrics
        )
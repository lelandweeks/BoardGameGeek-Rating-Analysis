"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import argparse

from data import load_data
from features import get_features, LR_CONFIG
from models import run_regression
from evaluate import evaluate_regression, get_regression_metrics
from logger import log_run

OUTPUT_DIR = 'output/'

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    choices=["q1", "q2", "q3", "all"],
                    default="all")
args = parser.parse_args()

# load the data
print("Loading data...")
dfs = load_data()

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
    print("Running Linear Regression...")

    X, y = get_features(dfs, target, model_name)
    y_test, y_pred = run_regression(X, y)

    metrics = get_regression_metrics(y_test, y_pred)
    evaluate_regression(metrics)

    print()
    log_run(
        question = question,
        model_name = model_name,
        features = LR_CONFIG[0],
        preprocessing = LR_CONFIG[1],
        feature_engineering= LR_CONFIG[2],
        results = metrics
    )

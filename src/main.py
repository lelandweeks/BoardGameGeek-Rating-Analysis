"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import argparse

from data import load_data
from features import get_features
from models import run_regression
from evaluate import evaluate_regression

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
    target = "AvgRating"
    print("What features make a boardgame enjoyable?")
    print("Running Linear Regression...")
    X, y = get_features(dfs, target)
    results = run_regression(X, y)
    evaluate_regression(results, target)

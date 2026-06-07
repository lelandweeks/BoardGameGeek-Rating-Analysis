"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data import load_data, merge_data
from features import get_linear_features, get_genre_features
from features import get_class_features, create_rating_label
from features import CONFIG, CAT_GENRES
from models import run_logistic_reg
from models import run_linear_reg, run_lasso, run_ridge
from models import run_dt, run_rf
from evaluate import print_class_metrics, eval_reg, get_reg_metrics
from logger import log_run

from sklearn.linear_model import LogisticRegression
from models import run_grid_search

OUTPUT_DIR = 'output/'
MAX_ITER = 10000 # 1000=>5000 were both too low for convergence
LASSO_ALPHA = 0.001
RIDGE_ALPHA = 1.0
DT_MAX_DEPTH = None
RF_N_ESTIMATORS = 100

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

if "q1" in models:
    question = "Q1: What games are overrated and underrated?"
    model_name = "Logistic Regression"
    target = "AvgRating"
    print(question)
    print()

    df = create_rating_label(df)
    X, y = get_class_features(df)

    # run logistic regression
    print("Running Logistic Regression...")
    y_test, y_pred = run_logistic_reg(X, y, max_iter=MAX_ITER)
    metrics = print_class_metrics(y_test, y_pred)
    log_run(
        question = question,
        model_name = model_name,
        features = CONFIG[0],
        preprocessing = CONFIG[1] + ["StandardScaler"],
        feature_engineering= CONFIG[2],
        hyperparameters = {"max_iter": MAX_ITER},
        results = metrics
    )
    print()

    # following the grid search example from class
    y_test, y_pred, best_params = run_grid_search(
        X, y, LogisticRegression(
            max_iter=MAX_ITER,
            random_state=42
        ),
        {"C": [0.01, 0.1, 1.0, 10.0]},
        scoring='accuracy'
    )
    metrics = print_class_metrics(y_test, y_pred)
    log_run(
        question=question, 
        model_name="LogisticRegression GridSearch", 
        features=CONFIG[0], 
        preprocessing=CONFIG[1] + ["StandardScaler"], 
        feature_engineering=CONFIG[2], 
        hyperparameters=best_params, 
        results=metrics
    )
    print()


if "q2" in models:
    question = "Q2: Can we predict a game's average rating based on its features?"
    model_name = "Linear Regression"
    target = "AvgRating"
    print(question)
    print()

    X, y = get_linear_features(df, target)

    # run linear regression
    print("Running Linear Regression...")
    y_test, y_pred = run_linear_reg(X, y)
    metrics = get_reg_metrics(y_test, y_pred)
    eval_reg(metrics)
    log_run(
        question = question,
        model_name = model_name,
        features = CONFIG[0],
        preprocessing = CONFIG[1] + ["StandardScaler"],
        feature_engineering= CONFIG[2],
        hyperparameters = None,
        results = metrics
    )
    print()

    # run lasso regression
    print("Running Lasso Regression...")
    y_test, y_pred = run_lasso(X, y, LASSO_ALPHA)
    metrics = get_reg_metrics(y_test, y_pred)
    eval_reg(metrics)
    log_run(
        question = question,
        model_name = "Lasso",
        features = CONFIG[0],
        preprocessing = CONFIG[1] + ["StandardScaler"],
        feature_engineering = CONFIG[2],
        hyperparameters = {"alpha": LASSO_ALPHA},
        results = metrics
    )
    print()

    # run ridge regression
    print("Running Ridge Regression...")
    y_test, y_pred = run_ridge(X, y, alpha=RIDGE_ALPHA)
    metrics = get_reg_metrics(y_test, y_pred)
    eval_reg(metrics)
    log_run(
        question = question,
        model_name = "Ridge",
        features = CONFIG[0],
        preprocessing = CONFIG[1] + ["StandardScaler"],
        feature_engineering = CONFIG[2],
        hyperparameters = {"alpha": RIDGE_ALPHA},
        results = metrics
    )
    print()

    # run decision tree
    print("Running Decision Tree Regression...")
    y_test, y_pred = run_dt(X, y, DT_MAX_DEPTH)
    metrics = get_reg_metrics(y_test, y_pred)
    eval_reg(metrics)
    log_run(
        question = question,
        model_name = "DecisionTree",
        features = CONFIG[0],
        preprocessing = CONFIG[1],
        feature_engineering = CONFIG[2],
        hyperparameters = {"max_depth": DT_MAX_DEPTH},
        results = metrics
    )
    print()

    # run random forest
    print("Running Random Forest Regression...")
    y_test, y_pred = run_rf(X, y, RF_N_ESTIMATORS)
    metrics = get_reg_metrics(y_test, y_pred)
    eval_reg(metrics)
    log_run(
        question = question,
        model_name = "RandomForest",
        features = CONFIG[0],
        preprocessing = CONFIG[1],
        feature_engineering = CONFIG[2],
        hyperparameters = {"n_estimators": RF_N_ESTIMATORS},
        results = metrics
    )
    print()


if "q3" in models:
    question = "Q3: Does complexity predict rating within a genre?"
    model_name = "Linear Regression"
    target = "AvgRating"
    print(question) 
    print()
    
    for genre in CAT_GENRES:
        X, y = get_genre_features(df, genre)

        y_test, y_pred = run_linear_reg(X, y)
        metrics = get_reg_metrics(y_test, y_pred)
        print(f"{genre}")
        eval_reg(metrics)
        log_run(
            question = question,
            model_name = model_name,
            features = ["GameWeight"],
            preprocessing = ["dropna"],
            feature_engineering = CONFIG[2],
            hyperparameters = {"genre": genre},
            results = metrics
        )
        print()
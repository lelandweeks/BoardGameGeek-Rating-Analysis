"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_regression_metrics(y_test, y_pred):

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    return {
        "rmse": round(float(rmse), 4),
        "mae":  round(float(mae),  4),
        "r2":   round(float(r2),   4),
    }


def evaluate_regression(metrics):

    print(f"Results:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  R2  : {metrics['r2']:.4f}")

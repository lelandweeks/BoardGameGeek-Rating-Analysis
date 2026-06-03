"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

def run_regression(X, y):

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42
    )

    # train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # generate predictions on the test set
    y_pred = model.predict(X_test)

    return y_test, y_pred

def run_lasso(X, y, alpha):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42
    )

    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


def run_ridge(X, y, alpha=1.0):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42
    )

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred
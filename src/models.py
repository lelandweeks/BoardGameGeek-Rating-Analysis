"""
DSCI 631 Applied ML for Data Science
BoardGameGeek.com Rating Analysis
Author: Leland Weeks
Date: June 2026
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 42

def run_classifier(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred



def run_logistic_reg(X, y, max_iter=1000):

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # train a logistic regression model
    model = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # generate predictions on the test set
    y_pred = model.predict(X_test)

    return y_test, y_pred   



def run_linear_reg(X, y):

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
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
        random_state=RANDOM_STATE
    )

    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


def run_ridge(X, y, alpha=1.0):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


# decision tree
def run_dt(X, y, max_depth=None):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred

# random forest
# 100 default
def run_rf(X, y, n_estimators=100):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred
# linear_models.py
import numpy as np
import evalu as ev
import statsmodels.api as sm

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def clustered_linear_regression_train(X, y):
    ev.print_line("LinearRegression for Cluster")


def sm_linear_regression_train(X, y):
    ev.print_line("StatsModels LinearRegression")
    model = sm.OLS(y, X)
    results = model.fit()
    return results


def linear_regression_train(X=None, y=None, df=None, **kwargs):
    """
    Train a Linear Regression model.

    Parameters:
    - X (array-like or DataFrame): Features.
    - y (array-like, optional): Target variable.
    - df (DataFrame, optional): If provided, X and y will
        be extracted from the DataFrame.
    - **kwargs: Additional parameters to pass to LinearRegression.

    Returns:
    - LinearRegression: Trained Linear Regression model.
    """
    ev.print_line("LinearRegression")
    if df is not None:
        # If a DataFrame is provided, extract X and y
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    lregr = linear_model.LinearRegression(**kwargs)
    lregr.fit(X, y)
    return lregr


def linear_ridge_regr_train(X, y, **kwargs):
    # if kwargs:
    #     print("setting kwargs ", kwargs)
    if "alphas" in kwargs:
        ev.print_line("RidgeCrossValidation")
        lrregr = linear_model.RidgeCV(**kwargs)
        lrregr.fit(X, y)
        print("lrregr.alpha_:", lrregr.alpha_)
    else:
        ev.print_line("Ridge")
        lrregr = linear_model.Ridge(**kwargs)
        lrregr.fit(X, y)
    return lrregr


def linear_lasso_regr_train(X, y, **kwargs):
    if "alphas" in kwargs:
        ev.print_line("LassoCrossValidation")
        llregr = linear_model.LassoCV(**kwargs)
        # print("llregr.alpha_:", llregr.alpha_)
        llregr.fit(X, y.ravel())
        print("llregr.alpha_:", llregr.alpha_)
    else:
        ev.print_line("Lasso")
        llregr = linear_model.Lasso(alpha=kwargs.get("alpha", 0.5))
        llregr.fit(X, y.ravel())
    return llregr


def bayesian_ridge_regr_train(X, y, **kwargs):
    ev.print_line("BayesianRidge")
    brregr = linear_model.BayesianRidge()
    brregr.set_params(**kwargs)
    brregr.fit(X, np.ravel(y))
    return brregr


def ard_regr_train(X, y, **kwargs):
    ev.print_line("ARD")
    ardregr = linear_model.ARDRegression()
    ardregr.fit(X, y)
    return ardregr


def elastic_net_regr_train(X, y, **kwargs):
    ev.print_line("ElasticNet")
    enregr = linear_model.BayesianRidge()
    enregr.fit(X, y)
    return enregr


def sgd_regr_train(X, y, **kwargs):
    ev.print_line("SGD")
    sgdregr = linear_model.SGDRegressor(**kwargs)
    sgdregr.fit(X, y.ravel())
    return sgdregr


def huber_regr_train(X, y, **kwargs):
    ev.print_line("Huber")
    sgdregr = linear_model.HuberRegressor(**kwargs)
    sgdregr.fit(X, y.ravel())
    return sgdregr


def linear_predict(model, X_test):
    return model.predict(X_test)


def extend_to_polynomial(X, **kwargs):
    ev.print_line("PolynomialExtension")
    if kwargs:
        print("degree n of X**n: ", kwargs.get("degree"))
    poly = PolynomialFeatures(**kwargs)
    return poly.fit_transform(X)


# ===========================================================
if __name__ == "__main__":
    print("This is the 'linear_model.py' file")

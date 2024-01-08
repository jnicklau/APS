# linear_models.py
import numpy as np
import evalu as ev

from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def split_train_val_test(X, y, r1=0.5, r2=0.5, ps=False):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=r1, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=r2, shuffle=False
    )
    if ps:
        print("\nshape of X and y: ", np.shape(X), np.shape(y))
        print("shape of X_train and y_train: ", np.shape(X_train), np.shape(y_train))
        print("shape of X_val and y_val: ", np.shape(X_val), np.shape(y_val))
        print("shape of X_test and y_test: ", np.shape(X_test), np.shape(y_test), "\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


def linear_regression_train(X, y, **kwargs):
    ev.print_line("LinearRegression")
    lregr = linear_model.LinearRegression(positive=kwargs.get("positive", False))
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

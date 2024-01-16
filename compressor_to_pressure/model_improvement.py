# model_improvement.py
import evalu as ev
import numpy as np
import linear_models as lm


def train_and_evaluate_different_functions(X, y, foo, **kwargs):
    model = foo(
        X,
        y,
        params=kwargs.get("params", [1e-6] * 2),
    )
    y_pred, y_std = model.predict(X, return_std=True)
    return *ev.metrics(y, y_pred), y_std[-1]


def manual_hyp_param_search(
    foo, X_train, y_train, X_val, y_val, pname1=None, pname2=None, **kwargs
):
    """
    Perform manual hyperparameter search for a regression model.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target values.
    - X_val (numpy.ndarray): Validation feature matrix.
    - y_val (numpy.ndarray): Validation target values.
    - pname1 (str,optional): Name of the first hyperparameter.
    - pname2 (str,optional): Name of the second hyperparameter.
    - **kwargs: Optional hyperparameters for the regression model.

    Returns:
    - y_pred_array (numpy.ndarray): Array of predicted values.
    - resid_array (numpy.ndarray): Array of residuals.
    - models (list of lists):
    """
    # Default values
    n1 = kwargs.get("n1", 1)
    s1 = kwargs.get("s1", 1)
    e1 = kwargs.get("e1", 1)
    t1 = kwargs.get("t1", "linear")
    n2 = kwargs.get("n2", 1)
    s2 = kwargs.get("s2", 1)
    e2 = kwargs.get("e2", 1)
    t2 = kwargs.get("t2", "linear")

    if t1 == "linear":
        params1 = np.linspace(s1, e1, n1)
    else:
        params1 = np.logspace(s1, e1, n1)
    if t2 == "linear":
        params2 = np.linspace(s2, e2, n2)
    else:
        params2 = np.logspace(s2, e2, n2)

    if n1 * n2 > 1:
        ev.print_line("Manual HPS")
    y_pred_array = np.zeros((n1, n2, np.shape(y_val)[0]))
    resid_array = np.zeros((n1, n2, np.shape(y_train)[0]))
    models = [[0] * n2] * n1
    for i in range(n1):
        for j in range(n2):
            if pname1 is not None:
                if pname2 is not None:
                    lr_model = foo(
                        X_train, y_train, **{pname1: params1[i], pname2: params2[j]}
                    )
            else:
                lr_model = lm.sm_linear_regression_train(
                    X_train,
                    y_train,
                )
            models[i][j] = lr_model
            y_pred = lr_model.predict(X_val)
            residuals = -lr_model.predict(X_train) + y_train.ravel()
            y_pred_array[i, j, :] = y_pred.reshape(1, -1)
            resid_array[i, j, :] = residuals
    return y_pred_array, resid_array, models


if __name__ == "__main__":
    print("model_improvement.py")

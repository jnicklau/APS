# eval.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import (
    cross_val_score,
    LearningCurveDisplay,
)
import reading_data as rd


def plot_tree_from_model(model):
    # tree = model.tree_
    tree.plot_tree(model)
    plt.show()


def get_variable_name(variable, namespace):
    return [name for name, obj in namespace.items() if obj is variable]


def print_line(mstring=""):
    s_len = len(mstring)
    print(
        "#---------------------------",
        "%s" % mstring,
        "-" * (50 - s_len),
    )


def comp_and_eval_predictions(y_pred_array, y_true, **kwargs):
    """
    Evaluate predictions by computing R-squared and MSE for each.

    Parameters:
    - y_pred_array (numpy.ndarray): Array containing predictions.
    - y_true (numpy.ndarray): True values.

    Returns:
    numpy.ndarray: Array containing R-squared and MSE values for each prediction.
    """
    metrics_array = np.zeros((np.shape(y_pred_array)[0], np.shape(y_pred_array)[1], 2))
    for i in range(np.shape(y_pred_array)[0]):
        for j in range(np.shape(y_pred_array)[1]):
            y_pred = y_pred_array[i, j, :]
            metrics_array[i, j, :] = metrics(y_true, y_pred.reshape(-1, 1))
    return metrics_array


def cross_validation(X=None, y=None, df=None, model=None, **kwargs):
    """
    Perform cross-validation for a given model.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target values.
    - model: The machine learning model.
    - **kwargs: Additional keyword arguments for cross_val_score.

    Returns:
    numpy.ndarray: Array of cross-validation scores.
    """
    # If a DataFrame is provided, extract X and y
    print_line("CROSS VALIDATION")
    if df is not None:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    scores = cross_val_score(model, X, y.ravel(), **kwargs)
    print(
        "%0.2f mean with a standard deviation of %0.2f" % (scores.mean(), scores.std())
    )
    return scores


def qqplot(y, dist, distname):
    """
    Generate a Quantile-Quantile plot.

    Parameters:
    - y (numpy.ndarray): Data to be plotted.
    - dist: Theoretical distribution.
    - distname (str): Name of the distribution.

    Returns:
    None
    """
    sm.qqplot(y, dist, fit=True, line="45")
    plt.title("Quantile-Quantile Plot vs %s-Distribution" % distname)
    plt.show()


def metrics(y_test, y_pred):
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def print_metrics_array(m):
    """m: arraylike"""
    print(
        "MSE: --> first parameter downward, second parameter to the right  \n",
        m[:, :, 0],
    )
    print("R2: \n", m[:, :, 1])


def coeff_analysis(model, idlist, plotbool=False, reggoal="K2V0"):
    # The coefficients
    if reggoal == "K2V0":
        print("shape of model.coef_", np.shape(model.coef_))
        coefs = model.coef_.reshape(1, 26)
        print("shape of coefs", np.shape(coefs))

        print("Coefficients for motors: \n", coefs[:, 0:13])
        print("Coefficients for flow: \n", coefs[:, 13:26])
        if plotbool:
            plot_model_weights(
                [coefs[:, 0:13], coefs[:, 13:26]],
                c_nums=idlist["Airleaderkanal"],
            )
    if reggoal == "Vi2p":
        print("Coefficients: \n", model.coef_)
        if plotbool:
            plot_model_weights(model.coef_, c_nums=idlist.columns, reggoal=reggoal)


def plot_model_vs_real(y_true, y_pred, y_pred_baseline, **kwargs):
    seconds = np.arange(np.shape(y_true)[0])
    plt.plot(seconds, y_true, color="green", linewidth=2, label="y_true")
    mse_pred, r2_pred = metrics(y_true, y_pred)
    plt.plot(
        seconds,
        y_pred,
        color="blue",
        linewidth=3,
        label="y_pred \n MSE: %.3f R2: %.3f " % (mse_pred, r2_pred),
    )
    mse_base, r2_base = metrics(y_true, y_pred_baseline)
    plt.plot(
        seconds,
        y_pred_baseline,
        color="black",
        linewidth=1,
        label="y_pred_baseline\n MSE: %.3f R2: %.3f " % (mse_base, r2_base),
    )
    if kwargs.get("std", False):
        ystd = kwargs.get("ystd")
        plt.fill_between(
            seconds,
            y_pred - ystd,
            y_pred + ystd,
            color="pink",
            alpha=0.5,
            label="predict std",
        )
    plt.legend(loc=1)
    plt.title(kwargs.get("title", ""))
    xlab, ylab = get_labels_from_reggoal(kwargs.get("reggoal", None))
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.show()


def get_labels_from_reggoal(reggoal):
    if reggoal == "K2V0":
        return "data(/time) points", r"volume flow $\dot V_0$"
    elif reggoal == "Vi2p":
        return "data(/time) points", "pressure $p$"
    else:
        return "", ""


def full_residual_analysis(residuals, data):
    print_line("full residual analysis")
    plot_resids(residuals)
    plot_resids_dist(residuals)
    qqplot(residuals, norm, "Normal")
    plot_resids_vs_predictors(
        residuals,
        data,
        kind="hist",
        bins=100,
    )
    plot_resids_vs_pattern_predictors(
        residuals,
        data,
        "dwa",
        kind="hist",
        bins=100,
    )


def plot_resids_dist(residuals, **kwargs):
    sns.set_theme()
    sns.displot(
        residuals,
        bins=20,
        **kwargs,
    )
    plt.xlabel("Residuals")
    plt.ylabel("Counts")
    plt.title("Histogram of Residuals")
    plt.tight_layout()
    plt.show()


def plot_resids_vs_predictors(residuals, datat, pcolumns=None, **kwargs):
    """
    This function creates a series of joint plots, each displaying
    the relationship between the residuals and an individual predictor
    variable. The joint plots include scatter plots with marginal histograms.

    Parameters:
    - residuals (numpy.ndarray): Array of residuals obtained from a regression model.
    - X (numpy.ndarray): Feature matrix corresponding to the predictors.
    - datat (pandas.DataFrame): DataFrame containing the predictor variables.
    - pcolumns (list): List of columns for which the residuals are to be plotted agains
    - **kwargs: Additional keyword arguments to be passed to seaborn.jointplot.

    Returns:
    None
    """
    if pcolumns is None:
        pcolumns = list(np.arange(len(datat.columns)))
    for i, predictor in enumerate(datat.columns):
        if i in pcolumns:
            sns.jointplot(data=datat, x=predictor, y=residuals, **kwargs)
            plt.xlabel(predictor)
            plt.ylabel("Residuals")
            plt.title("Residuals vs. %s" % (predictor))
            plt.show()


def plot_single_patterns(
    residuals, datat, phase, xticks, xticklabels, predictor_name, **kwargs
):
    # Create joint plot
    g = sns.jointplot(data=datat, x=phase, y=residuals, **kwargs)
    g.ax_joint.set_xticks(xticks)
    g.ax_joint.set_xticklabels(xticklabels)
    plt.xlabel(predictor_name)
    plt.ylabel("Residuals")
    plt.title("Residuals vs. phase of %s" % (predictor_name))
    plt.show()


def plot_resids_vs_pattern_predictors(residuals, datat, duration, **kwargs):
    """
      Parameters:
      - residuals (numpy.ndarray): Array of residuals obtained from a regression model.
      - datat (pandas.DataFrame): DataFrame containing the predictor variables.
      - col1 (str): Name of the first column for creating the complex vector.
      - col2 (str): Name of the second column for creating the complex vector.
      - xlabel (str): Name of pattern
    - **kwargs: Additional keyword arguments to be passed to seaborn.jointplot.

      Returns:
      None
    """
    nplots = 0
    if "d" in duration:
        nplots += 1
        predictor_name = "daily_pattern"
        col1 = predictor_name + "_real"
        col2 = predictor_name + "_imag"
        phase = np.angle(datat[col1] + 1j * datat[col2])
        xticks = np.linspace(0, 24, 9)
        xticklabels = [str(int(element)) for element in list(xticks)]
        phase = 12 * ((phase / np.pi)) % 24
        plot_single_patterns(
            residuals, datat, phase, xticks, xticklabels, predictor_name, **kwargs
        )
    if "w" in duration:
        nplots += 1
        predictor_name = "weekly_pattern"
        col1 = predictor_name + "_real"
        col2 = predictor_name + "_imag"
        phase = np.angle(datat[col1] + 1j * datat[col2])
        xticks = np.linspace(0, 7, 8)
        xticklabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]
        phase = 3.5 * ((phase / np.pi)) % 7
        plot_single_patterns(
            residuals, datat, phase, xticks, xticklabels, predictor_name, **kwargs
        )
    if "a" in duration:
        nplots += 1
        predictor_name = "annual_pattern"
        col1 = predictor_name + "_real"
        col2 = predictor_name + "_imag"
        phase = np.angle(datat[col1] + 1j * datat[col2])
        xticks = np.linspace(0, 365, 5)
        xticklabels = ["Jan", "Apr", "Juli", "Sept", ""]
        phase = 365 / 2 * ((phase / np.pi)) % 365
        plot_single_patterns(
            residuals, datat, phase, xticks, xticklabels, predictor_name, **kwargs
        )
    if all(char not in duration for char in ["a", "w", "d"]):
        xticks = np.linspace(-np.pi, np.pi, 5)
        xticklabels = [
            r"$-\pi$",
            r"$-\frac{\pi}{2}$",
            "0",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
        ]
        col1 = predictor_name + "_real"
        col2 = predictor_name + "_imag"
        phase = np.angle(datat[col1] + 1j * datat[col2])
        plot_single_patterns(
            residuals, datat, phase, xticks, xticklabels, predictor_name, **kwargs
        )


def plot_resids_vs_target(residuals, datat, targetname, **kwargs):
    sns.jointplot(data=datat, x=datat[targetname], y=residuals, **kwargs)
    plt.xlabel(targetname)
    plt.ylabel("Residuals")
    plt.title("Residuals vs. %s" % (targetname))
    plt.show()


def plot_resids(residuals, **kwargs):
    plt.ylabel("Residuals")
    sns.lineplot(residuals)
    plt.show()


def plot_ar_model_vs_real(alltime, testtime, alldata, predictions, **kwargs):
    # Plot the results
    test_length = len(testtime)
    plt.plot(alltime[-test_length:], alldata[-test_length:], label="Original Data")
    plt.plot(
        testtime,
        predictions,
        "o",
        label="AR Model Predictions",
        linestyle="-",
        color="red",
    )
    if "conf_int" in kwargs:
        conf_int = kwargs.get("conf_int")
        plt.fill_between(
            testtime, conf_int[:, 0], conf_int[:, 1], color="red", alpha=0.2
        )
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.legend()
    plt.show()


def plot_learn_curve(model, X, y, **kwargs):
    common_params = {
        "train_sizes": np.linspace(0.05, 1.0, 5),
        "n_jobs": 3,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
    }
    LearningCurveDisplay.from_estimator(
        model,
        X,
        y,
        **common_params,
    )
    plt.show()


def plot_model_weights(coeffs, c_nums, reggoal="K2V0"):
    coeffs = [coeffs[0][0], coeffs[1][0]]
    fig, ax = plt.subplots()
    c_weight = {}
    if reggoal == "K2V0":
        c_labels = [".R2", ".AE1"]
        """ potentially turn list of coefficients around
        to see everything in the plot of coefficients"""
        if check_inequality(coeffs[0], coeffs[1], "<="):
            coeffs = coeffs[::-1]
            c_labels = c_labels[::-1]
    elif reggoal == "Vi2p":
        c_labels = ["Netzdruck"]

    for i in range(len(coeffs)):
        c_weight.update({c_labels[i]: coeffs[i]})
    for boolean, weight in c_weight.items():
        plt.bar(c_nums, weight, label=boolean)
    ax.legend()
    plt.xticks(c_nums)
    plt.xlabel("Airleaderkanal")
    plt.ylabel("weight of coefficient")
    plt.show()
    return fig


def check_inequality(list1, list2, inequality):
    """
    checks weather entirety of list follows a certain inequality
    # eg:   list1 = [1, 2, 3], list2 = [4, 5, 6], inequality = '<'
    """

    # Make sure the lists are of the same length
    if len(list1) != len(list2):
        return False
    # Check the inequality for each pair of elements
    for elem1, elem2 in zip(list1, list2):
        if not eval(f"{elem1} {inequality} {elem2}"):
            return False
    # If all comparisons passed, return True
    return True


def single_compressor_contributions(model, compressors):
    single_compr_cons = np.zeros(13)
    for i in range(13):
        a = np.zeros((1, 13 * 2))
        a[0, i] = 1
        a[0, i + 13] = rd.get_max_volume_flow(compressors, i)
        # print(a)
        single_compr_cons[i] = model.predict(a)[0]
    return single_compr_cons


def print_model_metrics(y_true, y_pred, y_pred_baseline):
    # The mean squared error
    print_line("Model Metrics")
    print("Mean squared error of model: %.4f" % mean_squared_error(y_true, y_pred))
    print(
        "Mean squared error of baseline: %.4f"
        % mean_squared_error(y_true, y_pred_baseline)
    )
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination R2 of model: %.3f" % r2_score(y_true, y_pred))
    print(
        "Coefficient of determination R2 of baseline: %.3f"
        % r2_score(y_true, y_pred_baseline)
    )
    return


# ===========================================================
if __name__ == "__main__":
    print("This is the 'eval.py' file")

# eval.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    cross_val_score,
    LearningCurveDisplay,
    learning_curve,
)


def print_line(mstring=""):
    s_len = len(mstring)
    print(
        "#---------------------------",
        "%s" % mstring,
        "-" * (50 - s_len),
    )


def comp_and_eval_predictions(y_pred_array, y_true, **kwargs):
    """for each predictions make a space for r2 value and mse value"""
    metrics_array = np.zeros((np.shape(y_pred_array)[0], np.shape(y_pred_array)[1], 2))
    for i in range(np.shape(y_pred_array)[0]):
        for j in range(np.shape(y_pred_array)[1]):
            y_pred = y_pred_array[i, j, :]
            metrics_array[i, j, :] = metrics(y_true, y_pred.reshape(-1, 1))
    return metrics_array


def qqplot(y):
    sm.qqplot(y, stats.t, fit=True, line="45")
    plt.show()


def metrics(y_test, y_pred):
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def print_metrics_array(m):
    """m: arraylike"""
    print("MSE: \n", m[:, :, 0])
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


def plot_model_vs_real(seconds, y_test, y_pred, y_pred_baseline, **kwargs):
    plt.plot(seconds, y_test, color="green", linewidth=2, label="y_true")
    mse_pred, r2_pred = metrics(y_test, y_pred)
    plt.plot(
        seconds,
        y_pred,
        color="blue",
        linewidth=3,
        label="y_pred \n MSE: %.3f R2: %.3f " % (mse_pred, r2_pred),
    )
    mse_base, r2_base = metrics(y_test, y_pred_baseline)
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
    # train_sizes, _, test_scores_nb, fit_times_nb, score_times_nblearning_curve(model,X,y,return_times = True)
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
        a[0, i + 13] = get_max_volume_flow(compressors, i)
        # print(a)
        single_compr_cons[i] = model.predict(a)[0]
    return single_compr_cons


def print_model_metrics(y_test, y_pred, y_pred_baseline):
    # The mean squared error
    print("Mean squared error of model: %.4f" % mean_squared_error(y_test, y_pred))
    print(
        "Mean squared error of baseline: %.4f"
        % mean_squared_error(y_test, y_pred_baseline)
    )
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination R2 of model: %.3f" % r2_score(y_test, y_pred))
    print(
        "Coefficient of determination R2 of baseline: %.3f"
        % r2_score(y_test, y_pred_baseline)
    )


# ===========================================================
if __name__ == "__main__":
    print("This is the 'eval.py' file")

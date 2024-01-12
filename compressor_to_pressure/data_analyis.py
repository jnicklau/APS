# data_analyis.py
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import evalu as ev

# Apply the default theme
sns.set_theme()


def heat_corr(data, method, **kwargs):
    """Create a heatmap of a
    correlation_matrix from data"""
    correlation_matrix = data.corr(method=method)
    sns.heatmap(correlation_matrix, **kwargs)
    plt.title("%s Correlation Matrix Heatmap" % method)
    plt.tight_layout()
    plt.show()
    return correlation_matrix


def make_hists(data, **kwargs):
    for i, col in enumerate(data.columns):
        sns.displot(data, x=col, kde=True, **kwargs)
        plt.show()


def describe_df(data):
    for col in data.columns:
        data[col].describe()


def get_distr_metrics(a):
    ev.print_line("Describing Data Moments")
    a = a.ravel()
    mean = np.mean(a)
    variance = np.var(a)
    skew = stats.skew(a)
    kurtosis = stats.kurtosis(a)
    return {"mean": mean, "var": variance, "skew": skew[0], "kurtosis": kurtosis[0]}


if __name__ == "__main__":
    print("data_analyis.py")

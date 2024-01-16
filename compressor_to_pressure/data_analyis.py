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


def measure_cluster_goodness(df):
    print("estimating clustering goodness")
    sscore = silhouette_score(dataframe, dataframe["cluster"])
    print("silhouette_score(data, labels) high is good: ", sscore)
    # Calculate centroids, Euclidean distances between data points and centroids and sum up
    centroids = dataframe.groupby(dataframe["cluster"]).mean()
    distances = pairwise_distances(
        dataframe, centroids, metric="euclidean", squared=True
    )
    inertia = np.sum(distances)
    print("inertia(data, labels) low is good: ", inertia)
    dbsocre = davies_bouldin_score(data, dataframe["cluster"])
    print("davies_bouldin_score(data, labels) low is good: ", dbscore)
    chscore = calinski_harabasz_score(dataframe, dataframe["cluster"])
    print("calinski_harabasz_score(data, labels) high is good: ", chscore)
    return sscore, inertia, dbscore, chscore


def make_hists(data, **kwargs):
    for i, col in enumerate(data.columns):
        sns.displot(data, x=col, **kwargs)
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

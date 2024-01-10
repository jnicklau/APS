# data_analyis.py
import seaborn as sns
import matplotlib.pyplot as plt


def heat_corr(data, method, **kwargs):
    """Create a heatmap of a
    correlation_matrix from data"""
    correlation_matrix = data.corr(method=method)
    sns.heatmap(correlation_matrix, **kwargs)
    plt.title("%s Correlation Matrix Heatmap" % method)
    plt.tight_layout()
    plt.show()
    return correlation_matrix


if __name__ == "__main__":
    print("data_analyis.py")

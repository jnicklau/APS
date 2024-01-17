# combining_models.py
import pandas as pd
import joblib
from sklearn import linear_model
import evalu as ev
from sklearn.ensemble import AdaBoostRegressor


def adaboost(base_regressor, X=None, y=None, df=None, **kwargs):
    ev.print_line("AdaBoosting")
    # If a DataFrame is provided, extract X and y
    if df is not None:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    # Create an AdaBoost regressor with the decision tree as the base learner
    adaboost_regressor = AdaBoostRegressor(base_regressor, **kwargs)
    # Train the AdaBoost regressor
    adaboost_regressor.fit(X, y)
    return adaboost_regressor


def apply_lreg_to_clusters(dataframe, cluster_labels_column="cluster"):
    """
    Apply linear regression independently to each cluster of a pre-clustered DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The pre-clustered DataFrame with the first column as the
        target variable and the rest as feature variables.
    - cluster_labels_column (str, optional): Name of the column containing cluster labels
        in the DataFrame. Default is 'Cluster'.

    Returns:
    - dict: A dictionary containing linear regression models for each unique cluster label.
    """

    # Extract features and target variable
    features = dataframe.columns[1:]
    target = dataframe.columns[0]

    # Initialize a dictionary to store linear regression models for each cluster
    regression_models = {}

    # Apply linear regression independently to each cluster
    unique_clusters = dataframe[cluster_labels_column].unique()
    for cluster_label in unique_clusters:
        cluster_data = dataframe[dataframe[cluster_labels_column] == cluster_label]
        cluster_X = cluster_data[features]
        cluster_y = cluster_data[target]

        # Fit linear regression model
        regression_model = linear_model.LinearRegression()
        regression_model.fit(cluster_X, cluster_y)
        # Save the model to a file
        joblib.dump(
            regression_model, f"linear_regression_model_cluster_{cluster_label}.joblib"
        )
        # Store the model in the dictionary
        regression_models[cluster_label] = regression_model

    return regression_models


def predict_on_new_clustered_data(
    new_data, regression_models, cluster_labels_column="cluster"
):
    """
    Predict target variable values for new labeled data using pre-trained regression models.

    Parameters:
    - new_data (pd.DataFrame): New labeled data with cluster labels.
    - regression_models (dict): Dictionary containing linear regression
        models for each unique cluster label.
    - cluster_labels_column (str, optional): Name of the column containing
         cluster labels in the new data. Default is 'Cluster'.

    Returns:
    - predictions (pd.Series): Predicted target variable values corresponding
        to the original indices in the new data.

    Example:
    regr_models = cm.apply_lreg_to_clusters(train_data, "cluster")
    y_pred_train = cm.predict_on_new_clustered_data(train_data, regr_models, "cluster")
    y_pred_val = cm.predict_on_new_clustered_data(val_data, regr_models, "cluster")

    """

    # Check if the 'Cluster' column exists in the new data
    if cluster_labels_column not in new_data.columns:
        raise ValueError(
            f"The '{cluster_labels_column}' column is not present in the new data."
        )
    # Extract features and target variable
    features = new_data.columns[1:]
    target = new_data.columns[0]
    # Initialize an empty Series to store predictions
    predictions = pd.Series(index=new_data.index)

    # Iterate over unique clusters
    for cluster_label, regression_model in regression_models.items():
        # Filter data for the current cluster
        cluster_data = new_data[new_data[cluster_labels_column] == cluster_label]
        # If there is data for the current cluster, make predictions
        if not cluster_data.empty:
            cluster_X = cluster_data[features]
            cluster_predictions = regression_model.predict(cluster_X)
            # Assign predictions to the corresponding rows in the original index
            predictions.loc[cluster_data.index] = cluster_predictions

    return predictions


# ===========================================================
if __name__ == "__main__":
    print("This is the 'combining_models.py' file")

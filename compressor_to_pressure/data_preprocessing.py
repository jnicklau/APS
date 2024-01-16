# data_preprocessing.py
import evalu as ev
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import linear_models as lm


def split_train_val_test_xy(X, y, r1=0.5, r2=0.5, ps=False, **kwargs):
    """
    Split the dataset

    This function partitions the input features (X) and
    target values (y) into training, validation,
    and test sets using the provided ratios.
    It uses the `train_test_split` function from scikit-learn.

    Parameters:
    - X (numpy.ndarray): Feature matrix.
    - y (numpy.ndarray): Target values.
    - r1 (float): Ratio of the dataset to be allocated
      for the training set.
    - r2 (float): Ratio of the dataset to be allocated
      for the validation set (remaining for the test set).
    - ps (bool): If True, print the shapes of
      X, y, X_train, y_train, X_val, y_val, X_test, and y_test.

    Returns:
    tuple: A tuple containing X_train, X_val, X_test, y_train, y_val, y_test.
    """
    ev.print_line("Splitting into Training, Validation and Testing")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=r1, **kwargs)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=r2, **kwargs
    )
    if ps:
        print("shape of X and y: ", np.shape(X), np.shape(y))
        print("shape of X_train and y_train: ", np.shape(X_train), np.shape(y_train))
        print("shape of X_val and y_val: ", np.shape(X_val), np.shape(y_val))
        print("shape of X_test and y_test: ", np.shape(X_test), np.shape(y_test), "\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_val_test_df(df, r1=0.5, r2=0.5, ps=True, **kwargs):
    """
    Split the DataFrame into training, validation, and test sets.

    This function assumes that the first column of the input DataFrame
    is the target variable (y). It partitions the DataFrame into
    training, validation, and test sets using the provided ratios.
    It uses the `train_test_split` function from scikit-learn.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing features and
      the target variable as the first column.
    - r1 (float): Ratio of the dataset for the training set.
    - r2 (float): Ratio of the dataset for the validation set.
    - ps (bool): If True, print the shapes of the resulting DataFrames.

    Returns:
    tuple: A tuple containing DataFrames for training, validation, and test sets.
    """
    ev.print_line("Splitting into Training, Validation and Testing")
    y_col = df.columns[0]
    X = df.drop(y_col, axis=1)
    y = df[y_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=r1, **kwargs)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=r2, **kwargs
    )

    if ps:
        print("\nshape of df: ", df.shape)
        print("shape of X_train and y_train: ", X_train.shape, y_train.shape)
        print("shape of X_val and y_val: ", X_val.shape, y_val.shape)
        print("shape of X_test and y_test: ", X_test.shape, y_test.shape, "\n")

    train_df = pd.concat([y_train, X_train], axis=1)
    val_df = pd.concat([y_val, X_val], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)

    return (train_df, val_df, test_df)


def get_scaled_xy_from_splitted_df(splitted_df, scaler=None, dg=1):
    """
    Preprocess and scale data for training, validation, and testing.

    Parameters:
    - splitted_df (pd.DataFrame): Tuple of DataFrames
    - scaler: Scaler object for feature scaling (if None, no scaling is applied).
    - dg (int,optional): Degree of Polynomial Extension
    Returns:
    tuple: Tuples containing X_train, X_val, X_test, y_train, y_val, y_test, and scalers.
           If scaler is None, the last element is None.
    """
    train_df, val_df, test_df = splitted_df
    X_train = train_df.iloc[:, 1:].to_numpy()
    X_val = val_df.iloc[:, 1:].to_numpy()
    X_test = test_df.iloc[:, 1:].to_numpy()
    y_train = train_df.iloc[:, 0].to_numpy().reshape(-1, 1)
    y_val = val_df.iloc[:, 0].to_numpy().reshape(-1, 1)
    y_test = test_df.iloc[:, 0].to_numpy().reshape(-1, 1)
    if scaler is not None:
        ev.print_line("Scaling Data")
        scaler_X = scaler
        scaler_y = scaler
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        y_test = scaler_y.transform(y_test)
    else:
        scaler_X = None
        scaler_y = None
    if dg > 1:
        X_train = lm.extend_to_polynomial(X_train)
        X_val = lm.extend_to_polynomial(X_val)
        X_test = lm.extend_to_polynomial(X_test)
    return (X_train, X_val, X_test), (y_train, y_val, y_test), (scaler_X, scaler_y)


def add_difference_column(df1, column_name, df2, new_column_name):
    ev.print_line("Adding Difference Column")
    """
    Adds a new column with 'new_column_name' to df2
    that contains the differences between
    subsequent values in the 'column_name' column of df1.
    """
    # Check if the input is a DataFrame
    if not isinstance(df1, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    if not isinstance(df2, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    # Check if the specified column exists in the DataFrame
    if column_name not in df1.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")

    # Calculate the differences between subsequent values in the specified column
    differences = df1[column_name].diff()
    differences.iloc[int(df1.index[0])] = 0
    # Add the differences as a new column to df
    df2[new_column_name] = differences
    return df2


def sum_and_remove_columns(input_df, columns_to_sum, new_column_name):
    ev.print_line("Simplifying Network Data")
    """
    Takes a DataFrame as input, calculates the sum of selected columns,
    writes the sum into a new column, and removes the selected columns.

    Parameters:
    - input_df: DataFrame
        The input DataFrame.
    - columns_to_sum: list
        List of column names to be summed.
    - new_column_name: str
        Name of the new column where the sum will be stored.
    Returns:
    - output_df = DataFrame
        Modified DataFrame with the sum and selected columns removed.
    """
    output_df = input_df.copy()
    # Calculate the sum of selected columns
    output_df[new_column_name] = output_df[columns_to_sum].sum(axis=1)
    # Drop the selected columns
    output_df.drop(columns=columns_to_sum, inplace=True)
    return output_df


def add_time_patterns(df, start_date="2023-11-01"):
    """
    Add columns for daily, weekly, and annual patterns to a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input df with a numerical index representing seconds.
    - start_date (str): The desired starting date in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: The DataFrame with additional columns for patterns.
    """
    # Convert the index to datetime with a custom start date
    ev.print_line("Add Time Patterns")
    start_datetime = pd.to_datetime(start_date)
    df.index = start_datetime + pd.to_timedelta(df.index, unit="s")

    minutes_in_day = 24 * 60
    hours_in_week = 7 * 24
    days_in_year = 365.25  # Consider leap years

    df["daily_pattern_real"] = np.real(
        np.exp(2j * np.pi * (df.index.hour * 60 + df.index.minute) / minutes_in_day)
    )
    df["daily_pattern_imag"] = np.imag(
        np.exp(2j * np.pi * (df.index.hour * 60 + df.index.minute) / minutes_in_day)
    )
    df["weekly_pattern_real"] = np.real(
        np.exp(2j * np.pi * (df.index.dayofweek * 24 + df.index.hour) / hours_in_week)
    )
    df["weekly_pattern_imag"] = np.imag(
        np.exp(2j * np.pi * (df.index.dayofweek * 24 + df.index.hour) / hours_in_week)
    )
    df["annual_pattern_real"] = np.real(
        np.exp(2j * np.pi * df.index.dayofyear / days_in_year)
    )
    df["annual_pattern_imag"] = np.imag(
        np.exp(2j * np.pi * df.index.dayofyear / days_in_year)
    )
    return df


def scale_Xy(X, y, scaler):
    """
    rescaling numpy arrays to a given scaler
    """
    ev.print_line("Scaling Data")
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y


def kmeans_cluster_df(dataframe, num_clusters, **kwargs):
    ev.print_line("KMeans Clustering into %s" % num_clusters)
    # Extract numerical columns from the DataFrame
    numerical_columns = dataframe.select_dtypes(include=["float64", "int64"]).columns
    numerical_data = dataframe[numerical_columns]
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, **kwargs)
    dataframe["cluster"] = kmeans.fit_predict(numerical_data)

    return dataframe


if __name__ == "__main__":
    print("data_preprocessing.py")

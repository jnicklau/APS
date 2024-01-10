# data_preprocessing.py
import evalu as ev
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
    - DataFrame
        Modified DataFrame with the sum and selected columns removed.
    """
    # Calculate the sum of selected columns
    input_df[new_column_name] = input_df[columns_to_sum].sum(axis=1)
    # Drop the selected columns
    input_df.drop(columns=columns_to_sum, inplace=True)
    return input_df


def scale_Xy(X, y, scaler):
    """
    rescaling numpy arrays to a given scaler
    """
    ev.print_line("Scaling Data")
    print("")
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y


if __name__ == "__main__":
    print("data_preprocessing.py")

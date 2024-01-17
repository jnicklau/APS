# decision_tree_models.py
import evalu as ev

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def random_forest_train(X=None, y=None, df=None, **kwargs):
    ev.print_line("RandomForest")
    # If a DataFrame is provided, extract X and y
    if df is not None:
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    if kwargs:
        print("setting kwargs ", kwargs)
    rforest = RandomForestRegressor(**kwargs)
    rforest.fit(X, y.ravel())
    return rforest


def decision_tree_train(X=None, y=None, df=None, **kwargs):
    """
    Train a DecisionTreeRegressor model.

    Parameters:
    - X (array-like or DataFrame): Features.
    - y (array-like, optional): Target variable.
    - df (DataFrame, optional): If provided, X and y will
        be extracted from the DataFrame.
    - **kwargs: Additional parameters to pass to DecisionTreeRegressor.

    Returns:
    - DecisionTreeRegressor: Trained DecisionTreeRegressor model.
    """
    ev.print_line("DecisionTree")
    if df is not None:
        # If a DataFrame is provided, extract X and y
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
    if kwargs:
        print("setting kwargs ", kwargs)
    dtregr = DecisionTreeRegressor(**kwargs)
    dtregr.fit(X, y)
    return dtregr


# ===========================================================
if __name__ == "__main__":
    print("This is the 'linear_model.py' file")

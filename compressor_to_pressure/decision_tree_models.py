# decision_tree_models.py
import evalu as ev

from sklearn.tree import DecisionTreeRegressor


def decision_tree_train(X, y, **kwargs):
    ev.print_line("DecisionTree")
    if kwargs:
        print("setting kwargs ", kwargs)
    dtregr = DecisionTreeRegressor(**kwargs)
    dtregr.fit(X, y)
    return dtregr


# ===========================================================
if __name__ == "__main__":
    print("This is the 'linear_model.py' file")

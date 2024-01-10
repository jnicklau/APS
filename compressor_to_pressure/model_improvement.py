# model_improvement.py


def train_and_evaluate_different_functions(X, y, foo, **kwargs):
    model = foo(
        X_train,
        y_train,
        params=kwargs.get("params", [1e-6] * 2),
    )
    y_pred, y_std = model.predict(X_test, return_std=True)
    return *ev.metrics(y_test, y_pred), y_std[-1]


if __name__ == "__main__":
    print("model_improvement.py")

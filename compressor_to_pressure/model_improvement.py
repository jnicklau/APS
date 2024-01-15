# model_improvement.py
import evalu as ev


def train_and_evaluate_different_functions(X, y, foo, **kwargs):
    model = foo(
        X,
        y,
        params=kwargs.get("params", [1e-6] * 2),
    )
    y_pred, y_std = model.predict(X, return_std=True)
    return *ev.metrics(y, y_pred), y_std[-1]


if __name__ == "__main__":
    print("model_improvement.py")

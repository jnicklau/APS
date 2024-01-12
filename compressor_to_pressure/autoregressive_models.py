# autoregressive_models.py

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
import evalu as ev


def ar_train(y, **kwargs):
    if "exog" in kwargs:
        print("AR-X")
    else:
        print("AR")
    order_ar_model = kwargs.get("order", 3)
    model = AutoReg(y, lags=order_ar_model)
    results = model.fit()
    return results


def ar_sel_train(y, **kwargs):
    print("AutoReg with Order Selection")
    sel = ar_select_order(y, glob=True, **kwargs)
    lags = sel.ar_lags
    results = sel.model.fit()
    return results, lags


def arimax_train(y, **kwargs):
    if "exog" in kwargs:
        ev.print_line("ARIMAX")
    else:
        ev.print_line("ARIMA")
    model = ARIMA(y, **kwargs)
    results = model.fit()
    return results, model.model_orders

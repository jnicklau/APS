# train.py

import pandas as pd
import numpy as np

import linear_models as lm
import autoregressive_models as am
import filenames as fn
import reading_data as rd
import evalu as ev
import data_preprocessing as dpp
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)

# airleader_files = fn.all_air_leader_files
airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
# airleader_files = [fn.short_air_leader_file]

# airflow_files = fn.flow_file
airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
# airflow_files = fn.short_flow_file
scaler = StandardScaler()

# ------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    ev.print_line("READING DATA")
    air_leader = rd.fetch_airleader(airleader_files)
    air_flow = rd.fetch_airflow(airflow_files)
    # ------------------------------------------------------------
    common_times = rd.get_common_times(air_leader, air_flow)
    air_leader, air_flow = rd.put_on_same_time_interval(
        air_leader,
        air_flow,
        common_times,
        method="linear",
    )
    # rd.print_df_information(air_leader, name="air_leader", nhead=30)
    # rd.print_df_information(air_flow, name="air_flow", nhead=30)
    # ------------------------------------------------------------
    V_out, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    X, y = V_out.to_numpy(), p.to_numpy()
    X, y = dpp.scale_Xy(X, y, scaler)
    # X = lm.extend_to_polynomial(X,degree = 2)
    myseed = 42
    X_train, X_val, X_test, y_train, y_val, y_test = dpp.split_train_val_test(
        X, y, 0.1, ps=True
    )

    # ------------------------------------------------------------
    lr_model = lm.linear_ridge_regr_train(
        X_train,
        y_train,
        # positive=True,
        alphas=np.logspace(-9, 6, 31),
        # alpha=3.16,
        # params = [1e-3,1e-3]
    )
    # ------------------------------------------------------------
    lr_train = lr_model.predict(X_train)
    lr_val = lr_model.predict(X_val)
    lr_test = lr_model.predict(X_test)
    # ------------------------------------------------------------
    ar_model, ar_lags = am.arimax_train(
        y_train,
        order=([1, 2, 3, 9, 10], 0, 1),
        exog=lr_train,
        trend="ct",
    )
    # print(ar_model.summary())
    # ------------------------------------------------------------
    ar_pred = ar_model.get_prediction(
        start=len(y_train),
        end=len(y_train) + len(np.concatenate((y_val, y_test), axis=0)) - 1,
        # exog = lr_train,
        exog=np.concatenate((lr_val, lr_test), axis=0),
        dynamic=True,
        trend="ct",
    )
    ar_conf_int = ar_pred.conf_int()
    ar_y_pred = ar_pred.predicted_mean
    # y_pred = model.predict(X_test)
    # y_pred, y_std = model.predict(X_test, return_std=True)
    y_pred_baseline = np.full(np.shape(y_test), np.mean(y_train))

    # ------------------------------------------------------------
    # ev.coeff_analysis(
    #     model,
    #     air_flow,
    #     plotbool=False,
    #     reggoal="Vi2p",
    # )
    ev.print_model_metrics(
        y_test,
        ar_y_pred[np.shape(y_val)[0] :],
        y_pred_baseline,
    )

    # # plot how well we perform on training data
    # ev.plot_model_vs_real(
    #     np.arange(np.shape(X_train)[0]),
    #     y_train,
    #     model.predict(X_train),
    #     np.full(np.shape(y_train), np.mean(y_train)),
    # )
    # # plot how well we perform on test data
    # ev.plot_model_vs_real(
    #     np.arange(np.shape(X_test)[0]),
    #     y_test,
    #     y_pred,
    #     y_pred_baseline,
    #     # std=True,
    #     # ystd=y_std,
    # )
    # plot how well we perform on test data
    print(np.shape(ar_model.conf_int()))
    print(np.shape(ar_model.conf_int()[:, 0]))

    ev.plot_ar_model_vs_real(
        np.arange(np.shape(y)[0]),
        np.arange(
            start=np.shape(y)[0] - np.shape(ar_y_pred)[0], stop=np.shape(y)[0], step=1
        ),
        y,
        ar_y_pred,
        conf_int=ar_conf_int,
    )

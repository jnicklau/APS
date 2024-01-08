# train_to_p_lreg.py

import pandas as pd
import numpy as np

import linear_models as lm
import autoregressive_models as arm
import filenames as fn
import reading_data as rd
import evalu as ev
import decision_tree_models as dtm

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)

airleader_files = [fn.d1_air_leader_file]
airflow_files = fn.d1_flow_file
scaler = StandardScaler()
n_cv = 5
n_cv1 = 5
n_cv2 = 1
dg = 2

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
    X, y = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    X, y = rd.scale_Xy(X, y, scaler)
    # X = lm.extend_to_polynomial(X, degree=dg)
    X_train, X_val, X_test, y_train, y_val, y_test = lm.split_train_val_test(
        X, y, 0.1, ps=True
    )

    # ------------------------------------------------------------
    y_pred_array = np.zeros((n_cv1, n_cv2, np.shape(y_test)[0]))
    depths = np.linspace(1, 10, n_cv1)
    mids = np.logspace(-10, -5, n_cv1)  # min_impurity_decrease
    if n_cv1 * n_cv2 > 1:
        ev.print_line("Manual HPS")

    for i in range(n_cv1):
        for j in range(n_cv2):
            lr_model = dtm.decision_tree_train(
                X_train,
                y_train,
                # max_depth=int(depths[i]),
                min_impurity_decrease=mids[i]
                # fit_intercept = True,
                # max_iter = 300
                # positive=True,
                # alphas=np.logspace(-5, 5, 11),
                # alpha = alphas[i],
                # tol = tols[i]
            )
            y_pred = lr_model.predict(X_test)
            y_pred_array[i, j, :] = y_pred.reshape(1, -1)
    metrics_array = ev.comp_and_eval_predictions(y_pred_array, y_test)
    # y_pred, y_std = lr_model.predict(X_test, return_std=True)
    ev.print_metrics_array(metrics_array)
    # ev.plot_learn_curve(lr_model,X_train,y_train,cv = 5)
    # ------------------------------------------------------------
    y_pred_baseline = np.full(np.shape(y_test), np.mean(y_train))
    # ------------------------------------------------------------
    ev.cross_validation(X_train, y_train, lr_model, cv=n_cv)
    # ------------------------------------------------------------
    # #evaluate coefficients
    # ev.coeff_analysis(
    #     lr_model,
    #     air_flow,
    #     plotbool=False,
    #     reggoal="Vi2p",
    # )

    # ------------------------------------------------------------
    # print model metrics (MSE, Rsquared)
    ev.print_model_metrics(
        y_test,
        y_pred,
        y_pred_baseline,
    )

    # ------------------------------------------------------------
    # plot how well we perform on training data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_train)[0]),
        y_train,
        lr_model.predict(X_train),
        np.full(np.shape(y_train), np.mean(y_train)),
        title="Training Data",
    )
    # plot how well we perform on test data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_test)[0]),
        y_test,
        y_pred,
        y_pred_baseline,
        # std=True,
        # ystd=y_std,
        title="Test Data",
    )
    # ------------------------------------------------------------

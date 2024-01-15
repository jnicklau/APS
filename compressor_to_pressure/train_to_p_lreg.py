# train_to_p_lreg.py
import numpy as np
import pandas as pd
import linear_models as lm
import autoregressive_models as arm
import decision_tree_models as dtm
import filenames as fn
import reading_data as rd
import evalu as ev
import model_improvement as mi
import data_preprocessing as dpp
import statsmodels.api as sm
import scipy.stats as stats


from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)

airleader_files = fn.all_air_leader_files
# airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
airleader_files = [fn.short_air_leader_file]

airflow_files = fn.flow_file
# airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
airflow_files = fn.short_flow_file

V_internal_names = [
    "7A Netz 700.5",
    "7A Netz 700.6",
]
V_out_names = [
    "7B Netz 800.1",
    "7C Netz 900.1",
    "7A Netz 700.1",
]
n_cv = 5
n_cv1 = 1
n_cv2 = 1
dg = 2
r1 = 0.2
scaler = StandardScaler()
seperator = 90
reference_date = "2023-11-01"
# ------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    ev.print_line("READING DATA")
    air_leader = rd.fetch_airleader(airleader_files, reference_date)
    air_flow = rd.fetch_airflow(airflow_files, reference_date)
    # ------------------------------------------------------------
    common_times = rd.get_common_times(air_leader, air_flow)
    air_leader, air_flow = rd.put_on_same_time_interval(
        air_leader,
        air_flow,
        common_times,
        method="ffill",
    )
    # rd.print_df_information(air_leader, name="air_leader", nhead=30)
    # rd.print_df_information(air_flow, name="air_flow", nhead=30)
    # ------------------------------------------------------------
    Vi, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    # Vi = dpp.add_difference_column(p, "Netzdruck", Vi, "Netzdruck Diff")
    V_out = Vi.drop(columns=V_internal_names)
    V_simple = dpp.sum_and_remove_columns(V_out, V_out_names, "V_out sum")
    # ------------------------------------------------------------
    data = pd.concat([p, V_simple], axis=1)
    # rd.print_df_information(data, name="data")
    data = dpp.add_time_patterns(data, reference_date)
    # rd.print_df_information(data, name="data with time patterns")
    data_high_in = data[data["Consumption"] >= seperator]
    train_data, val_data, test_data = dpp.split_train_val_test_df(
        data_high_in,
        r1=r1,
        ps=True,
        shuffle=True,
    )
    # rd.print_df_information(train_data,name='train_data')
    # ------------------------------------------------------------
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scalers,
    ) = dpp.get_scaled_xy_from_splitted_df(
        train_data,
        val_data,
        test_data,
        scaler,
    )
    # ------------------------------------------------------------
    depths = np.linspace(1, 10, n_cv1)
    mids = np.logspace(-10, -5, n_cv1)  # min_impurity_decrease
    if n_cv1 * n_cv2 > 1:
        ev.print_line("Manual HPS")

    y_pred_array = np.zeros((n_cv1, n_cv2, np.shape(y_val)[0]))
    resid_aray = np.zeros((n_cv1, n_cv2, np.shape(y_train)[0]))
    ev.print_line("Train")
    for i in range(n_cv1):
        for j in range(n_cv2):
            lr_model = lm.sm_linear_regression_train(
                X_train,
                y_train,
                # max_depth=int(depths[i]),
                # min_impurity_decrease=mids[i]
                # fit_intercept = True,
                # max_iter = 300
                # positive=True,
                # alphas=np.logspace(-5, 5, 11),
                # alpha = alphas[i],
                # tol = tols[i]
            )
            y_pred = lr_model.predict(X_val)
            residuals = -lr_model.predict(X_train) + y_train.ravel()
            y_pred_array[i, j, :] = y_pred.reshape(1, -1)
            resid_aray[i, j, :] = residuals
    ev.print_line("Analyse")
    for i in range(n_cv1):
        for j in range(n_cv2):
            ev.plot_resids(residuals)
            ev.plot_resids_dist(residuals)
            ev.plot_resids_vs_target(
                residuals,
                y_train,
                train_data,
                "Netzdruck",
                kind="hex",
                gridsize=50,
            )
            ev.plot_resids_vs_predictors(
                residuals, X_train, train_data, pcolumns=[1, 2], kind="hex", gridsize=50
            )
            ev.qqplot(residuals, stats.norm, "norm")
    metrics_array = ev.comp_and_eval_predictions(y_pred_array, y_val)
    # y_pred, y_std = lr_model.predict(X_val, return_std=True)
    ev.print_metrics_array(metrics_array)
    # ev.plot_learn_curve(lr_model,X_train,y_train,cv = 5)
    # ------------------------------------------------------------
    y_pred_baseline = np.full(np.shape(y_val), np.mean(y_train))
    # ------------------------------------------------------------
    # ev.cross_validation(X_train, y_train, lr_model, cv=n_cv)
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
        y_val,
        y_pred,
        y_pred_baseline,
    )

    # ------------------------------------------------------------
    # plot how well we perform on training data
    ev.plot_model_vs_real(
        np.arange(np.shape(y_train)[0]),
        y_train,
        lr_model.predict(X_train),
        np.full(np.shape(y_train), np.mean(y_train)),
        title="Training Data",
    )
    # plot how well we perform on test data
    ev.plot_model_vs_real(
        np.arange(np.shape(y_val)[0]),
        y_val,
        y_pred,
        y_pred_baseline,
        # std=True,
        # ystd=y_std,
        title="Validation Data",
    )
    # ------------------------------------------------------------

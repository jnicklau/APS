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
# airleader_files = [fn.short_air_leader_file]

airflow_files = fn.flow_file
# airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
# airflow_files = fn.short_flow_file

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
n_clusters = 2
dg = 1
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
        method="linear",
    )
    # ------------------------------------------------------------
    Vi, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    V_out = Vi.drop(columns=V_internal_names)
    V_simple = dpp.sum_and_remove_columns(V_out, V_out_names, "V_out sum")
    p_diff = pd.DataFrame({"Netzdruck_diff": p["Netzdruck"].diff()})
    p_diff = p_diff.shift(-1)
    p_diff = p_diff.fillna(0)
    # rd.print_df_information(p_diff,name = 'p_diff')
    # ------------------------------------------------------------
    data = pd.concat([p, V_simple], axis=1)
    data = dpp.add_time_patterns(data, reference_date)
    # data = dpp.kmeans_cluster_df(data, n_clusters, n_init="auto")
    spltd_data = dpp.split_train_val_test_df(
        data,
        r1=r1,
        ps=False,
        shuffle=True,
    )
    train_data, val_data, test_data = spltd_data
    # ------------------------------------------------------------
    (
        spltd_X,
        spltd_y,
        scalers,
    ) = dpp.get_scaled_xy_from_splitted_df(spltd_data, dg=dg)

    X_train, X_val, X_test = spltd_X
    y_train, y_val, y_test = spltd_y
    scaler_X, scaler_y = scalers
    # ------------------------------------------------------------

    ev.print_line("Train")
    foo = lm.sm_linear_regression_train
    y_pred_array, resid_array, models = mi.manual_hyp_param_search(
        foo,
        X_train,
        y_train,
        X_val,
        y_val,
        n1=n_cv1,
        n2=n_cv2,
    )

    ev.print_line("Analyze")
    y_pred_baseline = np.full(np.shape(y_val), np.mean(y_train))
    for i in range(n_cv1):
        for j in range(n_cv2):
            y_pred = y_pred_array[i, j, :]
            model = models[i][j]
            # ------------------------------------------------------------
            # ev.plot_learn_curve(model,X_train,y_train,cv = 5)
            # ------------------------------------------------------------
            # ev.cross_validation(X_train, y_train, model, cv=n_cv)
            # ------------------------------------------------------------
            ev.print_line("Results Analysis")
            # # evaluate coefficients
            # ev.coeff_analysis(
            #     model,
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
                model.predict(X_train),
                np.full(np.shape(y_train), np.mean(y_train)),
                title="Training Data",
            )
            # plot how well we perform on test data
            ev.plot_model_vs_real(
                np.arange(np.shape(y_val)[0]),
                y_val,
                y_pred,
                y_pred_baseline,
                title="Validation Data",
            )
            # ------------------------------------------------------------
            ev.print_line("Residual Analysis")
            residuals = resid_array[i, j, :]
            ev.plot_resids(residuals)
            ev.plot_resids_dist(residuals)
            ev.plot_resids_vs_predictors(
                residuals,
                X_train,
                train_data,
                pcolumns=[0, 1, 2],
                kind="hex",
                gridsize=50,
            )
            ev.plot_resids_vs_pattern_predictors(
                residuals,
                train_data,
                duration="dw",
                kind="hex",
                gridsize=100,
            )

            # ev.qqplot(residuals, stats.norm, "norm")
    # metrics_array = ev.comp_and_eval_predictions(y_pred_array, y_val)
    # ev.print_metrics_array(metrics_array)

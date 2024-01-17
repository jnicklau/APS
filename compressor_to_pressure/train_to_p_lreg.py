# train_to_p_lreg.py
import numpy as np
import pandas as pd
import time
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
import combining_models as cm


from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)
from sklearn.tree import DecisionTreeRegressor


V_internal_names = [
    "7A Netz 700.5",
    "7A Netz 700.6",
]
V_out_names = [
    "7B Netz 800.1",
    "7C Netz 900.1",
    "7A Netz 700.1",
]
airleader_files = fn.all_air_leader_files
airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
# airleader_files = [fn.short_air_leader_file]

airflow_files = fn.flow_file
airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
# airflow_files = fn.short_flow_file
n_cv = 10
n_hp1 = 5
n_hp2 = 5
n_clusters = 1
dg = 1
r1 = 0.95
scaler = StandardScaler()
seperator = 90
reference_date = "2023-11-01"
# ------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    # ------------------------------------------------------------
    ev.print_line("READING DATA")
    air_leader = rd.fetch_airleader(airleader_files, reference_date)
    air_flow = rd.fetch_airflow(airflow_files, reference_date)
    common_times = rd.get_common_times(air_leader, air_flow)
    air_leader, air_flow = rd.put_on_same_time_interval(
        air_leader,
        air_flow,
        common_times,
        method="linear",
    )
    reading_time = time.time()
    # ------------------------------------------------------------
    ev.print_line("PREPROCESSING DATA")
    Vi, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    V_out = Vi.drop(columns=V_internal_names)
    V_simple = dpp.sum_and_remove_columns(V_out, V_out_names, "V_out sum")
    p_diff = pd.DataFrame({"Netzdruck_diff": p["Netzdruck"].diff()})
    p_diff = p_diff.shift(-1)
    p_diff = p_diff.fillna(0)
    # ------------------------------------------------------------
    data = pd.concat([p_diff, V_out], axis=1)
    data = dpp.add_time_patterns(data, reference_date)
    # data = dpp.kmeans_cluster_df(data, n_clusters, n_init="auto")
    spltd_data = dpp.split_train_val_test_df(
        data,
        r1=r1,
        ps=False,
        shuffle=True,
    )
    scaled_data = dpp.scale_splitted_data(spltd_data, scaler)
    train_data, val_data, test_data = scaled_data
    # rd.print_df_information(train_data, name="train_data")
    # ------------------------------------------------------------
    X, y = dpp.get_xy_from_splitted_df(scaled_data)
    X = dpp.tuple_polynomial_extension(X, dg)
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y
    preprocess_time = time.time()
    # ------------------------------------------------------------

    # foo = dtm.random_forest_train
    # mi.run_and_eval_manual_hyp_param_search(
    #     foo, X_train, y_train, X_val, y_val, ra=False,
    #     pname1 = 'max_depth', n1=5, s1 = 1, e1 = 100,
    #     pname2 = "min_samples_leaf", n2=1,s2 = 1, e2 = 10,
    #     n_estimators = 10
    # )

    # ------------------------------------------------------------

    # model = dtm.random_forest_train(
    #     df=train_data, max_depth=10, min_samples_leaf=1, n_estimators=10
    # )

    # base_regr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=1)
    # model = cm.adaboost(base_regr, df=train_data,)
    # model = lm.linear_regression_train(df = train_data,)

    train_time = time.time()
    # ------------------------------------------------------------
    # ev.plot_tree_from_model(model)
    # scores = ev.cross_validation(df=val_data, model=model, cv=5)
    # print(scores)

    y_pred_val = model.predict(val_data.iloc[:, 1:])
    y_pred_train = model.predict(train_data.iloc[:, 1:])
    residuals = y_val.ravel() - y_pred_val

    y_pred_baseline_train = np.full(np.shape(y_train), np.mean(y_train))
    ev.print_model_metrics(y_train, y_pred_train, y_pred_baseline_train)
    y_pred_baseline_val = np.full(np.shape(y_val), np.mean(y_train))
    ev.print_model_metrics(y_val, y_pred_val, y_pred_baseline_val)

    evaluation_time = time.time()
    # ev.full_residual_analysis(residuals,val_data)

    # ev.plot_model_vs_real(
    #     y_train,
    #     y_pred_train,
    #     y_pred_baseline_train,
    # )
    # ev.plot_model_vs_real(
    #     y_val,
    #     y_pred_val,
    #     y_pred_baseline_val,
    # )

    ev.print_line("Computation Time Analysis")
    print("Time for Reading", "%.3f" % (reading_time - start_time))
    print("Time for Preprocessing", "%.3f" % (preprocess_time - reading_time))
    print("Time for Training", "%.3f" % (train_time - preprocess_time))
    print("Time for Evaluating", "%.3f" % (evaluation_time - train_time))

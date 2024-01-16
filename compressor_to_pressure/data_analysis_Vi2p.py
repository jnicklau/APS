# data_analysis_Vi2p.py
import pandas as pd
import numpy as np
import filenames as fn
import reading_data as rd
import evalu as ev
import data_analyis as da
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)
import seaborn as sns
import matplotlib.pyplot as plt
import data_preprocessing as dpp

airleader_files = fn.all_air_leader_files
# airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
# airleader_files = [fn.short_air_leader_file]

airflow_files = fn.flow_file
# airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
# airflow_files = fn.short_flow_file

scaler = StandardScaler()
V_internal_names = [
    "7A Netz 700.5",
    "7A Netz 700.6",
]
V_out_names = [
    "7B Netz 800.1",
    "7C Netz 900.1",
    "7A Netz 700.1",
]
n_clusters = 2
reference_date = "2023-11-01"

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
    # rd.print_df_information(air_leader, name="air_leader")
    # rd.print_df_information(air_flow, name="air_flow")
    # ------------------------------------------------------------
    Vi, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    Vi = dpp.add_difference_column(p, "Netzdruck", Vi, "Netzdruck Diff")
    V_out = Vi.drop(columns=V_internal_names)
    V_simple = dpp.sum_and_remove_columns(V_out, V_out_names, "V_out sum")
    data = pd.concat([p, V_simple], axis=1)
    # data = dpp.kmeans_cluster_df(data, n_clusters, n_init="auto")
    data = dpp.add_time_patterns(data, reference_date)

    # X, y = Vi.to_numpy(), p.to_numpy()
    # X, y = dpp.scale_Xy(X, y, scaler)
    # ------------------------------------------------------------
    # Apply the default theme
    sns.set_theme()

    # da.make_hists(data, hue="cluster", bins=64, kde=False)
    # for i in range(n_clusters):
    #     da.heat_corr(data[data["cluster"] == i], "pearson", cmap="coolwarm", annot=True)

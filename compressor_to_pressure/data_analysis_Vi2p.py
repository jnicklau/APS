# data_analysis_Vi2p.py
import pandas as pd
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

airleader_files = fn.all_air_leader_files
# airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
# airleader_files = [fn.short_air_leader_file]

airflow_files = fn.flow_file
# airflow_files = fn.d1_flow_file
# airflow_files = fn.h1_flow_file
# airflow_files = fn.short_flow_file

scaler = StandardScaler()
V_out_names = [
    "7B Netz 800.1",
    "7A Netz 700.5",
    "7A Netz 700.6",
    "7C Netz 900.1",
    "7A Netz 700.1",
]

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
        method="ffill",
    )
    # rd.print_df_information(air_leader, name="air_leader")
    # rd.print_df_information(air_flow, name="air_flow")
    # ------------------------------------------------------------
    Vi, p = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    Vi = rd.add_difference_column(p, "Netzdruck", Vi, "Netzdruck Diff")
    # rd.print_df_information(p, name="p")
    # rd.print_df_information(Vi, name="Vi")
    data = pd.concat([p, Vi], axis=1)
    # rd.print_df_information(data, name="data",nhead = 40)
    rd.sum_and_remove_columns(data, V_out_names, "V_out")
    # rd.print_df_information(data, name="data_simplified",nhead = 40)
    # X, y = V_out.to_numpy(), p.to_numpy()
    # X, y = rd.scale_Xy(X, y, scaler)
    # ------------------------------------------------------------
    # Apply the default theme
    sns.set_theme()

    for i in range(data.shape[1]):
        #     sns.displot(data, x=data.columns[i], kde=True,bins = 50)
        print("sum of data.[%s]: " % data.columns[i], data[data.columns[i]].sum())
    #     plt.show()

    # sns.relplot(x=p.index, y=data["Netzdruck Diff"],)
    # plt.show()

    # sns.lmplot(
    #     data=data,
    #     x=V_out.columns[0],
    #     y="Netzdruck",
    # )
    # sns.displot(data, x="Netzdruck Diff", kde=True,bins = 50)
    # # sns.displot(data, x="Netzdruck Diff", kind="kde")
    # plt.show()
    # sns.pairplot(data)
    # plt.show()
    # sns.jointplot(x="Netzdruck Diff", y="Consumption", data=data, kind="reg")
    # plt.show()
    methods = [
        "pearson",
        # 'kendall',
        # 'spearman',
    ]
    corrs = []
    for i in range(len(methods)):
        corr = da.heat_corr(
            data,
            methods[i],
            cmap="coolwarm",
            annot=True,
        )
        corrs.append(corr)
    # for i in range(len(methods)):
    #     diff = corrs[i]-corrs[(i+1) % len(methods)]
    #     print('diff.max(): ',diff.max())
    # plt.show()

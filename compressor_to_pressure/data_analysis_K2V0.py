# data_analysis_K2V0.py
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
import data_preprocessing as dpp
import scipy.stats as stats


airleader_files = fn.all_air_leader_files
# airleader_files = [fn.d1_air_leader_file]
# airleader_files = [fn.h1_air_leader_file]
# airleader_files = [fn.short_air_leader_file]
reference_date = "2023-11-01"

if __name__ == "__main__":
    # ------------------------------------------------------------
    ev.print_line("READING DATA")
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    air_leader = rd.fetch_airleader(airleader_files, reference_date="2023-11-01")
    # ------------------------------------------------------------
    K_AE1, K_R2, V0 = rd.extract_training_data_from_df(
        [air_leader, compressors], reggoal="K2V0"
    )
    K = pd.concat([K_R2, K_AE1], axis=1)
    data = pd.concat([V0, K], axis=1)
    rd.print_df_information(K)
    # ------------------------------------------------------------
    da.describe_df(data)
    # da.make_hists(data.iloc[:,13:])
    # corr = da.heat_corr(data,'pearson',
    #         cmap="coolwarm",
    #         # annot=True,
    #         )

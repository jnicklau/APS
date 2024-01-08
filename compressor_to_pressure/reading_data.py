import numpy as np
import pandas as pd
import regex as re
import math
from sklearn.preprocessing import (
    normalize,
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)

import evalu as ev
import filenames as fn
from filenames import *


def fetch_airleader(fname):
    ev.print_line("FETCHING AIRLEADER")
    col_list = get_significant_columns(filetype="airleader")
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    li = read_flist([fname], col_list, compressors)
    air_leader = pd.concat(li, axis=0)
    return air_leader


def fetch_airflow(fname):
    ev.print_line("FETCHING AIRFLOW")
    col_list = get_significant_columns(filetype="volumenstrom")
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    li = read_flist([fname], col_list, compressors, "volumenstrom")
    air_flow = pd.concat(li, axis=0)
    return air_flow


def get_significant_columns(filetype="airleader"):
    """
    This function returns list of indices for which to select the columns
    """
    if filetype == "airleader":
        air_leader_col_list = [0, 1, 2, 3, 4, 6, 7, 8, 11, 15, 16, 17]
        for i in range(13):
            k = air_leader_col_list[-1]
            addlist = [1, 4, 8, 9, 10]
            addlist2 = [x + 1 + k for x in addlist]
            air_leader_col_list += addlist2
        return air_leader_col_list
    if filetype == "volumenstrom":
        return [0, 1, 2, 3, 4, 5]


def print_df_information(df, **kwargs):
    name = kwargs.get("name", "DataFrame")
    nhead = kwargs.get("nhead", 3)
    ntail = kwargs.get("ntail", 3)
    print("\n %s.head\n" % name, df.head(nhead))
    print("%s.tail\n" % name, df.tail(ntail))
    print("shape of %s:" % name, df.shape)
    if kwargs.get("indunique", False):
        print("Is index in %s unique?" % name, df.index.is_unique)


def extract_training_data_from_df(dfs, reggoal="K2V0"):
    """
    takes a list of DataFrames
    (either 'df' and 'comp_df' (air_leader and compressor_list),
     or 'net_df' and 'p_df' (air_flow and air_leader) )
    as arguments and returns an X,y trainable dataset of numpy arrays
    """
    ev.print_line("EXTRACT TRAINING DATA")
    if reggoal == "K2V0":
        """
        compressors K --> volume flow "consumption" V_0
        """
        scaler = StandardScaler()
        df = dfs[0]
        comp_df = dfs[1]
        """ K shall be of form:  K =  [seconds, compressor motor state/flow rate]
        """
        consumption = df["Consumption"].to_numpy()
        K_V_dot = np.zeros((df.index.size, comp_df.index.size))
        K_R2 = np.zeros((df.index.size, comp_df.index.size))
        for i in range(comp_df.index.size):
            n = comp_df.loc[i, "Airleaderkanal"]
            columnAE1 = "%s.AE1" % n
            columnR2 = "%s.R2" % n
            K_V_dot[:, i] = df[columnAE1].to_numpy()
            K_R2[:, i] = df[columnR2].to_numpy()
        # print(K_R2)
        # print(K_V_dot)
        K = np.concatenate((K_R2, K_V_dot), axis=1)
        K = scaler.fit_transform(K)
        consumption = scaler.fit_transform(consumption.reshape(-1, 1))
        return K, consumption
    if reggoal == "Vi2p":
        """volume flow at i positions V_i--> pressure p
        Vout shall be of form:
            Vout = [seconds, flow rate]
        """
        net_df = dfs[0]
        p_df = dfs[1]
        p = p_df["Master.AE1 (Netzdruck)"].to_numpy()
        """ use "consumption" to estimate pressure p """
        net_df["Consumption"] = p_df["Consumption"]
        """ extract the information about measurement point 700.1 """
        p_df, net_df["7A Netz 700.1"] = extract_flow7A(p_df)
        V_out = net_df.to_numpy()
        """ inverse p according to ideal gas law"""
        # p = 1 / p
        """ get pressure differences and adjust shape of V_out"""
        # p_diff = np.diff(p)
        # new_column = np.zeros((V_out.shape[0], 1))
        # new_column[1:, 0] = p_diff
        # V_out = np.hstack((V_out, new_column))
        # V_out = V_out[1:,]
        # p = p[1:]
        return V_out, p


def scale_Xy(X, y, scaler):
    """rescaling to a given scaler"""
    ev.print_line("FETCHING AIRLEADER")
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y


def get_max_volume_flow(df, i):
    vdot = df.loc[i, "mechan_Fluss_nom"]
    if vdot == math.isnan(vdot):
        # get the maximum volume flow in cm/min
        # via conversion from electric power in kW
        return df.loc[i, "Leistung_max_nom"] * 0.15
    else:
        return vdot


def get_compressornames(df, filetype="airleader"):
    """
    return a dict with the names of the compressors as keys and the number as the Air Leader uses it as values
    """
    if filetype == "airleader":
        # get the numbers of air leader channels by taking the first value of every Number column
        compressornames = {}
        for column in df.columns:
            if "Number" in column:
                compressornames[df[column].iloc[0]] = column.replace(".Number", "")
        inv_compressornames = {v: k for k, v in compressornames.items()}
        return inv_compressornames


def reformat_df(df, comp_df, filetype="airleader"):
    """
    reformats the DataFrame such that a 'seconds' column leads and reformats the title
    in a specific manner, dependent on the filetype
    """
    compressors_dict = get_compressornames(df)
    if filetype == "airleader":
        # Rearrange for seconds series to lead and
        # fill in columns for 'Consumption', 'Master.AE1 (Netzdruck)','7-bar A-Netz', 'Master.AE4'
        new_df = pd.DataFrame()
        first_list = [
            "seconds",
            "Consumption",
            "Master.AE1 (Netzdruck)",
            "7-bar A-Netz",
            "Master.AE4",
        ]
        for element in first_list:
            new_df[element] = df[element].astype(float)

        # Replace specific values in the dictionary based on DataFrame column titles
        keystring = ""
        # patterns to look for in columns worth replacing
        patterns_to_replace1 = list(compressors_dict.keys())
        pattern1 = re.compile("|".join(map(re.escape, patterns_to_replace1)))
        # patterns to get rid of in title when using the dict inv_compressornames
        patterns_to_replace2 = [".Number", ".R2", ".B", ".AE1", ".M"]
        pattern2 = re.compile("|".join(map(re.escape, patterns_to_replace2)))
        for column in df.columns:
            match = re.search(pattern1, column)
            if match:
                keystring = pattern2.sub("", column)
                new_name = pattern1.sub(str(compressors_dict[keystring]), column)
                new_df[new_name] = df[column].astype(
                    float
                )  # Replace with the desired column
        # get rid of every dataframe column with '.Number' in it
        df = new_df.filter(regex="^(?!.*.Number).*$", axis=1)
        # get rid of 7bar-A-Netz and save it seperately
        df.set_index("seconds", inplace=True)
        # edit air_leader, so that Motor=ON --> AE1 = Leistung_max_gemessen
        # for unregulated compressors
        for i in range(comp_df.index.size):
            n = comp_df.loc[i, "Airleaderkanal"]
            columnR2 = "%s.R2" % n
            columnM = "%s.M" % n
            columnAE1 = "%s.AE1" % n
            max_V_dot = get_max_volume_flow(comp_df, i)
            # Conditionally set values in columnAE1 to max_V_dot
            df.loc[
                (df[columnR2] == 1) & (df[columnM] == 1) & (df[columnAE1] == 0),
                columnAE1,
            ] = max_V_dot
        return df

    if filetype == "volumenstrom":
        # Rearrange for seconds series to lead and give series more informative names
        new_df = pd.DataFrame()
        for i in range(len(df.columns)):
            oldname = "%s" % df.columns[i - 1]
            patterns_to_replace = [
                "Trocknung-Durchfluss  ",
                "Trocknung-Durchfluss ",
                " [cm/min]",
            ]
            replacement = ""
            pattern = re.compile("|".join(map(re.escape, patterns_to_replace)))
            newname = pattern.sub(replacement, oldname, count=0)
            new_df[newname] = df[oldname].astype(float)
        # drop the empty column "7A Netz 700.1"
        df = new_df.drop(columns=["7A Netz 700.1"])
        # the seconds column the index
        df.set_index("seconds", inplace=True)
        # remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        return df


def extract_flow7A(df):
    flow7Anet = pd.DataFrame()
    flow7Anet.index = df.index
    flow7Anet["7A Netz 700.1"] = df["7-bar A-Netz"]
    df = df.drop(columns=["7-bar A-Netz"])
    # print("read flow7Anet\n", flow7Anet.head(4))
    return df, flow7Anet


def get_seconds_column(
    df, date="2023-11-01", filetype="airleader", datepattern="%d.%m.%Y %H:%M:%S"
):
    """
    This function transforms the file date and time inputs into a single seconds float column
    with a reference date
    """
    reference_date = pd.to_datetime(date)
    if filetype == "airleader":
        # Convert the 'Date' and 'TimeString' columns to datetime format
        df["time"] = df["Date"] + " " + df["TimeString"].astype(str)
        df["time"] = pd.to_datetime(df["time"], format=datepattern)
        # Convert the datetime values to seconds via a reference date and write into new series
        df["seconds"] = (df["time"] - reference_date).dt.total_seconds()
        # Drop the 'Date'/'Time'/'TimeString' column
        df = df.drop(columns=["Date", "Time", "TimeString"])
        return df
    if filetype == "volumenstrom":
        # rename 'Sps_datum' column
        df["Datum"] = df["Sps_datum"]
        df = df.drop(columns=["Sps_datum"])
        # Convert the 'Datum' column to datetime format
        df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M:%S")
        # Convert the datetime values to seconds via a reference date and write into new series
        df["seconds"] = (df["Datum"] - reference_date).dt.total_seconds()
        # Drop the 'Datum' column
        df = df.drop(columns=["Datum"])
        return df


def get_common_times(df1, df2, interval=1):
    """
    return an np.array for interpolating the time via the index column
    """
    return np.arange(
        min(df1.index.min(), df2.index.min()),
        max(df1.index.max(), df2.index.max()) + 1,
        interval,
    )


def put_on_same_time_interval(df1, df2, common_index, **kwargs):
    """
    fill NaN values between two known values with value according to
    a linearly time-dependent function
    method: should either be ffill or index
    """
    ev.print_line("PUT ON COMMON TIMES")
    df2_int = df2.reindex(common_index).interpolate(
        method=kwargs.get("method", "index"),
    )
    df1_int = df1.reindex(common_index).interpolate(
        method=kwargs.get("method", "index"),
    )
    # fill start and end NaNs by taking nearest value available
    df1_int_fna = df1_int.bfill()
    df2_int_fna = df2_int.bfill()
    return df1_int_fna, df2_int_fna


def read_flist(flist, col_list, compressors, filetype="airleader"):
    """
    read files from a flist
    """
    # read files
    li = []
    if filetype == "airleader":
        li7 = []
    for i, filename in enumerate(flist):
        # i: iterator of files
        # print(i,filename)
        if filetype == "airleader":
            if i < 2:
                df = pd.read_csv(filename, sep=";", usecols=col_list)
                df = get_seconds_column(df, filetype=filetype)
            else:
                df = pd.read_csv(filename, sep=";", usecols=col_list, header=1)
                df = get_seconds_column(
                    df, filetype=filetype, datepattern="%d-%m-%Y %H:%M:%S"
                )
        elif filetype == "volumenstrom":
            df = pd.read_csv(filename, sep=";", usecols=col_list, header=1)
            df = get_seconds_column(df, filetype=filetype)
        df.replace(",", ".", regex=True, inplace=True)  # replace ',' by '.'
        df = reformat_df(df, compressors, filetype=filetype)
        li.append(df)
    return li


# ===========================================================
if __name__ == "__main__":
    print("This is the 'reading_data.py' file")

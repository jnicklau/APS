# train_lreg_K_to_V_dot_0.py
import pandas as pd
import numpy as np
import linear_models as lm
import filenames as fn
import reading_data as rd
import feature_analysis as fa
import evalu as ev
from sklearn.preprocessing import (
    normalize,
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
)


airleader_files = fn.all_air_leader_files
scaler = StandardScaler()
reggoal = "K2V0"
# ------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    ev.print_line("READING DATA")
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    air_leader = rd.fetch_airleader(airleader_files)
    rd.print_df_information(air_leader, name="air_leader")

    # # ------------------------------------------------------------
    # fa.time_frequency_analysis(air_leader["Consumption"])
    # fa.time_frequency_analysis(air_leader["Master.AE1 (Netzdruck)"])

    # ------------------------------------------------------------
    X, y = rd.extract_training_data_from_df([air_leader, compressors], reggoal="K2V0")
    X, y = rd.scale_Xy(X, y, scaler)
    # X = lm.extend_to_polynomial(X,degree = 2)
    X_train, X_val, _, y_train, y_val, _ = lm.split_train_val_test(X, y, 0.8, ps=True)
    # print("X_train[0:20,:] \n", X_train[0:20, :])
    # print("y_train[0:25] \n", y_train[0:25])

    # # ------------------------------------------------------------
    lr_model = lm.linear_lasso_regr_train(
        X_train,
        y_train,
        # positive=True,
        alphas=np.logspace(-5, 5, 31),
        max_iter=3000,
        # alpha=3.16,
    )
    model = lr_model
    # ------------------------------------------------------------
    y_pred = lr_model.predict(X_val)
    y_pred_baseline = np.full(np.shape(y_pred), np.mean(y_train))
    # ev.plot_learn_curve(lr_model,X_train,y_train,cv = 5)

    # ------------------------------------------------------------
    ev.coeff_analysis(
        lr_model,
        compressors,
        plotbool=True,
        reggoal="K2V0",
    )
    ev.print_model_metrics(
        y_val,
        y_pred,
        y_pred_baseline,
    )

    # plot how well do we perform on training data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_train)[0]),
        y_train,
        lr_model.predict(X_train),
        np.full(np.shape(y_train), np.mean(y_train)),
        title="Training Data",
        reggoal=reggoal,
    )
    # plot how well do we perform on test data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_val)[0]),
        y_val,
        y_pred,
        y_pred_baseline,
        title="Validation Data",
        reggoal=reggoal,
    )

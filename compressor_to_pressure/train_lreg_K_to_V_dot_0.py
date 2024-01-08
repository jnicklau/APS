# train_lreg_K_to_V_dot_0.py

import pandas as pd
import linear_models as lm

from linear_models import *
from support_vector_models import *
import filenames as fn
from reading_data import *
from feature_analysis import *
import evalu as ev
from evalu import *


# ------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    # # ------------------------------------------------------------
    ev.print_line("READING DATA")
    col_list = get_significant_columns(filetype="airleader")
    li = read_flist(fn.all_air_leader_files, col_list, compressors)
    # li = read_flist([fn.short_air_leader_file], col_list, compressors)
    air_leader = pd.concat(li, axis=0)
    print_df_information(air_leader, name="air_leader")
    # print("len(air_leader['10.R2'].unique()) : ", len(air_leader["10.R2"].unique()))

    # # ------------------------------------------------------------
    # time_frequency_analysis(air_leader["Consumption"])
    # time_frequency_analysis(air_leader["Master.AE1 (Netzdruck)"])

    # ------------------------------------------------------------
    ev.print_line("EXTRACT TRAINING DATA")
    X, y = extract_training_data_from_df([air_leader, compressors], reggoal="K2V0")
    # X = lm.extend_to_polynomial(X,degree = 2)
    myseed = 42
    X_train, X_val, X_test, y_train, y_val, y_test = lm.split_train_val_test(
        X, y, 0.8, ps=True
    )
    # print("X_train[0:20,:] \n", X_train[0:20, :])
    # print("y_train[0:25] \n", y_train[0:25])

    # # ------------------------------------------------------------
    ev.print_line()
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
    y_pred = model.predict(X_test)
    y_pred_baseline = np.full(np.shape(y_pred), np.mean(y_train))
    # ev.plot_learn_curve(lr_model,X_train,y_train,cv = 5)

    # ------------------------------------------------------------
    ev.coeff_analysis(
        model,
        compressors,
        plotbool=True,
        reggoal="K2V0",
    )
    ev.print_model_metrics(
        y_test,
        y_pred,
        y_pred_baseline,
    )

    # plot how well do we perform on training data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_train)[0]),
        y_train,
        model.predict(X_train),
        np.full(np.shape(y_train), np.mean(y_train)),
        title="Training Data",
    )
    # plot how well do we perform on test data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_test)[0]),
        y_test,
        y_pred,
        y_pred_baseline,
        title="Test Data",
    )

# train_to_p_lreg.py

import pandas as pd
import numpy as np

import linear_models as lm
import autoregressive_models as arm
import filenames as fn
import reading_data as rd
import evalu as ev
import decision_tree_models as dtm
from sklearn.model_selection import cross_val_score



# ------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------
    ev.print_line('READING DATA')
    compressors = pd.read_table(fn.compressor_list_file, sep=",")
    
    # # ------------------------------------------------------------
    col_list = rd.get_significant_columns(filetype="airleader")
    # li = rd.read_flist([fn.short_air_leader_file], col_list, compressors)
    li = rd.read_flist([fn.d1_air_leader_file], col_list, compressors)
    # li = rd.read_flist(fn.all_air_leader_files, col_list, compressors)
    air_leader = pd.concat(li, axis=0)
    # rd.print_df_information(air_leader, name="air_leader", nhead=30)

    # ------------------------------------------------------------
    col_list = rd.get_significant_columns(filetype="volumenstrom")
    # li = rd.read_flist([fn.short_flow_file], col_list, compressors, "volumenstrom")
    li = rd.read_flist([fn.d1_flow_file], col_list, compressors, "volumenstrom")
    # li = rd.read_flist([fn.flow_file], col_list, compressors, "volumenstrom")
    air_flow = pd.concat(li, axis=0)
    # rd.print_df_information(air_flow, name="air_flow", nhead=30)

    # ------------------------------------------------------------
    ev.print_line('PUT ON COMMON TIMES')
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
    ev.print_line('EXTRACT TRAINING DATA')
    X, y = rd.extract_training_data_from_df([air_flow, air_leader], reggoal="Vi2p")
    X = lm.extend_to_polynomial(X,degree = 2)
    myseed = 42
    X_train, X_val, X_test, y_train, y_val, y_test = lm.split_train_val_test(
        X, y, 0.1, ps=True
    )

    # ------------------------------------------------------------
    n_cv1 = 5
    n_cv2 = 1
    y_pred_array = np.zeros((n_cv1, n_cv2, np.shape(y_test)[0]))
    depths = np.linspace(1,100,n_cv1)
    mids = np.linspace(0,1,n_cv2) #min_impurity_decrease 
    if n_cv1 *n_cv2 >1:
        ev.print_line('Manual CV')

    for i in range(n_cv1):
        for j in range(n_cv2):
            lr_model = dtm.decision_tree_train(
                X_train,
                y_train,
                max_depth = int(depths[i]),
                # min_impurity_decrease = mids[j]
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
    # ev.print_line('CROSS VALIDATION')
    # scores = cross_val_score(lr_model,X_train,y_train.ravel(),cv = 5)
    # print("%0.2f mean with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
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
        title = 'Training Data'
    )
    # plot how well we perform on test data
    ev.plot_model_vs_real(
        np.arange(np.shape(X_test)[0]),
        y_test,
        y_pred,
        y_pred_baseline,
        # std=True,
        # ystd=y_std,
        title = 'Test Data'
    )
    # ------------------------------------------------------------

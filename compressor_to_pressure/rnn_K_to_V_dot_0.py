import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.initializers import GlorotNormal

from reading_data import *
from filenames import *
from feature_analysis import *


def build_rnn_model(seed=None, **kwargs):
    # Set random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    GlorotNormal(seed)
    # Build the RNN model
    model = Sequential()
    # Add a SimpleRNN layer with provided units
    model.add(
        SimpleRNN(
            units=kwargs.get("units", 50),
            activation=kwargs.get("activation", "relu"),
            input_shape=kwargs.get("input_shape", (10, 2)),
        )
    )
    # Add a Dense layer with one output unit (for regression problems)
    model.add(Dense(units=kwargs.get("output_units", 1)))
    # Compile the model
    model.compile(
        optimizer=kwargs.get("optimizer", "adam"),
        loss=kwargs.get("loss", "mean_squared_error"),
    )
    model.summary()
    return model


# ===========================================================
if __name__ == "__main__":
    # ------------------------------------------------------------

    compressors = pd.read_table(compressor_list_file, sep=",")
    # ------------------------------------------------------------

    col_list = get_significant_columns(filetype="airleader")
    li = read_file_list(all_air_leader_files, col_list, compressors)
    air_leader = pd.concat(li, axis=0)

    print("read and reformatted AirLeader\n", air_leader.head(4))
    print("shape of AirLeader:\n", air_leader.shape)
    print("Are values in AirLeader unique?\n", air_leader.index.is_unique)

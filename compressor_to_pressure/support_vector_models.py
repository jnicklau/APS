# support_vector_models.py
from sklearn.svm import LinearSVR
import numpy as np


def lsvm_train(X, y, **kwargs):
    print("SupportVectorRegression")
    print(np.shape(y))
    regr = LinearSVR()
    regr.fit(X, np.ravel(y))
    return regr

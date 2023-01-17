import pickle
import numpy as np

def clf_predict(x, model):
    x[0] = 20 - x[0]
    x[1] = 20 - x[1]
    pred = model.predict(np.array([x]))

    return pred

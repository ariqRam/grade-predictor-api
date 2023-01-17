import pickle
import numpy as np

def clf_predict(x):
    x[0] = 20 - x[0]
    x[1] = 20 - x[1]
    model = pickle.load(open('clf/clf_model.sav', 'rb'))
    pred = model.predict(np.array([x]))

    return pred

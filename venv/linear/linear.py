import pickle
import numpy as np

def linear_predict(data, model):
    # data is a 2d numpy array
    model = pickle.load(open('linear/linear_model.sav', 'rb'))

    pipeline = pickle.load(open('linear/standard_scaler.sav', 'rb'))
    
    data = pipeline.transform(data)
    
    pred = model.predict(data)

    return pred

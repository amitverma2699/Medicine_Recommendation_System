import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import Customexception
from src.logger import logging

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dumb(obj, file_obj)
            
    except Exception as e:
        raise Customexception(e,sys)
    

def Evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)
            
            #Predict Testing Data
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report 
    except Exception as e:
        logging.info('Exception occured during model training')
        raise Customexception(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured while load_object function in utils')
        raise Customexception(e,sys)
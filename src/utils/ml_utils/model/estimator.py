from src.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os 
import sys 

from src.exception.exception import DeliveryTimeException
from src.logging.logger import logging

class DeliveryPredictionModel:
    def __init__(self, model):
        try:
            self.model = model
        except Exception as e:
            raise DeliveryTimeException(e, sys)
        
    def predict(self, x):
        try:
            y_hat = self.model.predict(x)
            return y_hat
        except Exception as e:
            raise DeliveryTimeException(e, sys)
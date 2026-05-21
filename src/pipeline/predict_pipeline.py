import sys
import pandas as pd
import os 

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(
            self,
            trip_miles,
            hour,
            day_of_week,
            is_weekend,
            wait_time,
            is_uber,
            is_airport,
            PULocationID,
            DOLocationID,
            congestion_surcharge,
            shared_request_flag):


        self.trip_miles = trip_miles
        self.hour = hour
        self.day_of_week = day_of_week
        self.is_weekend = is_weekend
        self.wait_time = wait_time
        self.is_uber = is_uber
        self.is_airport = is_airport
        self.PULocationID = PULocationID
        self.DOLocationID = DOLocationID
        self.congestion_surcharge = congestion_surcharge
        self.shared_request_flag = shared_request_flag
    
    def get_data_as_data_frame(self):
        try:
           custom_data_input_dict = {
                "trip_miles": [self.trip_miles],
                "hour": [self.hour],
                "day_of_week": [self.day_of_week],
                "is_weekend": [self.is_weekend],
                "wait_time": [self.wait_time],
                "is_uber": [self.is_uber],
                "is_airport": [self.is_airport],
                "PULocationID": [self.PULocationID],
                "DOLocationID": [self.DOLocationID],
                "congestion_surcharge": [self.congestion_surcharge],
                "shared_request_flag": [self.shared_request_flag]
            }
           
           return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    
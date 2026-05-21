from flask import Flask,request,render_template
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(

            trip_miles=float(request.form.get('trip_miles')),

            hour=int(request.form.get('hour')),

            day_of_week=request.form.get('day_of_week'),

            is_weekend=int(request.form.get('is_weekend')),

            wait_time=float(request.form.get('wait_time')),

            is_uber=int(request.form.get('is_uber')),

            is_airport=int(request.form.get('is_airport')),

            PULocationID=int(request.form.get('PULocationID')),

            DOLocationID=int(request.form.get('DOLocationID')),

            congestion_surcharge=float(request.form.get('congestion_surcharge')),

            shared_request_flag=int(request.form.get('shared_request_flag'))

)


        pred_df= data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        final_result = round(results[0], 2)
        return render_template('home.html',results=final_result)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)


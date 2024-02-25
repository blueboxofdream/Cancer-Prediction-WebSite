import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from LungCancers import LungCancer
import numpy as np
import joblib
import pandas as pd

app = FastAPI()

origins = ["http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # expose_headers=["*"],
    # middleware_class=CORSMiddleware
)

try:
    joblib_in = open("model_fit.joblib", "rb")
    model_fit = joblib.load(joblib_in)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_fit = None

@app.get('/')
async def index():
    return {'message': 'Hello, World'}

# @app.get('/{name}')
# async def get_name(name: str):
#     return {'Welcome To Cancer Detecting Application': f'{name}'}


@app.post('/predict')
async def predict_lungcancer(data1: LungCancer):
    try:
        # print("Hello World")
        data = data1.model_dump()
        AGE = data['AGE']
        SMOKING = data['SMOKING']
        YELLOW_FINGERS = data['YELLOW_FINGERS']
        ANXIETY = data['ANXIETY']
        PEER_PRESSURE = data['PEER_PRESSURE']
        CHRONIC_DISEASE = data['CHRONIC_DISEASE']
        WHEEZING = data['WHEEZING']
        ALCOHOL_CONSUMING = data['ALCOHOL_CONSUMING']
        COUGHING = data['COUGHING']
        SHORTNESS_OF_BREATH = data['SHORTNESS_OF_BREATH']
        SWALLOWING_DIFFICULTY = data['SWALLOWING_DIFFICULTY']
        CHEST_PAIN = data['CHEST_PAIN']
        GENDER_NEW = data['GENDER_NEW']

        if model_fit is not None:
            # print("Hello World3")
            prediction_value = model_fit.predict([[AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, GENDER_NEW]])
            # print("Hello World4")

            if prediction_value[0] < 0.5:
                prediction = "You have low Risk of Lung Cancer"
            else:
                prediction = "You have high of risk of Lung Cancer"
        else:
            prediction = "Error: Model not loaded successfully."

    except Exception as e:
        print(f"An error occurred while processing the request: {e}")
        prediction = "Error: Unable to make predictions."

    return {'prediction': prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}

# prediction endpoint
@app.get('/predict')
def predict(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):

    # create a datetime object from the user provided datetime

    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    ormatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")


    X_pred = pd.DataFrame(data={
            'key': '2013-07-06 17:18:00.000000119',
            "pickup_datetime": [ormatted_pickup_datetime],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "passenger_count": [int(passenger_count)]
    }, index=[0])

    pipe = joblib.load('model.joblib')

    y_pred = pipe.predict(X_pred)

    return {'fare': y_pred[0]}

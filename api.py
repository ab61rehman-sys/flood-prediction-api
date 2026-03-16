from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("middlesbrough_flood_xgb.pkl")

with open("feature_order.json") as f:
    FEATURES = json.load(f)


class ForecastDay(BaseModel):
    temperature_2m_max: float
    temperature_2m_min: float
    precipitation_sum: float
    rain_sum: float
    snowfall_sum: float
    precipitation_hours: float
    wind_speed_10m_max: float
    wind_direction_10m_dominant: float
    wind_gusts_10m_max: float
    pressure_msl_mean: float
    soil_moisture_0_to_7cm_mean: float
    soil_temperature_0_to_100cm_mean: float
    relative_humidity_2m_mean: float
    dew_point_2m_mean: float
    surface_pressure_mean: float


class RequestData(BaseModel):
    days: List[ForecastDay]


@app.get("/")
def root():
    return {"status": "API running"}


@app.post("/predict")
def predict(data: RequestData):

    df = pd.DataFrame([d.dict() for d in data.days])

    df = df[FEATURES]

    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []

    for i, (p, pr) in enumerate(zip(preds, probs)):
        results.append({
            "day": i+1,
            "prediction": "Flood Alert" if p == 1 else "No Flood",
            "probability": float(pr)
        })

    return {"predictions": results}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=10000)
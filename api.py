# api.py
# Flood Prediction API (Render-compatible)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import json
import os

# ------------------------------------------------
# Create FastAPI app
# ------------------------------------------------

app = FastAPI(title="Middlesbrough Flood Prediction API")

# Allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# Load model + feature order safely
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "middlesbrough_flood_xgb.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_order.json")

model = None
FEATURES = []

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

try:
    with open(FEATURE_PATH) as f:
        FEATURES = json.load(f)
    print("Feature list loaded")
except Exception as e:
    print("Error loading feature list:", e)

# ------------------------------------------------
# Request data model
# ------------------------------------------------

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


class PredictionRequest(BaseModel):
    days: List[ForecastDay]


# ------------------------------------------------
# Health endpoint
# ------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Flood Prediction API running",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES)
    }


# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------

@app.post("/predict")
def predict(req: PredictionRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:

        # Convert request to DataFrame
        rows = []
        for day in req.days:
            rows.append(day.dict())

        df = pd.DataFrame(rows)

        # Ensure correct feature order
        df = df.reindex(columns=FEATURES, fill_value=0)

        # Model predictions
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)

        results = []

        for i, (pred, prob) in enumerate(zip(preds, probs)):

            results.append({
                "day": i + 1,
                "prediction": "Flood Alert" if pred == 1 else "No Flood",
                "probability": round(float(prob), 4)
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------
# Local run (optional)
# ------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=10000)

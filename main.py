# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model
model = load_model("house_price_api")

# ✅ Input model
class InputModel(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# ✅ Output model
class OutputModel(BaseModel):
    prediction: float

# API endpoint
@app.post("/predict", response_model=OutputModel)
def predict(data: InputModel):
    df = pd.DataFrame([data.model_dump()])
    predictions = predict_model(model, data=df)
    return {"prediction": float(predictions["prediction_label"].iloc[0])}

# Run
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
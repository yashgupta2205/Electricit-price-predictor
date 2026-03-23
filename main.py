from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
transformer = joblib.load("yeojohnson_resident_transformer.pkl")

class InputData(BaseModel):
    site_area: float
    water_consumption: float
    recycling_rate: float
    utilisation_rate: float
    air_quality_index: float
    issue_resolution_time: float
    structure_type_encoded: int
    resident_count_raw: float

@app.post("/predict")
def predict(data: InputData):
    transformed_count = transformer.transform([[data.resident_count_raw]])[0][0]
    features = np.array([[
        data.site_area,
        data.water_consumption,
        data.recycling_rate,
        data.utilisation_rate,
        data.air_quality_index,
        data.issue_resolution_time,
        data.structure_type_encoded,
        transformed_count
    ]])
    pred = model.predict(features)[0]
    return {"predicted_cost": float(pred)}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load models
models = {
    "riyadh": joblib.load("Riyadh_KM.joblib"),
    "western": joblib.load("Western_KM.joblib"),
    "southern": joblib.load("Southern_KM.joblib"),
    "eastern": joblib.load("Eastern_KM.joblib"),
}

# Load the scaler (make sure you have a saved scaler file)
scaler = joblib.load("scaler.joblib")  # Ensure this file exists

# Define input schema
class ModelInput(BaseModel):
    Type_encoding: int
    Price: float
    Area_m2: float

# Preprocessing functionimport numpy as np

import numpy as np

def preprocessing(input_features: ModelInput):
    # Force shape (1,3)
    arr = np.array([[input_features.Type_encoding,
                     input_features.Price,
                     input_features.Area_m2]], dtype=float)
    
    # Transform all three (including Type_encoding)
    scaled_arr = scaler.transform(arr)  # shape (1,3)
    
    # Overwrite the scaled Type_encoding column with the raw (unscaled) value
    scaled_arr[0, 0] = input_features.Type_encoding

    return scaled_arr  # shape (1,3)

# Prediction function
def predict(model, data):
    try:
        preprocessed_data = preprocessing(data)
        prediction = model.predict(preprocessed_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define API endpoints for each model
@app.post("/predict/riyadh")
async def predict_riyadh(input_data: ModelInput):
    return predict(models["riyadh"], input_data)

@app.post("/predict/western")
async def predict_western(input_data: ModelInput):
    return predict(models["western"], input_data)

@app.post("/predict/southern")
async def predict_southern(input_data: ModelInput):
    return predict(models["southern"], input_data)

@app.post("/predict/eastern")
async def predict_eastern(input_data: ModelInput):
    return predict(models["eastern"], input_data)

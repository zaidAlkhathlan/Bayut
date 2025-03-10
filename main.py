from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

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

# Preprocessing function
def preprocessing(input_features: ModelInput):
    """Applies the same preprocessing steps as used during model training."""
    
    dict_f = {
        "Type_encoding": input_features.Type_encoding,
        "Price": input_features.Price,
        "Area_m2": input_features.Area_m2,
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features using the trained scaler
    scaled_features = scaler.transform([features_list])

    return scaled_features

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

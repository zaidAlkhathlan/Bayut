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

def preprocessing(input_features: ModelInput):
    """Applies preprocessing while keeping Type_encoding unchanged."""
    
    # Keep categorical feature unchanged
    type_encoding = input_features.Type_encoding  

    # Extract only numerical features for scaling
    numeric_features = [input_features.Price, input_features.Area_m2]
    
    # Apply scaling to numerical features
    scaled_numeric_features = scaler.transform([numeric_features])

    # Combine unscaled and scaled features
    final_features = [type_encoding] + scaled_numeric_features.tolist()[0]

    return final_features


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

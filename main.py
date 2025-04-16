import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Delivery Predictor API",
    description="API for predicting customer delivery times",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor
try:
    model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {str(e)}")
    raise

class DeliveryFeatures(BaseModel):
    """
    Represents the features required for predicting delivery time.
    """
    customer_latitude: float
    customer_longitude: float
    restaurant_latitude: float
    restaurant_longitude: float
    order_time: str
    weather_condition: str
    traffic_condition: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to Customer Delivery Predictor API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_delivery_time(features: DeliveryFeatures):
    try:
        # Prepare features for prediction
        features_dict = features.dict()
        
        # Transform features using preprocessor
        features_transformed = preprocessor.transform([list(features_dict.values())])
        
        # Make prediction
        prediction = model.predict(features_transformed)[0]
        
        logger.info(f"Prediction successful: {prediction} minutes")
        return {"predicted_delivery_time_minutes": float(prediction)}
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 4))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload
    ) 
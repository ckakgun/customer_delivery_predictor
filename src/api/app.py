from typing import Optional
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from ..logger import setup_logger

# Logger setup
setup_logger()
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Delivery Time Prediction API",
    description="API for predicting food delivery times",
    version="1.0.0"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, replace with your domain
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

class DeliveryInput(BaseModel):
    """
    Represents the input data for predicting delivery time.
    """
    delivery_person_age: float
    delivery_person_ratings: float
    vehicle_condition: int
    multiple_deliveries: int
    weatherconditions: str
    road_traffic_density: str
    type_of_order: str
    type_of_vehicle: str
    festival: str
    city: str
    restaurant_latitude: float
    restaurant_longitude: float
    delivery_location_latitude: float
    delivery_location_longitude: float

class PredictionResponse(BaseModel):
    """
    Represents the response from the prediction API.
    """
    predicted_delivery_time: float
    confidence: Optional[float] = None

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """
    Returns a message indicating that the API is running.
    """
    return {"message": "Delivery Time Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict_delivery_time(request: Request, input_data: DeliveryInput):
    """
    Predicts the delivery time for a given input data.
    """
    try:
        logger.info("Received prediction request")
        
        # Convert input to CustomData
        custom_data = CustomData(
            delivery_person_age=input_data.delivery_person_age,
            delivery_person_ratings=input_data.delivery_person_ratings,
            vehicle_condition=input_data.vehicle_condition,
            multiple_deliveries=input_data.multiple_deliveries,
            weatherconditions=input_data.weatherconditions,
            road_traffic_density=input_data.road_traffic_density,
            type_of_order=input_data.type_of_order,
            type_of_vehicle=input_data.type_of_vehicle,
            festival=input_data.festival,
            city=input_data.city,
            restaurant_latitude=input_data.restaurant_latitude,
            restaurant_longitude=input_data.restaurant_longitude,
            delivery_location_latitude=input_data.delivery_location_latitude,
            delivery_location_longitude=input_data.delivery_location_longitude
        )

        # Get features
        features = custom_data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features)
        
        logger.info("Prediction successful: %s minutes", prediction[0])
        
        return PredictionResponse(
            predicted_delivery_time=float(prediction[0]),
            confidence=0.95
        )

    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
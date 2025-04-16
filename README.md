# Customer Delivery Time Predictor

This project predicts food delivery times based on various factors like weather, traffic, and location.

## API Endpoints

### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**: Welcome message

### Predict Delivery Time
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
    "delivery_person_age": 25.0,
    "delivery_person_ratings": 4.5,
    "vehicle_condition": 2,
    "multiple_deliveries": 0,
    "weatherconditions": "Sunny",
    "road_traffic_density": "Medium",
    "type_of_order": "Snack",
    "type_of_vehicle": "motorcycle",
    "festival": "No",
    "city": "Urban",
    "restaurant_latitude": 41.0082,
    "restaurant_longitude": 28.9784,
    "delivery_location_latitude": 41.0082,
    "delivery_location_longitude": 28.9784
}
```
- **Response**:
```json
{
    "predicted_delivery_time": 30.5,
    "confidence": 0.95
}
```

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `uvicorn src.api.app:app --reload`

## Environment Variables

Copy `.env.example` to `.env` and configure:
- `API_HOST`: API host address
- `API_PORT`: API port number
- `LOG_LEVEL`: Logging level
- `MODEL_PATH`: Path to model file
- `PREPROCESSOR_PATH`: Path to preprocessor file

## Deployment

The application is deployed on Render. To deploy:
1. Fork this repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure the environment variables
5. Deploy!
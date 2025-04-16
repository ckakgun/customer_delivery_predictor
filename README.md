# Customer Delivery Time Predictor

A machine learning-based API that predicts food delivery times based on various factors such as weather conditions, traffic density, and location data.

## Project Structure

```
customer_delivery_predictor/
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
├── src/
│   ├── api/
│   │   └── app.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── logger.py
│   └── utils.py
├── .env.example
├── requirements.txt
└── README.md
```

## Features

- **Machine Learning Models**: Uses multiple models (XGBoost, CatBoost) for accurate delivery time predictions
- **RESTful API**: FastAPI-based API with automatic documentation
- **Security Features**: 
  - Rate limiting
  - Security headers
  - CORS protection
  - Input validation
- **Logging**: Comprehensive logging system
- **Environment Configuration**: Easy configuration through environment variables

## Technology Stack

- **Framework**: FastAPI
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, RandomForestRegressor, DecisionTreeRegressor
- **Data Processing**: pandas, numpy
- **Geolocation**: geopy
- **Security**: slowapi (rate limiting)
- **Deployment**: Render

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer_delivery_predictor.git
cd customer_delivery_predictor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## Usage

### Local Development

1. Start the API server:
```bash
uvicorn src.api.app:app --reload
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Rate Limit**: 10 requests/minute
- **Response**: Welcome message

#### Predict Delivery Time
- **URL**: `/predict`
- **Method**: `POST`
- **Rate Limit**: 5 requests/minute
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

## Development Process

1. **Data Preparation**:
   - Data ingestion from multiple sources
   - Feature engineering
   - Data transformation and preprocessing

2. **Model Development**:
   - Multiple model training (XGBoost, CatBoost)
   - Model evaluation and selection
   - Model persistence

3. **API Development**:
   - FastAPI implementation
   - Input/Output models
   - Error handling
   - Logging system

4. **Security Implementation**:
   - Rate limiting
   - Security headers
   - Input validation
   - CORS configuration

5. **Deployment**:
   - Environment configuration
   - Render deployment setup
   - Continuous deployment with GitHub

## Deployment

The application is deployed on Render. To deploy your own instance:

1. Fork this repository
2. Create a new Web Service on Render
3. Configure the following:
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
4. Set environment variables:
   ```
   PYTHONPATH=/opt/render/project/src
   HOST=0.0.0.0
   LOG_LEVEL=INFO
   MODEL_PATH=artifacts/model.pkl
   PREPROCESSOR_PATH=artifacts/preprocessor.pkl
   ```

## Security Considerations

- Rate limiting implemented to prevent abuse
- Input validation for all API endpoints
- Security headers for protection against common web vulnerabilities
- CORS configuration for API access control
- Regular dependency updates for security patches

## Future Improvements

- [ ] Add user authentication
- [ ] Implement model retraining pipeline
- [ ] Add more performance metrics
- [ ] Implement caching for frequent predictions
- [ ] Add monitoring and alerting

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details
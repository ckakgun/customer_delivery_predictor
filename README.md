# End to end ML project

# Customer Delivery Time Prediction

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Training
1. Run the training pipeline:
```bash
python src/pipeline/train_pipeline.py
```

This will:
- Train multiple models
- Select the best performing model
- Save the model and preprocessor to `artifacts/` directory

## Making Predictions
1. Use the predict pipeline:
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create input data
custom_data = CustomData(
    delivery_person_age=25.0,
    delivery_person_ratings=4.5,
    vehicle_condition=2,
    multiple_deliveries=1,
    weatherconditions="Sunny",
    road_traffic_density="Medium",
    type_of_order="Buffet",
    type_of_vehicle="motorcycle",
    festival="No",
    city="Metropolitian",
    restaurant_latitude=12.9716,
    restaurant_longitude=77.5946,
    delivery_location_latitude=12.9716,
    delivery_location_longitude=77.5946
)

# Make prediction
predict_pipeline = PredictPipeline()
features = custom_data.get_data_as_data_frame()
prediction = predict_pipeline.predict(features)
```

## Notes
- The `artifacts/` directory contains trained models and preprocessors
- This directory is not tracked by git
- You need to train the model locally to generate these files
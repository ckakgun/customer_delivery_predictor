import os
import sys
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """Class for prediction pipeline operations."""
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.data_transformation = DataTransformation()

    def predict(self, features):
        """Predict using the trained model."""
        try:
            # Load model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Convert input features to DataFrame
            features_df = pd.DataFrame(features)
            
            # Calculate distance
            features_df = self.data_transformation.calculate_distance(features_df)
            
            # Transform features using preprocessor
            transformed_features = preprocessor.transform(features_df)
            
            # Make prediction
            prediction = model.predict(transformed_features)
            
            return prediction

        except Exception as e:
            raise CustomException(e, sys) from e

class CustomData:
    """Class for custom data operations."""
    def __init__(self,
                 delivery_person_age: float,
                 delivery_person_ratings: float,
                 vehicle_condition: int,
                 multiple_deliveries: int,
                 weatherconditions: str,
                 road_traffic_density: str,
                 type_of_order: str,
                 type_of_vehicle: str,
                 festival: str,
                 city: str,
                 restaurant_latitude: float,
                 restaurant_longitude: float,
                 delivery_location_latitude: float,
                 delivery_location_longitude: float):
        
        self.delivery_person_age = delivery_person_age
        self.delivery_person_ratings = delivery_person_ratings
        self.vehicle_condition = vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.weatherconditions = weatherconditions
        self.road_traffic_density = road_traffic_density
        self.type_of_order = type_of_order
        self.type_of_vehicle = type_of_vehicle
        self.festival = festival
        self.city = city
        self.restaurant_latitude = restaurant_latitude
        self.restaurant_longitude = restaurant_longitude
        self.delivery_location_latitude = delivery_location_latitude
        self.delivery_location_longitude = delivery_location_longitude

    def get_data_as_data_frame(self):
        """Get custom data as a DataFrame."""
        try:
            custom_data_input_dict = {
                "Delivery_person_Age": [self.delivery_person_age],
                "Delivery_person_Ratings": [self.delivery_person_ratings],
                "Vehicle_condition": [self.vehicle_condition],
                "multiple_deliveries": [self.multiple_deliveries],
                "Weatherconditions": [self.weatherconditions],
                "Road_traffic_density": [self.road_traffic_density],
                "Type_of_order": [self.type_of_order],
                "Type_of_vehicle": [self.type_of_vehicle],
                "Festival": [self.festival],
                "City": [self.city],
                "Restaurant_latitude": [self.restaurant_latitude],
                "Restaurant_longitude": [self.restaurant_longitude],
                "Delivery_location_latitude": [self.delivery_location_latitude],
                "Delivery_location_longitude": [self.delivery_location_longitude]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys) from e
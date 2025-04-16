import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

class TrainPipeline:
    """Train the model."""
    def __init__(self):
        self.data_path = None
        self.target_column = None
        self.numerical_features = None
        self.categorical_features = None

    def initiate_training(self, data_path, target_column, numerical_features, categorical_features):
        """Initiate the training pipeline."""
        try:
            self.data_path = data_path
            self.target_column = target_column
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features

            # Read the dataset
            df = pd.read_csv(self.data_path)
            logging.info('Read the dataset as dataframe')

            # Create artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Split the data into training and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets
            train_path = os.path.join('artifacts', 'train.csv')
            test_path = os.path.join('artifacts', 'test.csv')
            
            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)

            logging.info("Initiated Data Transformation")
            data_transformation = DataTransformation()
            
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path,
                test_path,
                self.numerical_features,
                self.categorical_features
            )

            logging.info("Initiated Model Training")
            model_trainer = ModelTrainer()
            best_model_name, model_report = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info(f"Best Model: {best_model_name}")
            logging.info("Model Report:")
            for model_name, metrics in model_report.items():
                logging.info(f"\n{model_name}:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value}")

            return best_model_name, model_report

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.initiate_training(
        data_path="notebook/data/food_delivery.csv",
        target_column="Time_taken(min)",
        numerical_features=[
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Vehicle_condition",
            "multiple_deliveries",
            "distance"
        ],
        categorical_features=[
            "Weatherconditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City"
        ]
    )
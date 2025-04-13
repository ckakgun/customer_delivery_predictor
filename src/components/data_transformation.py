import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from geopy.distance import geodesic

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation paths and parameters."""
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Class for data transformation operations."""
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation based on the data type of the columns.
        """
        try:
            numerical_columns = [
               'Delivery_person_Age',
    'Delivery_person_Ratings',
    'Vehicle_condition',
                'distance',
                'Order_day',
                'Order_month',
                'Order_year',
                'multiple_deliveries'
            ]
            categorical_columns = [
                'Weather_Conditions',
                'Road_traffic_density',
                'Type_of_order',
                'Type_of_vehicle',
                'Festival',
                'City'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), #handling missing values
                    ("scaler", StandardScaler()) #scaling the data
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), #handling missing values
                    ("one_hot_encoder", OneHotEncoder()), #encoding the data
                    ("scaler", StandardScaler(with_mean=False)) #scaling the data
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for data transformation based on the data type of the columns.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Process date columns
            def process_date(df):
                df['Order_Date'] = pd.to_datetime(df['Order_Date'])
                df['Order_day'] = df['Order_Date'].dt.day
                df['Order_month'] = df['Order_Date'].dt.month
                df['Order_year'] = df['Order_Date'].dt.year
                return df

            # Calculate distance
            def calculate_distance(df):
                df['distance'] = df.apply(
                    lambda row: geodesic(
                        (row['Restaurant_latitude'], row['Restaurant_longitude']),
                        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                    ).kilometers,
                    axis=1
                )
                return df

            logging.info("Processing dates and calculating distances")
            train_df = process_date(train_df)
            test_df = process_date(test_df)
            train_df = calculate_distance(train_df)
            test_df = calculate_distance(test_df)

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Time_taken(min)"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )   
            
        except Exception as e:
            raise CustomException(e, sys) from e




import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from geopy.distance import geodesic

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation paths and parameters."""
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """Class for data transformation operations."""
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def calculate_distance(self, df):
        """Calculate distance between restaurant and delivery location."""
        try:
            df['distance'] = df.apply(
                lambda row: geodesic(
                    (row['Restaurant_latitude'], row['Restaurant_longitude']),
                    (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
                ).kilometers,
                axis=1
            )
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_transformer_object(self, numerical_features, categorical_features):
        """Get data transformer object."""
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path, numerical_features, categorical_features):
        """Initiate data transformation."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Preprocess Time_taken(min) column
            train_df['Time_taken(min)'] = train_df['Time_taken(min)'].str.strip()
            test_df['Time_taken(min)'] = test_df['Time_taken(min)'].str.strip()
            
            # Remove any non-numeric characters and convert to float
            train_df['Time_taken(min)'] = train_df['Time_taken(min)'].str.replace(r'[^\d.]', '', regex=True)
            test_df['Time_taken(min)'] = test_df['Time_taken(min)'].str.replace(r'[^\d.]', '', regex=True)
            
            # Convert to float
            train_df['Time_taken(min)'] = pd.to_numeric(train_df['Time_taken(min)'], errors='coerce')
            test_df['Time_taken(min)'] = pd.to_numeric(test_df['Time_taken(min)'], errors='coerce')
            
            # Check for NaN values before filling
            logging.info(f"NaN values in train target: {train_df['Time_taken(min)'].isna().sum()}")
            logging.info(f"NaN values in test target: {test_df['Time_taken(min)'].isna().sum()}")
            
            # Fill any NaN values with median
            median_time = train_df['Time_taken(min)'].median()
            train_df['Time_taken(min)'] = train_df['Time_taken(min)'].fillna(median_time)
            test_df['Time_taken(min)'] = test_df['Time_taken(min)'].fillna(median_time)
            
            # Verify no NaN values remain
            if train_df['Time_taken(min)'].isna().any() or test_df['Time_taken(min)'].isna().any():
                raise CustomException("NaN values still present in target column after preprocessing", sys)

            logging.info("Time_taken(min) preprocessing completed")

            # Calculate distance for both train and test data
            train_df = self.calculate_distance(train_df)
            test_df = self.calculate_distance(test_df)

            logging.info("Distance calculation completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(
                numerical_features,
                categorical_features
            )

            target_column_name = "Time_taken(min)"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Verify target columns have no NaN values
            if target_feature_train_df.isna().any() or target_feature_test_df.isna().any():
                raise CustomException("NaN values found in target columns after preprocessing", sys)

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert target to numpy array and verify no NaN values
            target_feature_train_arr = np.array(target_feature_train_df)
            target_feature_test_arr = np.array(target_feature_test_df)
            
            if np.isnan(target_feature_train_arr).any() or np.isnan(target_feature_test_arr).any():
                raise CustomException("NaN values found in target arrays after conversion", sys)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

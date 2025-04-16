import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    """Configuration for model training paths and parameters."""
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """Class for model training operations."""
    def __init__(self):
        """Initialize model trainer configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, x_train, y_train, x_test, y_test, models):
        """Evaluate multiple models and return their performance metrics."""
        try:
            report = {}
            for model_name, model in models.items():
                # Train model
                model.fit(x_train, y_train)

                # Make predictions
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                # Get R2 scores
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                # Get Cross validation score
                cv_scores = cross_val_score(model, x_train, y_train, cv=5)
                
                report[model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'mae': mean_absolute_error(y_test, y_test_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
                }
            
            return report

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self, train_array, test_array):
        """Train and evaluate multiple models, returning the best model's performance metrics."""
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            model_report = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(model_report.items(), 
                                        key=lambda x: x[1]['test_r2']))[1]['test_r2']
            
            best_model_name = max(sorted(model_report.items(), 
                                       key=lambda x: x[1]['test_r2']))[0]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, model_report

        except Exception as e:
            raise CustomException(e, sys) from e

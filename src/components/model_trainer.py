import os
import sys
from dataclasses import dataclass
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configuration for model training paths and parameters."""
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """Class for model training operations."""
    def __init__(self):
        """Initialize model trainer configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Train and evaluate multiple models, returning the best model's performance metrics."""
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": xgb.XGBRegressor(),
                "LightGBM": LGBMRegressor()
            }

            params = {
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'copy_X': [True, False],
                    'positive': [True, False]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "LightGBM": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }

            # First evaluate models without hyperparameter tuning
            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)
            
            # Get best model name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Perform hyperparameter tuning on the best model
            logging.info(f"Performing hyperparameter tuning for {best_model_name}")
            grid_search = GridSearchCV(
                estimator=models[best_model_name],
                param_grid=params[best_model_name],
                cv=5,
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(x_train, y_train)
            
            # Get the best model with tuned parameters
            best_model = grid_search.best_estimator_
            
            if best_model_score < 0.6:
                raise CustomException("There is no best model", sys)
            logging.info("Best model found on training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys) from e

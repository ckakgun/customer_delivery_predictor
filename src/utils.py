import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV    
from src.exception import CustomException



def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using pickle.
    
    Args:
        file_path (str): Path to save the object
        obj (object): Object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Load an object from a file using pickle.
    
    Args:
        file_path (str): Path to load the object from
        
    Returns:
        object: Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e

def evaluate_models(x_train, y_train,x_test,y_test,models,param):
    """Evaluate multiple models with hyperparameter tuning."""
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
    

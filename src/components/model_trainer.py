import os 
import sys 

from src.exception.exception import DeliveryTimeException
from src.logging.logger import logging


from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.ml_utils.model.estimator import DeliveryPredictionModel
from src.utils.main_utils.utils import save_object, load_object
from src.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.regression_metric import get_regression_score

import pandas as pd 
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import os 
import joblib
import sys 
import joblib

import mlflow

from dotenv import load_dotenv
load_dotenv()

os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config.feature_names =['multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings', 'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance','Delivery_person_Age',  'prep_traffic',  'City']
        except Exception as e:
            raise DeliveryTimeException(e, sys)
    
    def track_mlflow(self, best_model, regressionMetric):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        with mlflow.start_run():
            mlflow.log_metric("f1_score", regressionMetric.r2_score)
            mlflow.log_metric("precision", regressionMetric.mean_absolute_error)
            mlflow.log_metric("recall", regressionMetric.mean_squared_error)

            joblib.dump(best_model, "model.joblib")

            mlflow.log_artifact("model.joblib", artifact_path="model")


    def train_model(self,X_train, y_train, X_test, y_test):
        try:
            models = {
                "XGBoost Regression": xgb.XGBRegressor(random_state=42)
            }

            params = {
                "XGBoost Regression": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_by_tree':[0.6, 0.8, 1.0]
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)

            regression_train_metric=get_regression_score(y_true=y_train, y_pred=y_train_pred)

            self.track_mlflow(best_model, regression_train_metric)
            
            y_test_pred = best_model.predict(X_test)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model, regression_test_metric)

            
    
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            Delivery_Prediction_Model=DeliveryPredictionModel(model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Delivery_Prediction_Model)

            # Model pusher
            save_object("final_model/model.pkl", best_model)

            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_artifact=regression_train_metric,
                                                        test_metric_artifact=regression_test_metric
                                                        )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact





        except Exception as e:
            raise DeliveryTimeException(e, sys)
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info(f"shape of training data: {train_arr.shape}")
            logging.info(f"Testing array: {test_arr.shape}")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape{X_test.shape}")
            feature_names = self.model_trainer_config.feature_names

            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=feature_names)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=feature_names)

            joblib.dump({'scaler': scaler, 'feature_names': feature_names}, 'final_model/preprocessor.pkl')


            model_trainer_artifact = self.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise DeliveryTimeException(e, sys)
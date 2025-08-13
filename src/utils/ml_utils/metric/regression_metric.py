from src.entity.artifact_entity import RegressionMetricArtifact
from src.exception.exception import DeliveryTimeException
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

def get_regression_score(y_true, y_pred) ->RegressionMetricArtifact:
    try:
        model_r2_score=r2_score(y_true, y_pred)
        model_mean_absolute_error= mean_absolute_error(y_true, y_pred)
        model_mean_squared_error=mean_squared_error(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(
            r2_score=model_r2_score,
            mean_absolute_error=model_mean_absolute_error,
            mean_squared_error=model_mean_squared_error
        )
        return regression_metric
    
    except Exception as e:
        raise DeliveryTimeException(e, sys)
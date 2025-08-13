from src.exception.exception import DeliveryTimeException
from src.components.data_ingestion import DataIngestion
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig


from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig

from src.entity.config_entity import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

import sys

if __name__=='__main__':
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
        data_ingestion=DataIngestion(dataIngestionConfig)

        logging.info("Initiate the data ingestion")
        dataIngestionArtifact=data_ingestion.initiate_date_ingestion()
        logging.info("Data Initiation Completed")
        print(dataIngestionArtifact)
        logging.info("Data Ingestion Completed")

        data_validation_config=DataValidationConfig(trainingPipelineConfig)
        data_validation=DataValidation(dataIngestionArtifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config=DataTransformationConfig(trainingPipelineConfig)
        logging.info("Data transformation started")
        data_transformation=DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation Completed")

        logging.info("Model Training started")
        model_trainer_config=ModelTrainerConfig(trainingPipelineConfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")


    except Exception as e:
        raise DeliveryTimeException(e, sys)

import sys
import os 
import numpy as np 
import pandas as pd 
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.constants.training_pipeline import TARGET_COLUMN
from src.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from sklearn.pipeline import Pipeline
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import DeliveryTimeException
from src.logging.logger import logging
from src.utils.main_utils.utils import save_numpy_array_data, save_object
import joblib

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, 
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise DeliveryTimeException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            
            return pd.read_csv(file_path) 
        except Exception as e:
            raise DeliveryTimeException(e, sys)
        
    @staticmethod
    def extract_time_taken(val):
        try:
            if isinstance(val, str) and " " in val:
                return int(val.split(" ")[1])
            elif isinstance(val, (int, float)):
                return int(val)
            else:
                return np.nan  # fallback
        except:
            return np.nan
        
    def preprocess_data(self, df):
        try:
            df.rename(columns={'Weatherconditions': 'Weather_conditions'}, inplace=True)
            if 'Time_taken(min)' in df.columns:
                df['Time_taken(min)'] = df['Time_taken(min)'].apply(self.extract_time_taken)
            df['Weather_conditions'] = df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip() if pd.notnull(x) else x)
            df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]
            df.drop(['ID', 'Delivery_person_ID'], axis=1, inplace=True)
            df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce').astype('float64')
            df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce').astype('float64')
            df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce').astype('float64')
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], format="%d-%m-%Y")
            df.replace('NaN', float(np.nan), regex=True, inplace=True)
            df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age'].dropna()))
            df['Weather_conditions'] = df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions'].dropna()))
            df['City'] = df['City'].fillna(df['City'].mode()[0])
            df['Festival'] = df['Festival'].fillna(df['Festival'].mode()[0])
            df['multiple_deliveries'] = df['multiple_deliveries'].fillna(df['multiple_deliveries'].mode()[0])
            df['Road_traffic_density'] = df['Road_traffic_density'].fillna(df['Road_traffic_density'].mode()[0])
            df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())

            logging.info("Data preprocessing completed for DataFrame")
            return df
        except Exception as e:
            raise DeliveryTimeException(e, sys)
        
    def label_encoding(self, df):
        try:
            categorical_columns = df.select_dtypes(include='object').columns
            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                joblib.dump(le, 'final_model/label_encoder.pkl')
            logging.info("Label encoding completed for DataFrame")
            return df
        except Exception as e:
            raise DeliveryTimeException(e, sys)

        
        
    def feature_engineering(self, df):
        try:
            logging.info(f"Starting feature engineering. Initial DataFrame shape: {df.shape}")

            # Drop rows with missing coordinates that will break geodesic distance calculation
            df.dropna(subset=[
                'Restaurant_latitude', 'Restaurant_longitude',
                'Delivery_location_latitude', 'Delivery_location_longitude'
            ], inplace=True)

            # --- Temporal features from Order_Date ---
            df["day"] = df.Order_Date.dt.day
            df["month"] = df.Order_Date.dt.month
            df["quarter"] = df.Order_Date.dt.quarter
            df["year"] = df.Order_Date.dt.year
            df["day_of_week"] = df.Order_Date.dt.day_of_week.astype(int)
            df["is_month_start"] = df.Order_Date.dt.is_month_start.astype(int)
            df["is_month_end"] = df.Order_Date.dt.is_month_end.astype(int)
            df["is_quarter_start"] = df.Order_Date.dt.is_quarter_start.astype(int)
            df["is_quarter_end"] = df.Order_Date.dt.is_quarter_end.astype(int)
            df["is_year_start"] = df.Order_Date.dt.is_year_start.astype(int)
            df["is_year_end"] = df.Order_Date.dt.is_year_end.astype(int)
            df["is_weekend"] = np.where(df["day_of_week"].isin([5, 6]), 1, 0)

            # --- Order preparation time ---
            df["Time_Orderd"] = pd.to_timedelta(df["Time_Orderd"].fillna("00:00:00"))
            df["Time_Order_picked"] = pd.to_timedelta(df["Time_Order_picked"].fillna("00:00:00"))
            df["Time_Ordered_formatted"] = df["Order_Date"] + df["Time_Orderd"]
            df["Time_Order_picked_base"] = df["Order_Date"] + df["Time_Order_picked"]
            mask = df["Time_Order_picked"] < df["Time_Orderd"]
            df["Time_Order_picked_formatted"] = df["Time_Order_picked_base"].copy()
            df.loc[mask, "Time_Order_picked_formatted"] += pd.Timedelta(days=1)
            df["order_prepare_time"] = (
                df["Time_Order_picked_formatted"] - df["Time_Ordered_formatted"]
            ).dt.total_seconds() / 60
            df["order_prepare_time"] = df["order_prepare_time"].fillna(df["order_prepare_time"].median())

            # Drop intermediate time columns
            df.drop([
                "Time_Orderd", "Time_Order_picked", "Time_Ordered_formatted",
                "Time_Order_picked_base", "Time_Order_picked_formatted", "Order_Date"
            ], axis=1, inplace=True)

            # --- Label encoding for categorical features (done before numeric operations) ---
            df = self.label_encoding(df)

            # --- Geodesic distance calculation ---
            restaurant_coords = df[["Restaurant_latitude", "Restaurant_longitude"]].to_numpy()
            delivery_coords = df[["Delivery_location_latitude", "Delivery_location_longitude"]].to_numpy()
            df["distance"] = np.array([
                geodesic(restaurant, delivery).kilometers
                for restaurant, delivery in zip(restaurant_coords, delivery_coords)
            ])

            # --- Derived composite features ---
            df["distance_traffic"] = df["distance"] * df["Road_traffic_density"]
            df["distance_deliveries"] = df["distance"] * df["multiple_deliveries"]
            df["prep_traffic"] = df["order_prepare_time"] * df["Road_traffic_density"]
            df["age_ratings"] = df["Delivery_person_Age"] * df["Delivery_person_Ratings"]
            df["prep_distance"] = df["order_prepare_time"] * df["distance"]

            # --- Outlier capping ---
            for col in ["distance", "order_prepare_time", "Delivery_person_Age", "multiple_deliveries"]:
                upper_limit = df[col].quantile(0.99)
                df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

            # --- Select final features ---
            selected_features = [
                "multiple_deliveries", "Road_traffic_density", "Vehicle_condition",
                "Delivery_person_Ratings", "distance_deliveries", "Weather_conditions",
                "Festival", "distance_traffic", "distance", "Delivery_person_Age",
                "prep_traffic", "City", "Time_taken(min)"
            ]

            clean_df = df[selected_features]

            # --- Remove rows with any NaN or inf values ---
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
            clean_df = clean_df.dropna()

            logging.info("Feature engineering completed successfully.")
            logging.info(f"Final cleaned DataFrame shape: {clean_df.shape}")

            return clean_df

        except Exception as e:
            raise DeliveryTimeException(e, sys)

        

    

    

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initialize_data_transformation method of Data transformation class")
        try:
            logging.info("Starting data transformation")

            # Reading train and test data 
            logging.info("Reading train and test data")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            preprocessed_train_df = self.preprocess_data(train_df)
            preprocessed_test_df = self.preprocess_data(test_df)

            feature_engineered_train_df = self.feature_engineering(preprocessed_train_df)
            feature_engineered_test_df = self.feature_engineering(preprocessed_test_df)

            # # Training dataframe
            # input_feature_train_df=feature_engineered_train_df.drop(columns=[TARGET_COLUMN], axis=1)
            # target_feature_train_df=feature_engineered_train_df[TARGET_COLUMN] 
            # target_feature_train_df= target_feature_train_df.replace(-1, 0)


            # ## Testing dataframe
            # input_feature_test_df=feature_engineered_test_df.drop(columns=[TARGET_COLUMN], axis=1)
            # target_feature_test_df = feature_engineered_test_df[TARGET_COLUMN]
            # target_feature_test_df = target_feature_test_df.replace(-1, 0)

            logging.info(f"Shape after train feature engineering {feature_engineered_train_df.shape}")
            logging.info(f"Shape after test feature engineering {feature_engineered_test_df.shape}")

            train_arr=np.c_[feature_engineered_train_df]
            test_arr = np.c_[feature_engineered_test_df]

            ## Save the numpy array data 
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            ## Preparing artifacts 
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise DeliveryTimeException(e, sys)
        
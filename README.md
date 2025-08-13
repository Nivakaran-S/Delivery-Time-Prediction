# 🚚 Delivery Time Prediction Pipeline

A complete **end-to-end Machine Learning pipeline** for predicting delivery times, integrated with a **Flask web application** for real-time predictions.  
The project also uses **MLflow** for monitoring model training and version management.

---

## 📌 Overview

This project is designed to **predict the estimated delivery time** for orders based on multiple features such as traffic density, weather conditions, delivery person ratings, and more.  

It consists of:

- **Modular ML Pipeline** for:
  - Data ingestion
  - Data validation
  - Data transformation
  - Model training  
- **Flask Web App** for real-time prediction via a user-friendly interface.
- **MLflow Tracking** to monitor experiments, training runs, and models.

---

## ⚙️ Project Structure

```bash 

├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_validation.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── entity/
│ │ └── config_entity.py
│ ├── exception/
│ │ └── exception.py
│ ├── logging/
│ │ └── logger.py
│ └── ...
├── templates/
│ └── index.html
├── static/
├── final_model/
│ ├── model.pkl
│ └── preprocessor.pkl
├── main.py
├── app.py
└── requirements.txt

```


---

## 📂 Pipeline Workflow

### **1️⃣ Data Ingestion**
- Reads and loads the dataset as per the `DataIngestionConfig`.
- Outputs a `dataIngestionArtifact` containing ingested data paths.

### **2️⃣ Data Validation**
- Validates dataset schema and ensures all required columns are present.
- Outputs a `dataValidationArtifact`.

### **3️⃣ Data Transformation**
- Applies preprocessing such as encoding categorical variables and scaling numerical features.
- Saves the `scaler` and feature names for inference.
- Outputs a `dataTransformationArtifact`.

### **4️⃣ Model Training**
- Trains an ML model on the transformed data.
- Saves the model in `final_model/model.pkl`.
- Tracks training runs in **MLflow**.

---

## 🌐 Flask Application

The `app.py` file runs a **Flask server** that:
- Renders a web form (`index.html`) for user input.
- Validates inputs with mapping dictionaries for categorical fields:
  - Weather conditions
  - Traffic density
  - Vehicle condition
  - Festival
  - City
- Uses the saved **scaler** and **model** for prediction.
- Returns the predicted delivery time in JSON format.

---

## 📊 Input Features

| Feature Name             | Type     | Description |
|--------------------------|----------|-------------|
| multiple_deliveries      | Numeric  | Number of deliveries made before the current order |
| Road_traffic_density     | Categorical | Traffic level (Low, Medium, High, Jam) |
| Vehicle_condition        | Categorical | Condition of delivery vehicle (Poor, Good, Excellent) |
| Delivery_person_Ratings  | Numeric  | Rating of delivery person |
| distance_deliveries      | Numeric  | Distance between consecutive deliveries |
| Weather_conditions       | Categorical | Current weather (Sunny, Cloudy, Fog, etc.) |
| Festival                 | Categorical | Festival time (Yes/No) |
| distance_traffic         | Numeric  | Distance considering traffic |
| distance                 | Numeric  | Actual distance of delivery |
| Delivery_person_Age      | Numeric  | Age of delivery person |
| prep_traffic             | Numeric  | Preparation time considering traffic |
| City                     | Categorical | Type of city (Urban, Semi-Urban, Metropolitan) |

---

## 🚀 How to Run

### **1️⃣ Install Dependencies**

```bash

pip install -r requirements.txt

```

### **2️⃣ Run the ML Pipeline**

```bash

python main.py

```

This will:
- Ingest data
- Validate and transform it
- Train the model
- Save the trained model and preprocessor in final_model/

## **3️⃣ Start the Flask App**

```bash

python app.py

```

The app will run on:

```bash

http://0.0.0.0:5000

```

## 📈 MLflow Tracking

- All training runs are tracked in MLflow.
- This includes:
    - Model parameters
    - Metrics
    - Artifacts (model files)
- Run the MLflow UI:
```bash

mlflow ui

```

## 🛠 Tech Stack
- Python (Data processing & ML)
- Flask (Web framework)
- pandas, numpy (Data handling)
- joblib (Model persistence)
- MLflow (Experiment tracking)
- scikit-learn (ML model training & preprocessing)


## ✨ Key Highlights
- Fully modular ML pipeline design
- Separation of concerns between training (main.py) and prediction (app.py)
- Robust validation for user inputs
- Scalable – can be extended with new features or models
- Tracked & reproducible experiments using MLflow
# 📉 ChurnIQ — Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn.

This project covers the complete Machine Learning lifecycle, including **data cleaning, data processing, exploratory data analysis (EDA), data visualization, feature selection, class imbalance handling using SMOTE, multiple model comparison, Random Forest model selection, Top 10 feature selection, model saving, and Flask deployment**.

# 🚀 Project Overview

Customer churn is a major challenge for telecommunication companies. Losing customers can directly affect revenue and business growth.

The goal of this project is to build a Machine Learning model that can predict whether a customer is likely to **churn** or **stay** based on their account and service information.

The project starts with a raw telecom customer dataset and goes through data cleaning, analysis, visualization, preprocessing, model training, feature selection, and finally deployment as a web application.

The final application uses a **Random Forest Classifier** with the **Top 10 important features** and provides real-time predictions through a Flask web interface.

---

# 🎯 Objectives

The main objectives of this project are:

- Perform data cleaning and preprocessing.
- Understand the dataset using Exploratory Data Analysis.
- Analyze customer churn patterns.
- Visualize important relationships in the dataset.
- Use Matplotlib and Seaborn for data visualization.
- Create correlation heatmaps.
- Analyze distributions and outliers using boxplots.
- Process categorical and numerical features.
- Handle class imbalance using SMOTE.
- Train multiple Machine Learning classification models.
- Compare different models using evaluation metrics.
- Select the Random Forest model.
- Analyze feature importance.
- Select the Top 10 important features.
- Train the final Machine Learning model.
- Save the trained model using Joblib.
- Deploy the model using Flask.
- Generate real-time churn predictions.
- Log predictions into a CSV file.

---

# 📊 Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains information about telecom customers and their services.

### Dataset includes information such as:

- Customer gender
- Senior citizen status
- Partner
- Dependents
- Tenure
- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming services
- Contract
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn

## Target Variable

The target variable is:

```text
Churn

# project workflow
                 RAW DATASET
                      │
                      ▼
               DATA CLEANING
                      │
                      ▼
              DATA PROCESSING
                      │
                      ▼
             EXPLORATORY DATA
                  ANALYSIS
                      │
                      ▼
              DATA VISUALIZATION
                      │
                      ▼
             FEATURE ENGINEERING
                      │
                      ▼
              TRAIN / TEST SPLIT
                      │
                      ▼
                  SMOTE
                      │
                      ▼
            MULTIPLE ML MODELS
                      │
                      ▼
             MODEL COMPARISON
                      │
                      ▼
             RANDOM FOREST
                      │
                      ▼
           FEATURE IMPORTANCE
                      │
                      ▼
             TOP 10 FEATURES
                      │
                      ▼
             FINAL MODEL
                      │
                      ▼
             MODEL SERIALIZATION
                      │
                      ▼
             FLASK APPLICATION
                      │
                      ▼
          REAL-TIME PREDICTION
                      │
                      ▼
             PREDICTION LOGGING


# prediction process

User enters customer information
            ↓
       HTML Form
            ↓
       Flask Backend
            ↓
   Convert Input Values
            ↓
      Feature Ordering
            ↓
      StandardScaler
            ↓
    Random Forest Model
            ↓
      Churn Prediction
            ↓
    Display Result


# project structure

Customer_churn_project/
│
├── churn_env/
│
├── data/
│   ├── customer_churn_cleaned.csv
│   ├── Final_data.csv
│   └── Telco-Customer-Churn.csv
│
├── data_clean_t.../
│   └── customer_churn...
│
├── EDA/
│   ├── customer_churn_rf_model.pkl
│   └── scaler.pkl
│
├── static/
│
├── templates/
│   ├── churn.html
│   └── index.html
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt



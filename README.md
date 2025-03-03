---

# Rossmann Pharmaceuticals Sales Forecasting

This project is an end-to-end solution to forecast daily sales for Rossmann Pharmaceuticals stores up to six weeks ahead. Using datasets from Kaggle (train.csv, store.csv, and test.csv), the solution covers data exploration, cleaning, feature engineering, model building (both classical machine learning and deep learning approaches), and serving predictions via a REST API.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Data Description](#data-description)
- [Features and Tasks](#features-and-tasks)
- [Usage](#usage)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Deep Learning Model (Optional)](#deep-learning-model-optional)
  - [REST API for Predictions](#rest-api-for-predictions)
- [Handling Missing Values](#handling-missing-values)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Rossmann Pharmaceuticals needs to forecast sales for its stores across multiple cities six weeks ahead. The finance team uses these predictions to plan for upcoming investments and store operations. Managers have historically relied on their own judgment for forecasts; this project uses data-driven methods to improve accuracy.

The solution:
- Reads and merges multiple datasets.
- Performs extensive data cleaning, including handling missing values.
- Engineers new features (date components, promotion indicators, etc.).
- Explores the data via visualizations and statistical summaries.
- Builds machine learning models using an sklearn pipeline (e.g., Random Forest).
- Optionally builds a deep learning LSTM model for time series forecasting.
- Evaluates the models with metrics such as MAE, RMSE, and MAPE.
- Serializes the trained model for production.
- Serves predictions via a REST API built with Flask.

## Repository Structure

```
rossmann_sales_forecast/
├── data/                   # CSV files: train.csv, store.csv, test.csv
├── notebooks/              # Jupyter Notebooks (e.g., EDA.ipynb)
├── models/                 # Serialized model files (e.g., random_forest_<timestamp>.pkl, lstm_model.h5)
├── scripts/                # Python scripts:
│   ├── feature_engineering.py  # Feature extraction and transformation functions
│   ├── train_ml_model.py       # ML model pipeline training and evaluation
│   └── train_dl_model.py       # LSTM model training (optional)
├── api.py                  # Flask API to serve predictions
├── requirements.txt        # Project dependencies
├── .gitignore              # Files and folders to ignore by Git
└── README.md               # This file
```

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/rossmann_sales_forecast.git
   cd rossmann_sales_forecast
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Data Description

- **train.csv:** Contains 1,017,209 records with fields such as Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, and SchoolHoliday.
- **store.csv:** Contains store-level information including StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth/Year, Promo2, Promo2SinceWeek/Year, and PromoInterval.
- **test.csv:** Contains data for which sales predictions are required.

## Features and Tasks

The project performs the following tasks:
- **Data Exploration:** Loading the datasets, checking for missing values, and visualizing distributions and relationships.
- **Data Cleaning:** Handling missing values in store data (e.g., imputing CompetitionDistance with 100000 for missing values).
- **Feature Engineering:** Extracting date features (Year, Month, Day, WeekOfYear, Weekday, IsWeekend) and merging store attributes.
- **Modeling:**
  - Building an sklearn pipeline with scaling, encoding, and Random Forest regression.
  - Evaluating model performance using MAE, RMSE, and MAPE.
  - Optionally, developing an LSTM model for time series forecasting.
- **Model Serialization:** Saving the trained model with a timestamp.
- **API Development:** Creating a Flask REST API to serve predictions.

## Usage

### Exploratory Data Analysis

Launch the Jupyter Notebook found in `notebooks/EDA.ipynb` to review data distributions, summary statistics, and visualizations.

### Machine Learning Pipeline

Run the ML training script to train and evaluate your model:

```bash
python scripts/train_ml_model.py
```

This script will:
- Load the cleaned and feature-engineered data.
- Split data into training and validation sets.
- Build a pipeline (with preprocessing and Random Forest regressor).
- Evaluate the model and print evaluation metrics.
- Serialize the model to the `models/` directory.

### Deep Learning Model (Optional)

If you wish to train an LSTM model for time series forecasting, run:

```bash
python scripts/train_dl_model.py
```

This script processes data for a single store time series and trains an LSTM model.

### REST API for Predictions

To start the REST API locally, run:

```bash
python api.py
```

The API will start on port 5000. You can then send a POST request with JSON data to `http://localhost:5000/predict`. For example:

```json
{
  "Store": 1,
  "DayOfWeek": 5,
  "Promo": 1,
  "SchoolHoliday": 0,
  "Year": 2015,
  "Month": 7,
  "Day": 31,
  "WeekOfYear": 31,
  "Weekday": 4,
  "IsWeekend": 0,
  "StoreType": "a",
  "Assortment": "a"
}
```

The API will return the predicted sales for the input features.

## Handling Missing Values

In the store dataset, missing values are imputed as follows:
- **CompetitionDistance:** Imputed with 100000.
- **CompetitionOpenSinceMonth & CompetitionOpenSinceYear:** Imputed with 0.
- **Promo2SinceWeek & Promo2SinceYear:** Imputed with 0.
- **PromoInterval:** Imputed with "None".

Example code snippet:

```python
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(100000)
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0)
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0)
store['PromoInterval'] = store['PromoInterval'].fillna('None')
```

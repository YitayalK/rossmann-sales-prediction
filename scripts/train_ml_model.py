import pandas as pd
import numpy as np
import joblib
import datetime
import os
# For machine learning pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load dataset
train = pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction/data/train.csv", low_memory=False)
test =  pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction//data/test.csv", low_memory=False)
store = pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction//data/store.csv", low_memory=False)

# Convert Date to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date']  = pd.to_datetime(test['Date'])

# Impute missing values in the store dataframe
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(100000)
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0)
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0)
store['PromoInterval'] = store['PromoInterval'].fillna('None')

# Extract date features from the Date column
def add_date_features(df):
    df['Year']    = df['Date'].dt.year
    df['Month']   = df['Date'].dt.month
    df['Day']     = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Weekday'] = df['Date'].dt.weekday  # 0=Monday, 6=Sunday
    df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    return df

train = add_date_features(train)
test  = add_date_features(test)

# Merge store data into train and test based on 'Store'
train = train.merge(store, on='Store', how='left')
test  = test.merge(store, on='Store', how='left')

# Define target and feature columns.
# we use numerical features and categorical features from both train and store info.
# Adjust the list based on your feature engineering.
target = 'Sales'
features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'IsWeekend']

# If store.csv contains additional features (like StoreType, Assortment, etc.), include them:
if 'StoreType' in train.columns:
    features.append('StoreType')
if 'Assortment' in train.columns:
    features.append('Assortment')

# Prepare X and y
X = train[features]
y = train[target]

# Split data into train and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# We build a pipeline that treats numerical and categorical features separately.
# Identify categorical and numerical features
categorical_features = [col for col in features if X_train[col].dtype == 'object']
numerical_features = [col for col in features if X_train[col].dtype in ['int64', 'float64']]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Build the final pipeline with a RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

#Evaluate the Model
# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Compute evaluation metrics
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
epsilon = 1e-10
mape = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100

#mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f"Validation MAE: {mae:.2f}")
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAPE: {mape:.2f}%")

#Serialize the Model
#Save the trained model for later use in production
# Save the pipeline with a timestamp in the filename
timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
model_filename = f"models/random_forest_{timestamp}.pkl"
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, model_filename)
#joblib.dump(pipeline, model_filename)
print(f"Model saved as {model_filename}")
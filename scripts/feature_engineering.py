import pandas as pd
import numpy as np

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

# Display columns after merging
print("Columns in train data after merging store info:")
print(train.columns)




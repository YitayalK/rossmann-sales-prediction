import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



# load dataset
train = pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction/data/train.csv", low_memory=False)
test =  pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction//data/test.csv", low_memory=False)
store = pd.read_csv("D:/File Pack/Courses/10Acadamey/Week 4/Technical Content/rossmann-sales-prediction//data/store.csv", low_memory=False)

# Convert Date to datetime
train['Date'] = pd.to_datetime(train['Date'])
test['Date']  = pd.to_datetime(test['Date'])


# For a single store time series, select data (e.g., store 1)
store1 = train[train['Store'] == 1].sort_values('Date')
sales_series = store1['Sales'].values.reshape(-1, 1)

# Scale the series to (-1, 1)
scaler_ts = MinMaxScaler(feature_range=(-1, 1))
sales_scaled = scaler_ts.fit_transform(sales_series)

# Function to transform time series into supervised data (sliding window)
def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = []
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    return agg.values

# Use past 7 days to predict next day
n_in = 7
n_out = 1
supervised_data = series_to_supervised(sales_scaled, n_in, n_out)
X_ts = supervised_data[:, :n_in]
y_ts = supervised_data[:, -1]

# Reshape input to be 3D [samples, timesteps, features]
X_ts = X_ts.reshape((X_ts.shape[0], n_in, 1))

# Build a simple LSTM model
model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(n_in, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

# Train the LSTM model
history = model_lstm.fit(X_ts, y_ts, epochs=20, batch_size=32, validation_split=0.2, verbose=2)

# After training, you can save the model:
model_lstm.save("models/lstm_model.h5")

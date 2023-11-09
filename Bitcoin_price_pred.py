#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

import math
from tensorflow.keras.layers import LSTM
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_gamma_deviance
from itertools import cycle
import plotly.express as px


# In[2]:


#Load dataset
data = pd.read_csv("BTC-USD.csv")
data


# In[3]:


#info about data
print(data.info())


# In[4]:


#understand the distribution of the data
print(data.describe())


# In[5]:


#Checking if there is a null value
null_values = data.isnull().sum()
print(null_values)


# In[6]:


#format the date column
data['Date'] = pd.to_datetime(data.Date,format='%Y-%m-%d')
data.head()


# In[7]:


#understand correlation between the columns
print(data.corr())


# In[8]:


#explore the relationship using heatmap
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(7,5))
sns.heatmap(data.corr(), annot = True, cmap = "RdYlGn")
plt.title("Heatmap")
plt.show()


# In[9]:


#select the colums needed for the project(feature selection)
closedf = data[['Date','Close']]
closedf0 = data[['Date','Close']]
closedf1 = data[['Date','Close','Volume']]


# In[10]:


#Visualise the Date and close price data in line plot before filtering
fig = plt.figure(figsize = (7, 5))
sns.relplot(data = closedf, x = "Date", y = "Close", kind = "line")
plt.title("Lineplot of Date vs Close price")
plt.xticks(rotation = 45)
plt.show()


# In[11]:


# print the shape of closedf
shape = closedf.shape
print("Shape of dataframe:", shape)


# In[12]:


# Filter the DataFrame to select the data sample we will use
closedf = closedf.loc[closedf['Date'] > '2021-02-19'] #after this date the data...

# Create a copy of the filtered DataFrame
close_stock = pd.DataFrame(closedf)

# Print the total data for prediction
print(f"Total data we will use for predictions: {close_stock.shape[0]}")


# In[13]:


#Visualise the Date and close price data in line plot after filtering
fig = plt.figure(figsize = (7,5))
sns.relplot(data = close_stock, x = "Date", y = "Close", kind = "line")
plt.title("Lineplot of Date vs Close price")
plt.xticks(rotation = 45)
plt.show()


# In[14]:


closedf


# In[15]:


#Normalise the data
# Remove the 'Date' column from closedf DataFrame so that we will have only close price for normalizatiion
closedf = closedf.drop('Date', axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(closedf.values.reshape(-1, 1))

# Print the shape of the transformed data
print(f"Transformed data shape: {closedf.shape}")


# In[16]:


# Calculate the sizes of the training and testing sets
training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size

# Split the data into training and testing sets
train_data = closedf[:training_size]
test_data = closedf[training_size:]

# Print the shapes of the training and testing sets
print(f"Train data: {train_data.shape}")
print(f"Test data: {test_data.shape}")


# In[17]:


def create_dataset(dataset, time_step=1):
    dataX = [dataset[i:(i+time_step), 0] for i in range(len(dataset)-time_step-1)]
    dataY = [dataset[i + time_step, 0] for i in range(len(dataset)-time_step-1)]
    return np.array(dataX), np.array(dataY)


# In[18]:


ts = 15

# Create training dataset
X_train, y_train = create_dataset(train_data, ts)

# Create testing dataset
X_test, y_test = create_dataset(test_data, ts)

# Print the shapes of the datasets
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


# In[19]:


import copy

# Create copies of the datasets
X_train_copy = copy.deepcopy(X_train)
y_train_copy = copy.deepcopy(y_train)
X_test_copy = copy.deepcopy(X_test)
y_test_copy = copy.deepcopy(y_test)

# Print the shapes of the copied datasets
print(f"X_train_copy: {X_train_copy.shape}")
print(f"y_train_copy: {y_train_copy.shape}")
print(f"X_test_copy: {X_test_copy.shape}")
print(f"y_test_copy: {y_test_copy.shape}")


# In[20]:


# Reshape the training and testing datasets for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Print the shapes of the reshaped datasets
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")


# In[21]:


# Create the Sequential model
model = Sequential([
    LSTM(10, input_shape=(None, 1), activation="relu"),
    Dense(1)
])

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")


# In[22]:


# Train the model and store the history
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=0  # Set verbose to 0 to remove the remaining values during training
)

# Get the final loss value
final_loss = history.history['loss'][-1] #-1 shows the last record's loss
print("Final loss:", final_loss)


# In[23]:


# Perform predictions on the training and testing datasets
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Print the shapes of the predicted outputs
print(f"train_predict shape: {train_predict.shape}")
print(f"test_predict shape: {test_predict.shape}")


# In[24]:


import matplotlib.pyplot as plt
# Get the training and validation loss from the history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a figure and plot the training and validation loss
plt.figure()
plt.plot(loss, 'r', label='Training loss')
plt.plot(val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best')

# Show the plot
plt.show()


# In[25]:


# Transform the predicted values back to the original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Reshape the original y_train and y_test arrays
o_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
o_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[26]:


# Calculate evaluation metrics for the train data
train_rmse = math.sqrt(mean_squared_error(o_ytrain, train_predict))
train_mse = mean_squared_error(o_ytrain, train_predict)
train_mae = mean_absolute_error(o_ytrain, train_predict)

# Calculate evaluation metrics for the test data
test_rmse = math.sqrt(mean_squared_error(o_ytest, test_predict))
test_mse = mean_squared_error(o_ytest, test_predict)
test_mae = mean_absolute_error(o_ytest, test_predict)

# Print the evaluation metrics
print("Train data RMSE: ", train_rmse)
print("Train data MSE: ", train_mse)
print("Train data MAE: ", train_mae)
print("-----------------------------------------------------")
print("Test data RMSE: ", test_rmse)
print("Test data MSE: ", test_mse)
print("Test data MAE: ", test_mae)


# In[27]:


# Calculate the explained variance regression score for the train and test data
#The explained variance score provides a useful summary of how well a regression model fits the data
train_explained_variance = explained_variance_score(o_ytrain, train_predict)
test_explained_variance = explained_variance_score(o_ytest, test_predict)

# Print the explained variance regression score
print("Train data explained variance regression score:", train_explained_variance)
print("Test data explained variance regression score:", test_explained_variance)


# In[28]:


#Visualise the explained variance score

# Create labels and values for train and test data
labels = ['Train Data', 'Test Data']
values = [train_explained_variance, test_explained_variance]

# Create bar positions
x = range(len(labels))

# Create the bar chart
plt.bar(x, values, tick_label=labels, color=['blue', 'green'])

# Add labels and title
plt.xlabel('Dataset')
plt.ylabel('Explained Variance Score')
plt.title('Explained Variance Score for Train and Test Data')

# Display the plot
plt.show()


# In[29]:


# Calculate the R2 score for the train and test data
#R2 score is a statistical metric used to evaluate the goodness of fit of a regression model.
train_r2_score = r2_score(o_ytrain, train_predict)
test_r2_score = r2_score(o_ytest, test_predict)

# Print the R2 score
print("Train data R2 score:", train_r2_score)
print("Test data R2 score:", test_r2_score)


# In[30]:


#Visualise the r2score
# Create labels and values for train and test data
labels = ['Train Data', 'Test Data']
values = [train_r2_score, test_r2_score]

# Create bar positions
x = range(len(labels))

# Create the bar chart
plt.bar(x, values, tick_label=labels, color=['blue', 'green'])

# Add labels and title
plt.xlabel('Dataset')
plt.ylabel('R2 Score')
plt.title('R2 Score for Train and Test Data')

# Display the plot
plt.show()


# In[31]:


import plotly.express as px

# Create arrays to hold the predicted values for plotting
trainPredictPlot = np.full_like(closedf, np.nan)
trainPredictPlot[ts:len(train_predict)+ts, :] = train_predict
testPredictPlot = np.full_like(closedf, np.nan)
testPredictPlot[len(train_predict)+(ts*2)+1:len(closedf)-1, :] = test_predict

# Create a DataFrame for plotting
plotdf = pd.DataFrame({
    'date': close_stock['Date'],
    'original_close': close_stock['Close'],
    'train_predicted_close': trainPredictPlot.reshape(1, -1)[0],
    'test_predicted_close': testPredictPlot.reshape(1, -1)[0]
})

# Plot the data using Plotly
fig = px.line(plotdf, x='date', y=['original_close', 'train_predicted_close', 'test_predicted_close'],
              labels={'value': 'Price', 'date': 'Date'},
              title='Comparison between original close price vs predicted close price')
fig.update_layout(plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.show()


# In[34]:


#Save the model
import joblib
model_file = open("LSTM_Bitcoin_price_pred.pkl","wb")
joblib.dump(model, model_file)
model_file.close()


# In[53]:


#let's predict today's price. Note: Todays price is 34500 lowest and 36000 highest
exp1 = pd.to_datetime('2023-11-14')
timestamp_feature = (exp1 - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# predict
prediction = model.predict([[timestamp_feature]])

print(prediction)


# In[54]:


# convert the scaled output
scaled_prediction = 0.11715446
scaled_prediction = np.array(scaled_prediction).reshape(-1, 1)

# Inverse transform to get the original scale
original_prediction = scaler.inverse_transform(scaled_prediction)

# The real value
print(original_prediction)


# The result is very close to today's bitcoin price. 

# # .
# # .
# # .
# # .
# # .
# # .
# # .
# # .
# # .

# **Below are other Models used during the model selection process. (ARIMA and VAL)**

# <Shift> ARIMA

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample data (replace with your actual data)
closedf = np.random.randn(100)

# Create ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(closedf, ax=axes[0], lags=20)
plot_pacf(closedf, ax=axes[1], lags=20)
axes[0].set_title("ACF Plot")
axes[1].set_title("PACF Plot")
plt.tight_layout()
plt.show()

# Define ARIMA hyperparameters
p = 1  # Order of Autoregressive (AR) component
d = 1  # Degree of differencing (I)
q = 1  # Order of Moving Average (MA) component

# Create the ARIMA model and fit it to the data
model = ARIMA(closedf, order=(p, d, q))
model_fit = model.fit()

# Specify the number of time steps into the future you want to predict
forecast_steps = 10

# Make predictions
forecast = model_fit.forecast(steps=forecast_steps)

# Sample data for evaluation (replace with your actual data)
actual_values = np.random.randn(10)
predicted_values = forecast

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
print(f"RMSE: {rmse}")

# Calculate MSE
mse = mean_squared_error(actual_values, predicted_values)
print(f"MSE: {mse}")

# Calculate MAE
mae = mean_absolute_error(actual_values, predicted_values)
print(f"MAE: {mae}")

# Calculate R2
r2 = r2_score(actual_values, predicted_values)
print(f"R2: {r2}")


# VAR MODEL

# In[33]:


import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

# Convert data to a DataFrame
df = pd.DataFrame(closedf1)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Check for stationarity and difference if necessary
def test_stationarity(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] > 0.05:
        print("Time Series is Non-Stationary")
    else:
        print("Time Series is Stationary")

# Example usage of test_stationarity
test_stationarity(df['Close'])

# Difference the data if necessary
# df_diff = df.diff().dropna()

# Split data into train and test sets
train_size = int(0.8 * len(df))
train, test = df[:train_size], df[train_size:]

# Train VAR model
model = VAR(train)
order = 20  # Order of the VAR model
model_fitted = model.fit(order)

# Forecast using the VAR model
forecast_steps = len(test)
forecast = model_fitted.forecast(test.values, steps=forecast_steps)

# Calculate RMSE for each variable
rmse = np.sqrt(((forecast - test.values) ** 2).mean(axis=0))

# Print the RMSE for each variable
for i, col in enumerate(df.columns):
    print(f"RMSE for {col}: {rmse[i]}")

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Close'], label='Actual')
forecast_index = pd.date_range(start=test.index[0], periods=forecast_steps, freq='D')
plt.plot(forecast_index, forecast[:, 0], label='Forecast', linestyle='--')  # Replace 0 with the column index of 'Close' in your data
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Forecast using VAR')
plt.legend()
plt.show()

# Calculate RMSE for each variable
rmse_var = np.sqrt(((forecast - test.values) ** 2).mean(axis=0))

# Calculate MSE for each variable
mse_var = ((forecast - test.values) ** 2).mean(axis=0)

# Calculate MAE for each variable
mae_var = np.abs(forecast - test.values).mean(axis=0)

# Calculate R2 for each variable
r2_var = r2_score(test.values, forecast, multioutput='raw_values')

# Print the RMSE, MSE, MAE, and R2 for each variable
for i, col in enumerate(df.columns):
    print(f"RMSE for {col}: {rmse_var[i]}")
    print(f"MSE for {col}: {mse_var[i]}")
    print(f"MAE for {col}: {mae_var[i]}")
    print(f"R2 for {col}: {r2_var[i]}")


# In[ ]:





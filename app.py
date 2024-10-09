import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Title of the app
st.title('Price Forecasting with Variance')

# Sample data (replace with actual data)
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
    'Actual_Price': [774.4, 758.7, 784.95, 743.4, 703.55, 705.95]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Month_Number'] = np.arange(1, len(df) + 1)

# Add seasonal feature (Cyclical)
df['Month_of_Year'] = df['Month_Number'] % 12
df['Previous_Month_Price'] = df['Actual_Price'].shift(1).fillna(df['Actual_Price'].mean())  # Lagged feature

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 12, 6)

# Prepare data for the model
X = df[['Month_Number', 'Month_of_Year', 'Previous_Month_Price']]
y = df['Actual_Price']

# Advanced model: XGBoost Regressor
model = XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.01, random_state=42)
model.fit(X, y)

# Predict prices for historical months
df['Predicted_Price'] = model.predict(X)

# Predict future prices
future_months = np.arange(len(df) + 1, len(df) + 1 + num_months)
future_df = pd.DataFrame({
    'Month_Number': future_months,
    'Month_of_Year': future_months % 12,
    'Previous_Month_Price': df['Actual_Price'].iloc[-1]  # Use last available price for future months
})

future_df['Predicted_Price'] = model.predict(future_df[['Month_Number', 'Month_of_Year', 'Previous_Month_Price']])

# Display future prices
st.subheader('Forecasted Prices for Upcoming Months')
st.write(future_df[['Month_Number', 'Predicted_Price']])

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots()
ax.plot(df['Month'], df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Month'], df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
future_month_labels = ['Month ' + str(i) for i in future_df['Month_Number']]
ax.plot(future_month_labels, future_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

ax.set_xlabel('Month')
ax.set_ylabel('Price')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage)
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices')
st.write(df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])

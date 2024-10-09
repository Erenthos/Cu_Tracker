import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Title of the app
st.title('Price Forecasting with Advanced Model (XGBoost)')

# Sample data (you can replace this with actual data)
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Actual_Price': [774.4, 758.7, 784.95, 743.4, 703.55, 705.95, 745.15, 736.8, 708.55, 702.75, 718.95, 727.75]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Month_Number'] = np.arange(1, len(df) + 1)

# Create lag features and rolling means
df['Lag_1'] = df['Actual_Price'].shift(1)
df['Rolling_Mean_3'] = df['Actual_Price'].rolling(window=3).mean()

# Fill NaN values
df.fillna(method='bfill', inplace=True)

# XGBoost Regressor Model
X = df[['Month_Number', 'Lag_1', 'Rolling_Mean_3']]
y = df['Actual_Price']
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict prices for historical months
df['Predicted_Price'] = model.predict(X)

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 12, 6)

# Predict future prices
future_months = np.arange(len(df) + 1, len(df) + 1 + num_months).reshape(-1, 1)
future_df = pd.DataFrame({'Month_Number': np.arange(len(df) + 1, len(df) + 1 + num_months)})
future_df['Lag_1'] = df['Actual_Price'].iloc[-1]  # Use the last known price as lag
future_df['Rolling_Mean_3'] = df['Actual_Price'].rolling(window=3).mean().iloc[-1]  # Use the last rolling mean

# Predict future prices
future_df['Predicted_Price'] = model.predict(future_df)

# Optional: Provide actual values for forecasted months
st.sidebar.subheader('Actual Values for Forecasted Months')
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted months (comma-separated):')

# If actual prices for forecasted months are provided, calculate variance
if actual_future_prices:
    actual_future_prices_list = list(map(float, actual_future_prices.split(',')))
    if len(actual_future_prices_list) == num_months:
        future_df['Actual_Price'] = actual_future_prices_list
        future_df['Variance (%)'] = ((future_df['Predicted_Price'] - future_df['Actual_Price']) / future_df['Actual_Price']) * 100
    else:
        st.error('Please provide the correct number of actual prices.')

# Display future prices
st.subheader('Forecasted Prices for Upcoming Months')
st.write(future_df)

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots()

# Plot actual and predicted prices for historical months
ax.plot(df['Month'], df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Month'], df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
future_month_labels = ['Month ' + str(i) for i in future_df['Month_Number']]
ax.plot(future_month_labels, future_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

# Rotate the x-axis labels to prevent overlap
plt.xticks(rotation=45)

ax.set_xlabel('Month')
ax.set_ylabel('Price')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Adjust the figure size dynamically
fig.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage) for historical data
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices')
st.write(df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])

# Section: Allow users to upload their own CSV for prediction
st.subheader('Upload Your Own Data for Forecasting')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If user uploads a CSV file
if uploaded_file is not None:
    # Read the uploaded file
    user_data = pd.read_csv(uploaded_file)

    # Ensure that the CSV contains a column for 'Month' or 'Date' and 'Price'
    if 'Month' in user_data.columns and 'Price' in user_data.columns:
        st.write('Uploaded Data:')
        st.write(user_data)

        # Generate numerical month values or use an index as the time series
        user_data['Month_Number'] = np.arange(1, len(user_data) + 1)

        # Create lag features and rolling means for uploaded data
        user_data['Lag_1'] = user_data['Price'].shift(1)
        user_data['Rolling_Mean_3'] = user_data['Price'].rolling(window=3).mean()
        user_data.fillna(method='bfill', inplace=True)

        # Train an XGBoost model on the uploaded data
        X_user = user_data[['Month_Number', 'Lag_1', 'Rolling_Mean_3']]
        y_user = user_data['Price']
        user_model = XGBRegressor(n_estimators=100, random_state=42)
        user_model.fit(X_user, y_user)

        # Predict future prices
        future_user_months = np.arange(len(user_data) + 1, len(user_data) + 1 + num_months).reshape(-1, 1)
        future_user_df = pd.DataFrame({'Month_Number': np.arange(len(user_data) + 1, len(user_data) + 1 + num_months)})
        future_user_df['Lag_1'] = user_data['Price'].iloc[-1]
        future_user_df['Rolling_Mean_3'] = user_data['Price'].rolling(window=3).mean().iloc[-1]
        future_user_df['Predicted_Price'] = user_model.predict(future_user_df)

        # Display the forecasted data for uploaded data
        st.subheader('Forecasted Prices for Uploaded Data')
        st.write(future_user_df)

        # Plot actual vs predicted prices for the uploaded data
        st.subheader('Price Forecast Visualization for Uploaded Data')
        fig_user, ax_user = plt.subplots()
        ax_user.plot(user_data['Month'], user_data['Price'], label='Actual Price', marker='o')
        ax_user.plot(user_data['Month'], user_model.predict(X_user), label='Predicted Price', linestyle='--', marker='x')

        # Plot future predicted prices for the uploaded data
        future_user_month_labels = ['Month ' + str(i) for i in future_user_df['Month_Number']]
        ax_user.plot(future_user_month_labels, future_user_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

        ax_user.set_xlabel('Month')
        ax_user.set_ylabel('Price')
        ax_user.set_title('Actual vs Predicted Prices (Uploaded Data)')
        ax_user.legend()

        # Rotate the x-axis labels to prevent overlap
        plt.xticks(rotation=45)

        fig_user.tight_layout()

        # Display the plot for the uploaded data
        st.pyplot(fig_user)

        # Allow user to calculate variance (optional)
        if 'Price' in user_data.columns:
            user_data['Predicted_Price'] = user_model.predict(X_user)
            user_data['Variance (%)'] = ((user_data['Predicted_Price'] - user_data['Price']) / user_data['Price']) * 100

            # Display variance for uploaded data
            st.subheader('Variance between Actual and Predicted Prices (Uploaded Data)')
            st.write(user_data[['Month', 'Price', 'Predicted_Price', 'Variance (%)']])
    else:
        st.error("Please upload a CSV file with 'Month' and 'Price'")

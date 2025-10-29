import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import streamlit as st

# --- CONFIG ---
FOLDER_PATH = r"C:\Users\asus\Documents\STOCKS\all_companies_csv"

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- STEP 1: List available company files ---
def list_companies(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    companies = [os.path.splitext(f)[0] for f in files]
    return companies

# --- STEP 2: Load company data ---
def load_company_data(company_name):
    file_path = os.path.join(FOLDER_PATH, f"{company_name}.csv")
    df = pd.read_csv(file_path)
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("CSV must have 'timestamp' or 'Date' column")
        st.stop()
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    return df

# --- STEP 3: Train SARIMA model and make predictions ---
def predict_stock(df):
    if 'close' not in df.columns:
        st.error("CSV must contain a 'close' column with closing prices.")
        st.stop()

    data = df['close']
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    current_date = datetime.now().date()

    prev_10_days = pd.date_range(end=current_date, periods=10)
    prev_forecast = results.get_prediction(start=len(data)-10, end=len(data)-1)
    prev_pred = prev_forecast.predicted_mean

    next_10_days = pd.date_range(start=current_date, periods=10)
    next_forecast = results.get_forecast(steps=10)
    next_pred = next_forecast.predicted_mean

    prev_df = pd.DataFrame({'Date': prev_10_days, 'Predicted Close': prev_pred.values})
    next_df = pd.DataFrame({'Date': next_10_days, 'Predicted Close': next_pred.values})

    return prev_df, next_df, data

# --- STREAMLIT UI ---
st.title("Time-Series Analysis of Stock Prices Using ARIMA and SARIMA Models")
st.write("This app predicts stock prices for the previous and next 10 days using the SARIMA model with reference to the current system date.")

companies = list_companies(FOLDER_PATH)
if not companies:
    st.error("No CSV files found in the specified folder.")
else:
    company_name = st.selectbox("Select a company:", companies)

    if st.button("Predict Stock Prices"):
        with st.spinner('Training model and generating predictions...'):
            df = load_company_data(company_name)
            prev_df, next_df, data = predict_stock(df)

            st.subheader(f"Previous 10 Days Predicted Prices for {company_name}")
            st.dataframe(prev_df, use_container_width=True)

            st.subheader(f"Next 10 Days Predicted Prices for {company_name}")
            st.dataframe(next_df, use_container_width=True)

            #  Plot results 
            fig, ax = plt.subplots(figsize=(15,4))
            ax.plot(prev_df['Date'], prev_df['Predicted Close'], label='Prev 10 Days', color='blue', linestyle='-')
            ax.plot(next_df['Date'], next_df['Predicted Close'], label='Next 10 Days', color='red', linestyle='-')
            ax.axvline(datetime.now(), color='black', linestyle=':', label='Current Date')
            ax.set_title(f"Stock Price Prediction for {company_name}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            ax.legend()
            ax.grid()
            st.pyplot(fig)


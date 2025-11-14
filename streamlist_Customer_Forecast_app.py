"""
Fresh Streamlit Customer Forecasting App
This is a clean, ready-to-upload version for Streamlit Cloud.

Features:
- Upload CSV file with date & value columns
- Choose forecast method: Prophet or Random Forest
- Select forecast horizon and frequency
- Show historical + forecast charts
- Download forecast CSV

How to deploy:
1. Save this file as `streamlit_customer_forecast_app.py`
2. Create `requirements.txt` with:
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   prophet
   joblib
3. Push both files to GitHub
4. Deploy on Streamlit Cloud (https://share.streamlit.io)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title='Customer Forecast', layout='wide')
st.title('Customer Trend Forecasting App')

# Sidebar: file upload and options
st.sidebar.header('Upload & Settings')
uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
use_sample = st.sidebar.checkbox('Use sample data (monthly)', value=True if not uploaded_file else False)

if use_sample and not uploaded_file:
    rng = pd.date_range(start='2022-01-01', periods=36, freq='M')
    uploaded_df_preview = pd.DataFrame({'date': rng, 'customers': (100 + np.random.randn(len(rng)).cumsum()).astype(int)})
else:
    uploaded_df_preview = None

if uploaded_file or uploaded_df_preview is not None:
    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
    else:
        df_preview = uploaded_df_preview

    date_col = st.sidebar.text_input('Date column name', value='date')
    value_col = st.sidebar.text_input('Value column name', value='customers')
    method = st.sidebar.selectbox('Forecast method', options=['rf'] + (['prophet'] if PROPHET_AVAILABLE else []))
    periods = st.sidebar.number_input('Forecast periods', min_value=1, value=12)
    freq = st.sidebar.selectbox('Frequency', options=['D','W','M','Q','Y'], index=2)
    run = st.sidebar.button('Run Forecast')

    # Show preview
    st.subheader('Data Preview')
    st.dataframe(df_preview.head(50))

    if run:
        df = pd.read_csv(uploaded_file) if uploaded_file else uploaded_df_preview
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, value_col]].rename(columns={date_col:'ds', value_col:'y'})

        # Historical chart
        st.subheader('Historical Series')
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df['ds'], df['y'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True)
        st.pyplot(fig)

        # Forecast
        st.subheader('Forecast Results')
        if method == 'prophet' and PROPHET_AVAILABLE:
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            forecast = m.predict(future)

            st.dataframe(forecast[['ds','yhat']].tail(periods))
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(df['ds'], df['y'], label='history')
            ax2.plot(forecast['ds'], forecast['yhat'], label='forecast')
            ax2.legend()
            st.pyplot(fig2)

            csv_buf = forecast[['ds','yhat']].tail(periods).to_csv(index=False).encode('utf-8')
            st.download_button('Download forecast CSV', data=csv_buf, file_name='forecast.csv', mime='text/csv')

        else:
            # Random Forest forecast
            df_feat = df.copy().set_index('ds')
            for lag in [1,2,3,7,14]:
                df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)
            df_feat = df_feat.dropna().reset_index()

            X = df_feat.drop(columns=['ds','y'])
            y = df_feat['y']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            last_known = df.set_index('ds')
            results = []
            freq_pd = pd.tseries.frequencies.to_offset(freq)
            curr_index = last_known.index.max()
            working = last_known.copy()

            for step in range(periods):
                row = working.tail(14)['y'].values[::-1]
                features = np.zeros((1,X.shape[1]))
                for i, lag in enumerate([1,2,3,7,14]):
                    features[0,i] = row[lag-1] if lag-1 < len(row) else row[-1]
                pred = model.predict(features)[0]
                next_index = curr_index + freq_pd
                results.append({'ds': next_index, 'yhat': pred})
                working.loc[next_index] = [pred]
                curr_index = next_index

            res_df = pd.DataFrame(results)
            st.dataframe(res_df)

            fig3, ax3 = plt.subplots(figsize=(10,4))
            ax3.plot(df['ds'], df['y'], label='history')
            ax3.plot(res_df['ds'], res_df['yhat'], label='forecast')
            ax3.legend()
            st.pyplot(fig3)

            csv_buf = res_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download forecast CSV', data=csv_buf, file_name='forecast_rf.csv', mime='text/csv')

else:
    st.info('Upload a CSV file or select "Use sample data" to begin.')

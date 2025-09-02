import warnings
warnings.filterwarnings("ignore")

import io
import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet (new package name `prophet`, older is `fbprophet`)
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None

# Optional live data
try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Stock Time Series Forecasting", layout="wide")
st.title("Stock Time Series Analysis & Forecasting")
st.caption("Tech: Pandas, NumPy, Plotly, Matplotlib, scikit-learn, Statsmodels, Prophet, TensorFlow/Keras (LSTM)")

# Sidebar controls
with st.sidebar:
    st.header("Data Source")
    data_mode = st.radio("Choose data input", ["Yahoo Finance Ticker", "Upload CSV"], index=0)

    if data_mode == "Yahoo Finance Ticker":
        ticker = st.text_input("Ticker (e.g., AAPL, TCS.NS)", value="AAPL")
        start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=365*3))
        end_date = st.date_input("End date", value=datetime.today())
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
        date_col = st.text_input("Date column name", value="Date")
        price_col = st.text_input("Price/Close column name", value="Close")

    st.header("Forecast Settings")
    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=60, step=1)
    model_choice = st.multiselect(
        "Models to run",
        ["Baseline (Naive)", "ARIMA (Statsmodels)", "Prophet", "LSTM (Keras)"],
        default=["Baseline (Naive)", "ARIMA (Statsmodels)", "Prophet"]
    )
    test_size_days = st.number_input("Test split size (days from end)", min_value=14, max_value=365, value=60, step=1)

    st.header("Advanced")
    arima_order_p = st.slider("ARIMA p", 0, 5, 1)
    arima_order_d = st.slider("ARIMA d", 0, 2, 1)
    arima_order_q = st.slider("ARIMA q", 0, 5, 1)

    lstm_epochs = st.slider("LSTM epochs", 1, 50, 10)
    lstm_window = st.slider("LSTM lookback window", 5, 60, 20)

# Data loading helpers
@st.cache_data(show_spinner=False)
def load_yf(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install it or use CSV upload.")
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()
    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    df = df[["date", "close"]].dropna()
    return df

@st.cache_data(show_spinner=False)
def load_csv(file, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{price_col}'.")
    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.sort_values("date").dropna()
    return df

# Get data
df = None
try:
    if data_mode == "Yahoo Finance Ticker":
        if ticker:
            df = load_yf(ticker, start_date, end_date)
    else:
        if uploaded is not None:
            df = load_csv(uploaded, date_col, price_col)
except Exception as e:
    st.error(f"Data loading error: {e}")

if df is None or df.empty:
    st.info("Provide a ticker and dates or upload a CSV to begin.")
    st.stop()

# Basic EDA
st.subheader("Exploratory Data Analysis")
col1, col2 = st.columns([2, 1])
with col1:
    fig1 = px.line(df, x="date", y="close", title="Close Price Over Time")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.write("Summary Stats")
    st.dataframe(df["close"].describe().to_frame())

# Train/test split by date tail N days as test
df = df.sort_values("date").reset_index(drop=True)
cutoff_date = df["date"].iloc[-1] - timedelta(days=int(test_size_days))
train = df[df["date"] <= cutoff_date].copy()
test = df[df["date"] > cutoff_date].copy()

st.write(f"Train range: {train['date'].min().date()} → {train['date'].max().date()} | Test range: {test['date'].min().date()} → {test['date'].max().date()}")

# Helper: evaluation metrics

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

results = []
forecast_frames = []

# BASELINE (Naive: last observed value)
if "Baseline (Naive)" in model_choice:
    yhat_test = np.roll(test["close"].values, 1)
    if len(train) > 0:
        yhat_test[0] = train["close"].iloc[-1]
    metrics = eval_metrics(test["close"].values, yhat_test)
    results.append(("Baseline (Naive)", metrics))
    # Future forecast: hold at last observed
    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")
    future_values = np.full(shape=(horizon,), fill_value=df["close"].iloc[-1])
    baseline_fc = pd.DataFrame({"date": future_dates, "forecast": future_values, "model": "Baseline (Naive)"})
    forecast_frames.append(baseline_fc)

# ARIMA (Statsmodels SARIMAX) – simple (p,d,q) from sidebar
if "ARIMA (Statsmodels)" in model_choice:
    try:
        endog = train.set_index("date")["close"].asfreq("D").interpolate()
        test_series = test.set_index("date")["close"].asfreq("D").interpolate()
        model = SARIMAX(endog, order=(arima_order_p, arima_order_d, arima_order_q), enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        # In-sample test forecast
        pred_test = fit.get_forecast(steps=len(test_series))
        yhat_test = pred_test.predicted_mean.values
        metrics = eval_metrics(test_series.values, yhat_test)
        results.append(("ARIMA (Statsmodels)", metrics))
        # Future forecast
        fut = fit.get_forecast(steps=int(horizon))
        fc_values = fut.predicted_mean.values
        fc_index = pd.date_range(start=df["date"].iloc[-1] + timedelta(days=1), periods=horizon, freq="D")
        arima_fc = pd.DataFrame({"date": fc_index, "forecast": fc_values, "model": "ARIMA (Statsmodels)"})
        forecast_frames.append(arima_fc)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")

# Prophet model
if "Prophet" in model_choice:
    if Prophet is None:
        st.warning("Prophet is not installed. Install `prophet` or `fbprophet` to use this model.")
    else:
        try:
            df_prophet = train.rename(columns={"date": "ds", "close": "y"})[["ds", "y"]]
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(df_prophet)
            # Create df for test period
            test_days = (test["date"].iloc[-1] - test["date"].iloc[0]).days + 1 if len(test) > 0 else 0
            total_days = len(train) + len(test)
            last_date = df["date"].iloc[-1]
            horizon_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")

            # Predict on history + test
            future_hist = pd.DataFrame({"ds": pd.date_range(start=train["date"].min(), end=test["date"].max() if len(test) else train["date"].max(), freq="D")})
            fc_hist = m.predict(future_hist)
            pred_test = fc_hist.set_index("ds").loc[test["date"], ["yhat"]]["yhat"].values if len(test) else np.array([])
            if len(test):
                metrics = eval_metrics(test["close"].values, pred_test)
                results.append(("Prophet", metrics))
            # Future forecast
            future = pd.DataFrame({"ds": horizon_dates})
            fc_future = m.predict(future)
            prophet_fc = pd.DataFrame({
                "date": fc_future["ds"],
                "forecast": fc_future["yhat"],
                "model": "Prophet"
            })
            forecast_frames.append(prophet_fc)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")

# LSTM (univariate)
if "LSTM (Keras)" in model_choice:
    if tf is None:
        st.warning("TensorFlow/Keras not installed. Install `tensorflow` to use LSTM model.")
    else:
        try:
            series = df.set_index("date")["close"].asfreq("D").interpolate()
            # Train/test index split
            split_idx = series.index.get_loc(cutoff_date, method="nearest")
            series_train = series.iloc[:split_idx+1]
            series_test = series.iloc[split_idx+1:]

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series_train.values.reshape(-1, 1))

            def make_sequences(arr, window):
                X, y = [], []
                for i in range(window, len(arr)):
                    X.append(arr[i-window:i, 0])
                    y.append(arr[i, 0])
                X = np.array(X)
                y = np.array(y)
                return X.reshape((X.shape[0], X.shape[1], 1)), y

            X_train, y_train = make_sequences(scaled, lstm_window)

            model = Sequential()
            model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=int(lstm_epochs), batch_size=32, verbose=0)

            # Predict test period recursively
            history = scaler.transform(series_train.values.reshape(-1, 1)).flatten().tolist()
            preds = []
            test_len = len(series_test)
            for _ in range(test_len):
                x_input = np.array(history[-lstm_window:]).reshape((1, lstm_window, 1))
                yhat = model.predict(x_input, verbose=0)[0, 0]
                preds.append(yhat)
                history.append(yhat)
            yhat_test = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            metrics = eval_metrics(series_test.values, yhat_test)
            results.append(("LSTM (Keras)", metrics))

            # Future forecast horizon
            future_scaled = history.copy()
            fut_preds = []
            for _ in range(int(horizon)):
                x_input = np.array(future_scaled[-lstm_window:]).reshape((1, lstm_window, 1))
                yhat = model.predict(x_input, verbose=0)[0, 0]
                fut_preds.append(yhat)
                future_scaled.append(yhat)
            fut_values = scaler.inverse_transform(np.array(fut_preds).reshape(-1, 1)).flatten()

            fut_index = pd.date_range(start=df["date"].iloc[-1] + timedelta(days=1), periods=horizon, freq="D")
            lstm_fc = pd.DataFrame({"date": fut_index, "forecast": fut_values, "model": "LSTM (Keras)"})
            forecast_frames.append(lstm_fc)
        except Exception as e:
            st.warning(f"LSTM failed: {e}")

# Metrics table
if results:
    st.subheader("Test Metrics")
    metrics_df = pd.DataFrame(
        [
            {"Model": name, **m} for name, m in results
        ]
    )
    st.dataframe(metrics_df.set_index("Model"))
else:
    st.info("No metrics available yet. Try enabling at least one model and ensure test split is valid.")

# Combine forecasts
if forecast_frames:
    fc_all = pd.concat(forecast_frames, ignore_index=True)
    st.subheader("Forecast Visualization")
    hist_tail = df.tail(200)  # show recent history for clarity
    fig_fc = px.line(hist_tail, x="date", y="close", labels={"close": "Close", "date": "Date"}, title="History and Forecast")
    for model_name in fc_all["model"].unique():
        sub = fc_all[fc_all["model"] == model_name]
        fig_fc.add_scatter(x=sub["date"], y=sub["forecast"], mode="lines", name=f"{model_name} forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Download forecasts
    csv_buf = io.StringIO()
    fc_all.to_csv(csv_buf, index=False)
    st.download_button("Download forecast CSV", data=csv_buf.getvalue(), file_name="forecasts.csv", mime="text/csv")
else:
    st.info("Run a model to see forecasts.")

st.markdown("---")
st.subheader("Methodology (Brief)")
st.markdown(
   
    """
    - **Baseline**: Last observed value carried forward.  
    - **ARIMA**: SARIMAX on daily frequency with user-chosen (p,d,q). Interpolates missing days.  
    - **Prophet**: Additive model with daily and yearly seasonality; predicts on calendar days.  
    - **LSTM**: Univariate sequence model on scaled series with recursive multi-step forecasting.  

    👉 Tip: Tune ARIMA orders, Prophet seasonality/holidays, and LSTM epochs/window for better accuracy.
    """
)



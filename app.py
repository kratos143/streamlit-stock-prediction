import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Input
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure Tensorflow
tf.get_logger().setLevel('ERROR')

# Streamlit page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load and cache stock data"""
    return yf.download(ticker, start=start_date, end=end_date, interval="1d")

def create_model(time_steps, n_features, lstm_units_1, lstm_units_2, dense_units, dropout_rate):
    """Create LSTM model with configurable parameters"""
    model = Sequential()
    
    model.add(Input(shape=(time_steps, n_features)))
    
    model.add(LSTM(units=lstm_units_1, return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=lstm_units_2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(units=dense_units))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def add_indicators(df):
    """Add RSI, SMA, and open price to dataframe"""
    # Add RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Add SMA (Simple Moving Average)
    df['SMA'] = ta.sma(df['Close'], length=9)
    
    # Add open price
    df['Open'] = df['Open']
    
    # Forward fill and backward fill NaN values
    df = df.ffill().bfill()
    return df

def prepare_stock_data(data_True):
    """Prepare stock data with indicators"""
    dates = pd.DatetimeIndex(pd.to_datetime(data_True.index))
    df_with_indicators = add_indicators(data_True)
    feature_columns = ['Close', 'Open', 'RSI', 'SMA']
    features = df_with_indicators[feature_columns].values
    return features, dates, df_with_indicators

def prepare_data(data, time_steps):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps, 0])
    return np.array(X), np.array(y), scaler

def predict_future(model, last_sequence, scaler, n_future):
    """Predict future stock prices"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        with tf.device('/CPU:0'):
            current_prediction = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        
        dummy_row = np.zeros((1, scaler.scale_.shape[0]))
        dummy_row[0, 0] = current_prediction.item()
        predicted_price = scaler.inverse_transform(dummy_row)[0, 0]
        future_predictions.append(predicted_price)
        
        new_row = current_sequence[-1].copy()
        new_row[0] = current_prediction.item()
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_row
        
    return np.array(future_predictions).reshape(-1, 1)

def plot_all_data(dates, df_indicators, actual_values, train_predictions, test_predictions, 
                  future_dates, future_predictions, train_size, time_steps, ticker):
    """Create interactive plotly chart"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Predictions', 'RSI', 'Volume'),
        row_heights=[0.5, 0.25, 0.25]
    )

    plot_dates = dates[time_steps:]
    
    # Main price chart
    fig.add_trace(
        go.Scatter(x=plot_dates, y=actual_values.flatten(),
                  name='Actual Price', line=dict(color='blue')), row=1, col=1)
    
    # Add SMA
    fig.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators['SMA'],
                  name='9-day SMA', line=dict(color='orange')), row=1, col=1)
    
    # Training predictions
    train_dates = plot_dates[:train_size]
    fig.add_trace(
        go.Scatter(x=train_dates, y=train_predictions.flatten(),
                  name='Training Predictions', line=dict(color='green', dash='dash')), row=1, col=1)
    
    # Testing predictions
    test_dates = plot_dates[train_size:]
    fig.add_trace(
        go.Scatter(x=test_dates, y=test_predictions.flatten(),
                  name='Testing Predictions', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Future predictions
    fig.add_trace(
        go.Scatter(x=future_dates, y=future_predictions.flatten(),
                  name='Future Predictions', line=dict(color='purple', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators['RSI'],
                  name='RSI', line=dict(color='blue')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df_indicators.index, y=df_indicators['Volume'],
               name='Volume'), row=3, col=1)

    fig.update_layout(
        title=f'{ticker} Stock Analysis and Predictions',
        height=1000,
        showlegend=True,
        xaxis3_title='Date',
        yaxis_title='Price',
        yaxis2_title='RSI',
        yaxis3_title='Volume'
    )

    return fig

def main():
    st.title("Stock Price Prediction with RSI and SMA")
    
    # Create two main columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Basic parameters
        ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", value="AAPL")
        future_days = st.slider("Future days to predict:", 1, 60, value=30)
        
        # Date selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("2024-10-31"))
    
    with right_col:
        st.subheader("Model Parameters")
        
        # Training parameters
        epochs = st.slider("Number of Epochs:", 10, 200, 50)
        batch_size = st.slider("Batch Size:", 8, 128, 32, 8)
        time_steps = st.slider("Time Steps (Sequence Length):", 10, 100, 60)
        
        # Model architecture parameters
        st.markdown("### LSTM Architecture")
        lstm_units_1 = st.slider("First LSTM Layer Units:", 20, 200, 50, 10)
        lstm_units_2 = st.slider("Second LSTM Layer Units:", 20, 200, 50, 10)
        dense_units = st.slider("Dense Layer Units:", 10, 100, 25, 5)
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.2, 0.1)
        
        # Training split
        train_split = st.slider("Training Data Split (%):", 60, 90, 80, 5)

    if st.button("Train Model and Predict"):
        try:
            with st.spinner("Loading data and making predictions..."):
                data = load_data(ticker, start_date, end_date)
                
                if data.empty:
                    st.error("No data found for this ticker and date range.")
                    return

                # Prepare data
                features, dates, df_indicators = prepare_stock_data(data)
                X, y, scaler = prepare_data(features, time_steps)
                
                # Split data using the user-defined split
                train_size = int(len(X) * (train_split / 100))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                n_features = X.shape[2]

                # Create progress bar for training
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback to update progress
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f'Training Progress: {int(progress * 100)}%')
                
                # Create and train model
                model = create_model(
                    time_steps, 
                    n_features, 
                    lstm_units_1, 
                    lstm_units_2, 
                    dense_units, 
                    dropout_rate
                )
                
                # Training with progress tracking
                history = model.fit(
                    X_train, 
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[ProgressCallback()]
                )

                # Make predictions
                train_predictions = model.predict(X_train, verbose=0)
                test_predictions = model.predict(X_test, verbose=0)
                last_sequence = X[-1]
                future_predictions = predict_future(model, last_sequence, scaler, future_days)

                # Display training metrics
                st.subheader("Training Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot training history
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig_loss.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig_loss)
                
                with col2:
                    # Display key metrics
                    train_mse = model.evaluate(X_train, y_train, verbose=0)
                    test_mse = model.evaluate(X_test, y_test, verbose=0)
                    st.metric("Training MSE", f"{train_mse:.4f}")
                    st.metric("Testing MSE", f"{test_mse:.4f}")

                # Transform predictions
                dummy_train = np.zeros((len(train_predictions), scaler.scale_.shape[0]))
                dummy_train[:, 0] = train_predictions.flatten()
                dummy_test = np.zeros((len(test_predictions), scaler.scale_.shape[0]))
                dummy_test[:, 0] = test_predictions.flatten()
                dummy_actual = np.zeros((len(y), scaler.scale_.shape[0]))
                dummy_actual[:, 0] = y

                train_predictions = scaler.inverse_transform(dummy_train)[:, 0].reshape(-1, 1)
                test_predictions = scaler.inverse_transform(dummy_test)[:, 0].reshape(-1, 1)
                actual_values = scaler.inverse_transform(dummy_actual)[:, 0].reshape(-1, 1)

                # Create future dates
                future_dates = pd.date_range(dates[-1] + timedelta(days=1), periods=future_days, freq='B')
                
                # Plot results
                fig = plot_all_data(dates, df_indicators, actual_values, train_predictions, 
                                  test_predictions, future_dates, future_predictions, 
                                  train_size, time_steps, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display predictions table
                st.subheader("Future Price Predictions:")
                predictions_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions.flatten()
                })
                st.dataframe(predictions_df.style.format({'Predicted Price': '{:.2f}'}))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

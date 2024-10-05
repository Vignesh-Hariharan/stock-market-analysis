import numpy as np
import pandas as pd
import sqlite3
import os
import logging
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(db_path, ticker=None):
    """Load data from SQLite database"""
    try:
        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM stock_data"
            if ticker:
                query += f" WHERE Ticker = '{ticker}'"
            df = pd.read_sql_query(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {db_path}: {e}")
        raise

def prepare_data(df, feature_columns, target_column, sequence_length=60):
    """Prepare data for LSTM model"""
    features = df[feature_columns].values
    target = df[target_column].values

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_target[i])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def build_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Train the LSTM model"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    return history, model

def evaluate_model(model, X_test, y_test, target_scaler):
    """Evaluate the model and print metrics"""
    y_pred = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R-squared Score: {r2}")

    return y_test_inv, y_pred_inv

def plot_results(y_test, y_pred, dates, title, output_dir):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, label='Actual')
    plt.plot(dates, y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(output_path)
    logger.info(f"Plot saved as {output_path}")

def predict_future(model, last_sequence, scaler, steps=30):
    """Make future predictions"""
    future_predictions = []
    current_sequence = last_sequence[-1].reshape((1, last_sequence.shape[1], last_sequence.shape[2]))

    for _ in range(steps):
        prediction = model.predict(current_sequence)
        future_predictions.append(prediction[0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = prediction

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Stock Price Prediction Model")
    parser.add_argument("--db_path", default="../data/stock_data.db", help="Path to the SQLite database")
    parser.add_argument("--ticker", help="Specific stock ticker to model (optional)")
    parser.add_argument("--output_dir", default="../output", help="Directory to save output files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.db_path, args.ticker)

    # Prepare data
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA7', 'MA30', 'RSI']
    target_column = 'Close'
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_data(df, feature_columns, target_column)

    # Build and train model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history, trained_model = train_model(model, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate model
    y_test_inv, y_pred_inv = evaluate_model(trained_model, X_test, y_test, target_scaler)
    test_dates = df['Date'].iloc[-len(y_test_inv):]
    plot_results(y_test_inv, y_pred_inv, test_dates, f'Stock Price Prediction for {args.ticker if args.ticker else "All Stocks"}', args.output_dir)

    # Make future predictions
    last_sequence = X_test[-1]
    future_pred = predict_future(trained_model, last_sequence, target_scaler)
    
    # Plot future predictions
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_pred)+1)]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'][-100:], df['Close'][-100:], label='Historical Data')
    plt.plot(future_dates, future_pred, label='Future Prediction')
    plt.title(f'Future Stock Price Prediction for {args.ticker if args.ticker else "All Stocks"}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    future_plot_path = os.path.join(args.output_dir, 'future_prediction.png')
    plt.savefig(future_plot_path)
    logger.info(f"Future prediction plot saved as {future_plot_path}")

    # Save the model
    model_path = os.path.join(args.output_dir, 'final_model.h5')
    trained_model.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

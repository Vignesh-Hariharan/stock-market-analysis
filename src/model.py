
import numpy as np
import pandas as pd
import sqlite3
import os
import logging
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from keras_tuner import RandomSearch
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(db_path):
    """Load data from SQLite database"""
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM stock_data", conn)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna()  # Remove any rows with NaN values after date conversion
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {db_path}: {e}")
        raise

def prepare_data(df, feature_columns, target_column, sequence_length=60):
    """Prepare data for LSTM model"""
    try:
        df = df.sort_values('Date')
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

        return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, df['Date'][-len(X_test):]
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def build_model(hp):
    """Build and compile LSTM model with hyperparameter tuning"""
    try:
        model = Sequential()
        model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                                     return_sequences=True, input_shape=(60, 7))))
        model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32))))
        model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=1))

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                      loss='mean_squared_error')
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}")
        raise

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """Train the LSTM model"""
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'model_{datetime.now().strftime("%Y%m%d_%H%M")}.h5', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        return history, model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test, target_scaler):
    """Evaluate the model and print metrics"""
    try:
        y_pred = model.predict(X_test)
        y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = target_scaler.inverse_transform(y_pred)

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        logger.info(f"Mean Squared Error: {mse}")
        logger.info(f"Mean Absolute Error: {mae}")
        logger.info(f"R-squared Score: {r2}")

        return y_test_inv, y_pred_inv, {'mse': mse, 'mae': mae, 'r2': r2}
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def plot_results(y_test, y_pred, dates, title):
    """Plot actual vs predicted values"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_test, label='Actual')
        plt.plot(dates, y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        logger.info(f"Plot saved as {title.replace(' ', '_').lower()}.png")
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
        raise

def make_future_prediction(model, last_sequence, target_scaler, steps=30):
    """Make future predictions"""
    try:
        future_predictions = []
        current_sequence = last_sequence[-1].reshape((1, last_sequence.shape[1], last_sequence.shape[2]))

        for _ in range(steps):
            prediction = model.predict(current_sequence)
            future_predictions.append(prediction[0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = prediction

        future_predictions = np.array(future_predictions)
        future_predictions = target_scaler.inverse_transform(future_predictions)
        return future_predictions.flatten()
    except Exception as e:
        logger.error(f"Error making future predictions: {e}")
        raise

def plot_feature_importance(model, X_train, feature_names):
    """Plot feature importance using SHAP values"""
    try:
        # Use a smaller subset of data for SHAP analysis to improve performance
        sample_size = min(100, len(X_train))
        X_sample = X_train[:sample_size]
        
        explainer = shap.DeepExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
        plt.savefig('feature_importance.png')
        plt.close()
        logger.info("Feature importance plot saved as feature_importance.png")
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        raise

def save_results(metrics, model_path):
    """Save model performance metrics and model path"""
    try:
        results = {
            'metrics': metrics,
            'model_path': model_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open('model_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Model results saved to model_results.json")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    try:
        # Load data
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        db_file = os.path.join(data_dir, 'stock_data.db')
        df = load_data(db_file)

        # Prepare data
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA7']
        target_column = 'Close'
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler, test_dates = prepare_data(df, feature_columns, target_column)

        # Hyperparameter tuning
        tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='hyperparameter_tuning',
            project_name='stock_prediction'
        )

        tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
        best_model = tuner.get_best_models(num_models=1)[0]

        # Train model
        history, trained_model = train_model(best_model, X_train, y_train, X_test, y_test)

        # Evaluate model
        y_test_inv, y_pred_inv, metrics = evaluate_model(trained_model, X_test, y_test, target_scaler)
        plot_results(y_test_inv, y_pred_inv, test_dates, 'Stock Price Prediction')

        # Plot feature importance
        plot_feature_importance(trained_model, X_train, feature_columns)

        # Make future predictions
        last_sequence = X_test[-1]
        future_pred = make_future_prediction(trained_model, last_sequence, target_scaler)
        
        # Plot future predictions
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_pred)+1)]
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'][-100:], df['Close'][-100:], label='Historical Data')
        plt.plot(future_dates, future_pred, label='Future Prediction')
        plt.title('Future Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('future_prediction.png')
        plt.close()
        logger.info("Future prediction plot saved as future_prediction.png")

        # Save model and results
        model_path = f'final_model_{datetime.now().strftime("%Y%m%d_%H%M")}.h5'
        trained_model.save(model_path)
        save_results(metrics, model_path)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()

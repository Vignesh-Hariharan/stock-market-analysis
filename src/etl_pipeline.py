import pandas as pd
import sqlite3
from datetime import datetime
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data(file_path):
    """
    Extract data from CSV file
    """
    logger.info(f"Extracting data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully extracted {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {e}")
        raise

def transform_data(df):
    """
    Transform the data
    """
    logger.info("Transforming data...")
    
    try:
        # Convert 'Date' to datetime, handle invalid formats
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate 7-day moving average
        df['MA7'] = df['Close'].rolling(window=7).mean()
        
        # Drop rows with NaN values
        original_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {original_len - len(df)} rows with NaN values")
        
        return df
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise

def load_data(df, db_path):
    """
    Load data into SQLite database
    """
    logger.info(f"Loading data into {db_path}...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS stock_data
            (Date TEXT PRIMARY KEY, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, 
            Daily_Return REAL, MA7 REAL)
            ''')
            # Insert data using a transaction
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
        
        logger.info(f"Successfully loaded {len(df)} rows of data")
    except Exception as e:
        logger.error(f"Error loading data into {db_path}: {e}")
        raise

def etl_process(csv_file, db_file):
    """
    Perform the full ETL process
    """
    logger.info("Starting ETL process...")

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        logger.error(f"Error: {csv_file} not found. Please run the scraper first.")
        return

    # Check if database file exists
    if not os.path.exists(db_file):
        logger.info(f"Database file {db_file} does not exist. A new one will be created.")
    
    try:
        # Extract
        df = extract_data(csv_file)
        
        # Transform
        df = transform_data(df)
        
        # Load
        load_data(df, db_file)
        
        logger.info("ETL process completed successfully!")
    except Exception as e:
        logger.error(f"ETL process failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL process for stock data")
    parser.add_argument("--csv", help="Path to input CSV file", default="stock_data.csv")
    parser.add_argument("--db", help="Path to output SQLite database file", default="stock_data.db")
    args = parser.parse_args()

    # Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_file = os.path.abspath(os.path.join(data_dir, args.csv))
    db_file = os.path.abspath(os.path.join(data_dir, args.db))
    
    etl_process(csv_file, db_file)

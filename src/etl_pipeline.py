import pandas as pd
import sqlite3
from datetime import datetime
import os
import logging
import argparse
import glob

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
        logger.info(f"Successfully extracted {len(df)} rows of data from {file_path}")
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
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure all required columns are present
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the data")

        # Calculate daily returns if not already present
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()

        # Calculate 7-day moving average if not already present
        if 'MA7' not in df.columns:
            df['MA7'] = df.groupby('Ticker')['Close'].rolling(window=7).mean().reset_index(0, drop=True)

        # Calculate 30-day moving average
        df['MA30'] = df.groupby('Ticker')['Close'].rolling(window=30).mean().reset_index(0, drop=True)

        # Calculate volatility (20-day rolling standard deviation of returns)
        df['Volatility'] = df.groupby('Ticker')['Daily_Return'].rolling(window=20).std().reset_index(0, drop=True)

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
            (Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, 
            Daily_Return REAL, MA7 REAL, MA30 REAL, Volatility REAL,
            PRIMARY KEY (Date, Ticker))
            ''')
            # Insert data using a transaction
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
        
        logger.info(f"Successfully loaded {len(df)} rows of data")
    except Exception as e:
        logger.error(f"Error loading data into {db_path}: {e}")
        raise

def etl_process(input_dir, db_file):
    """
    Perform the full ETL process for all CSV files in the input directory
    """
    logger.info("Starting ETL process...")
    
    all_data = []
    csv_files = glob.glob(os.path.join(input_dir, '*_stock_data.csv'))
    
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}. Please run the scraper first.")
        return

    for csv_file in csv_files:
        try:
            # Extract
            df = extract_data(csv_file)
            
            # Transform
            df = transform_data(df)
            
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")

    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Load
        load_data(combined_df, db_file)
        
        logger.info("ETL process completed successfully!")
    else:
        logger.error("No data to load. ETL process failed.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="ETL pipeline for stock data")
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(__file__), '..', 'data'), 
                        help="Directory containing the input CSV files")
    parser.add_argument("--db_file", default=os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.db'), 
                        help="Path to the output SQLite database file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.db_file), exist_ok=True)
    
    etl_process(args.input_dir, args.db_file)

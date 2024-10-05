# 🚀 StockSense: Real-Time Market Analysis & Prediction Platform

![StockSense Banner](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/stocksense_banner.png)

## 🌟 Project Overview

StockSense is a cutting-edge platform that leverages real-time data and artificial intelligence to provide unparalleled insights into stock market trends and predictions. By combining advanced web scraping techniques, robust ETL processes, cloud-based data warehousing, state-of-the-art machine learning models, and interactive data visualizations, StockSense empowers investors to make informed decisions with confidence.

### 🔑 Key Features

- **Real-time Data Scraping**: Stay ahead of the market with up-to-the-minute stock information
- **Robust ETL Pipeline**: Efficiently process and load data into a scalable cloud data warehouse
- **Intelligent Storage**: Leverage the power of Snowflake or Redshift for efficient data management
- **Predictive Analytics**: Utilize LSTM neural networks to forecast market trends
- **Interactive Visualizations**: Gain deep insights through stunning, interactive data representations

## 🛠 Technology Stack

StockSense harnesses a powerful combination of technologies:

- **Python Ecosystem**:
  - 🕷 **Selenium & BeautifulSoup**: For nimble web scraping
  - 🐼 **Pandas**: Data manipulation and analysis
  - 🧠 **TensorFlow & scikit-learn**: Powering our predictive AI models
  - 🔗 **SQLAlchemy**: Seamless database interactions

- **Data Warehousing**:
  - ❄ **Snowflake** or 🚀 **Redshift**: Scalable, cloud-native storage solutions

- **Visualization**:
  - 📊 **Plotly**: Create interactive HTML-based charts
  - 📈 **Matplotlib**: Generate static visualizations

## 📂 Project Structure

```
stock_market_analysis/
├── data/
├── src/
│   ├── scraper.py
│   ├── etl_pipeline.py
│   ├── model.py
│   └── visualization.py
├── output/
│   ├── AAPL_candlestick.html
│   ├── MSFT_candlestick.html
│   ├── GOOGL_candlestick.html
│   ├── AAPL_advanced_technical.html
│   ├── MSFT_advanced_technical.html
│   ├── GOOGL_advanced_technical.html
│   ├── AAPL_summary_statistics.csv
│   ├── MSFT_summary_statistics.csv
│   ├── GOOGL_summary_statistics.csv
│   ├── AAPL_seasonal_decomposition.png
│   ├── MSFT_seasonal_decomposition.png
│   ├── GOOGL_seasonal_decomposition.png
│   └── correlation_heatmap.png
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/Vignesh-Hariharan/stock-market-analysis.git
   cd stock-market-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your database credentials in a `.env` file:
   ```
   DB_USERNAME=your_username
   DB_PASSWORD=your_password
   DB_HOST=your_host
   DB_NAME=your_database_name
   ```

5. Run the main script:
   ```
   python src/main.py
   ```

## 📊 Data Flow

1. **Web Scraping**: Real-time stock data is scraped from financial websites using Selenium and BeautifulSoup.
2. **Data Storage**: Scraped data is initially saved as CSV files in the `data/` directory.
3. **ETL Process**: Data is extracted from CSV, transformed, and loaded into Snowflake/Redshift using SQLAlchemy.
4. **Machine Learning**: Historical data is used to train an LSTM model for price predictions.
5. **Visualization**: Various visualizations are generated and saved in the `output/` directory.

## 🧠 Machine Learning Model

StockSense uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical trends. The model is implemented using TensorFlow and trained on scaled historical price data.

![LSTM Model Architecture](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/lstm_architecture.png)

## 📈 Outputs and Visualizations

StockSense generates a variety of outputs to provide comprehensive insights:

### 1. Candlestick Charts
Interactive HTML charts for AAPL, MSFT, and GOOGL stocks.

![Sample Candlestick Chart](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/sample_candlestick.png)

### 2. Advanced Technical Indicator Plots
HTML-based interactive charts with various technical indicators.

![Sample Technical Indicators](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/sample_technical_indicators.png)

### 3. Summary Statistics
CSV files containing key statistical measures for each stock.

### 4. Seasonal Decomposition
PNG images showing trend, seasonal, and residual components of each stock's time series.

![Sample Seasonal Decomposition](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/sample_seasonal_decomposition.png)

### 5. Correlation Heatmap
A PNG image visualizing the correlation between different stocks.

![Correlation Heatmap](https://github.com/Vignesh-Hariharan/stock-market-analysis/raw/main/stock-market-analysis/output/correlation_heatmap.png)

All these outputs can be found in the `output/` directory of the project.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<p align="center">Made with ❤️ by the StockSense Team</p>

import yfinance as yf
import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_and_prepare_advanced_data(ticker="AAPL", period="2y"):
    print(f"--> Fetching market data & engineering advanced features for {ticker}...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError("No data returned. Check internet connection or ticker.")

    # 1. Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 2. RSI
    df['RSI_14'] = calculate_rsi(df['Close'], window=14)

    # 3. MACD (12-day EMA - 26-day EMA) & Signal Line (9-day EMA)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands (20-day, 2 Standard Deviations)
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_20 * 2)

    # 5. Volatility: Average True Range (ATR_14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # 6. Volume & Price Returns
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Daily_Return'] = df['Close'].pct_change()

    # 7. Target Variable (1 if tomorrow's price goes UP, 0 if DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Clean NaN values from rolling windows
    df.dropna(inplace=True)

    output_filename = "historical_stock_data.csv"
    df.to_csv(output_filename)
    print(f"--> Success! Generated {len(df)} rows with 10 technical features in '{output_filename}'.")
    return df

if __name__ == "__main__":
    fetch_and_prepare_advanced_data(ticker="AAPL", period="2y")

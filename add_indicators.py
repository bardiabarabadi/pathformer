import pandas as pd
import numpy as np

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load the data
df = pd.read_csv('./dataset/BTCUSD/BTCUSD_1m.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Add technical indicators
df['SMA_5'] = calculate_sma(df['Close'], 5)
df['SMA_10'] = calculate_sma(df['Close'], 10)
df['SMA_20'] = calculate_sma(df['Close'], 20)
df['EMA_12'] = calculate_ema(df['Close'], 12)
df['EMA_26'] = calculate_ema(df['Close'], 26)
df['RSI_14'] = calculate_rsi(df['Close'], 14)
df['Volume_SMA_5'] = calculate_sma(df['Volume'], 5)

# Drop rows with NaN (from rolling calculations)
df = df.dropna()

# Save back to CSV
df.to_csv('./dataset/BTCUSD/BTCUSD_1m.csv', index=False)

print("Indicators added successfully. New columns:", ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI_14', 'Volume_SMA_5'])
print(f"Data shape: {df.shape}")
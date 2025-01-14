import ccxt
import pandas as pd

def fetch_historical_data(symbol="BTC/USDT", interval="1h", limit=1000):
    """
    Fetch historical OHLCV data from Binance.
    """
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def preprocess_data(df):
    """
    Preprocess historical data for training.
    """
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)
    return df[['timestamp', 'close', 'return']]
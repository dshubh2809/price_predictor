*Assignment 2: Cryptocurrency Price Predictor*  
   An AI-powered system that predicts future cryptocurrency price movements using historical price data from Binance. It uses an LSTM (Long Short-Term Memory) model for accurate predictions.

#### *Features*
- Fetches historical OHLCV data (Open, High, Low, Close, Volume) from Binance.
- Preprocesses data for training.
- Trains an LSTM (Long Short-Term Memory) model to predict price movements.
- Performs backtesting to evaluate the model's performance.
- Provides a frontend interface to:
- Fetch historical data.
- Train models.
- Backtest and view predictions.

#### *How It Works*
1. *Fetch Historical Data*:
   - Uses Binance API to fetch historical price data for a specified trading pair.
2. *Preprocess Data*:
   - Normalizes and prepares data for the LSTM model.
3. *Train AI Model*:
   - Builds and trains an LSTM model to predict whether the price will go up or down.
4. *Backtest Model*:
   - Validates predictions against historical data to calculate accuracy and performance.
5. *Predict Future Prices*:
   - Uses the trained model to predict future price movements.


#### *Usage Instructions*
1. Start the backend API:
   python backend.py
   
2. Start the frontend:
   streamlit run app.py

#### *Project Structure*

price_predictor/
─ app.py                 # Frontend (Streamlit UI)
─ backend.py             # Core logic and API routes
─ utils.py               # Helper functions
─ models/
   ─ lstm_model.py      # LSTM model for price prediction
─ requirements.txt       # Python dependencies
─ README.md              # Documentation
─ .env                   # Environment variables for API keys
─ model_checkpoint/      # Directory to save trained models

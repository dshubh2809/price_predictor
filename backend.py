from sklearn.model_selection import train_test_split
import numpy as np

def train_model(df, model_save_path="model_checkpoint/lstm_model.h5"):
    """
    Train the LSTM model.
    """
    # Prepare data for LSTM
    X = np.array(df['return']).reshape(-1, 1)
    y = (df['return'] > 0).astype(int)  # Upward = 1, Downward = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Build and train model
    from models.lstm_model import build_lstm_model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model
    model.save(model_save_path)
    return model

def backtest(model, df):
    """
    Backtest the model on historical data.
    """
    X = np.array(df['return']).reshape(-1, 1)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    predictions = model.predict(X)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary
    df['prediction'] = predictions

    # Calculate accuracy
    df['actual'] = (df['return'] > 0).astype(int)
    accuracy = (df['prediction'] == df['actual']).mean()
    return accuracy, df
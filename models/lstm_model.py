import tensorflow as tf

def build_lstm_model(input_shape):
    """
    Build an LSTM model for price prediction.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Predict upward (1) or downward (0)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
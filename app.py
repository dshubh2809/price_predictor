import streamlit as st
from utils import fetch_historical_data, preprocess_data
from backend import train_model, backtest
from tensorflow.keras.models import load_model

st.title("Cryptocurrency Price Predictor")


symbol = st.text_input("Enter trading pair (e.g., BTC/USDT):", "BTC/USDT")
if st.button("Fetch Data"):
    df = fetch_historical_data(symbol)
    st.write("Historical Data:")
    st.write(df)

    
    df = preprocess_data(df)

    
    if st.button("Train Model"):
        model = train_model(df)
        st.success("Model trained and saved!")


        if st.button("Backtest"):
            accuracy, df_with_predictions = backtest(model, df)
            st.write(f"Backtesting Accuracy: {accuracy * 100:.2f}%")
            st.write(df_with_predictions)


st.subheader("Predict Future Prices")
uploaded_model = st.file_uploader("Upload a trained model", type=["h5"])
if uploaded_model:
    model = load_model(uploaded_model)
    st.success("Model loaded successfully!")

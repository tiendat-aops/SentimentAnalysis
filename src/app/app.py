import streamlit as st
# from predict import predict
from explore import explore
from text_p import text_processing
from predict import get_sentiment_prediction


page = st.sidebar.selectbox("Explore Or Predict", ("Explore", "Predict"))


if page == "Predict":
    get_sentiment_prediction()
else:
    explore()





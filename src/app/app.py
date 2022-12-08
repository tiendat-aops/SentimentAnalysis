import streamlit as st
from explore import explore
from predict import get_sentiment_prediction

page = st.sidebar.selectbox("Explore Or Predict", ("Explore", "Predict"))

if page == "Predict":
    get_sentiment_prediction()
else:
    explore()





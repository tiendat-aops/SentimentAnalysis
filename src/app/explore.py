import streamlit as st
import pandas as pd
from predict import preprocessing
from text_p import text_processing
from PIL import Image

pd.set_option("display.max_colwidth", 200)

@st.cache
def load_data(name_of_csv):
    return pd.read_csv(name_of_csv)

def explore():
    st.title("Sentiment Analysis for Movie")
    df = load_data('D:/movie-analysis-streamlit/data/IMDB Dataset.csv')

    placeholder_old = st.empty()
    placeholder = st.empty()

    st.sidebar.header("Dataset Exploration")

    if st.sidebar.button("Negative Review Examples"):
        placeholder_old.markdown(
            "<h3 style='text-align: center; color: white;'>5 Examples of negative reviews</h3>", unsafe_allow_html=True)
        if not st.sidebar.checkbox("Hide dataframe"):
            placeholder.table(df[df['sentiment'] == 'negative'].head(5))

    if st.sidebar.button("Positive Review Examples"):
        placeholder_old.markdown(
            "<h3 style='text-align: center; color: white;'>5 Examples of positive reviews</h3>", unsafe_allow_html=True)
        if not st.sidebar.checkbox("Hide dataframe"):
            placeholder.table(df[df['sentiment'] == 'positive'].head(5))

    image = Image.open('D:/BTL WEB/SentimentAnalysis/images/Negative vs Positive.png')
    st.image(image, caption='Length of Negative Movie Reviews vs Length of Positive Movie Reviews')
    
    with st.expander('See Text Processing'):
        my_slider_val = st.slider('Index of Review Text', 1, 40000)
        st.write("Before Processing")
        st.write(df['review'].iloc[my_slider_val-1])
        if st.checkbox("Process Text Data"):
            processed = text_processing(df['review'].iloc[my_slider_val-1])
            st.write("After Processing :")
            st.write(" ".join(processed))

    with st.expander('Reviews'):
        st.write("""
        #### No. of Positive Reviews : {}
        #### No. of Negative Reviews : {}
        """ .format(df['sentiment'].value_counts()[0], df['sentiment'].value_counts()[1]))

def main():
    explore()

if __name__ == '__main__':
    main()

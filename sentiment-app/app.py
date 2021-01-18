# Streamlit for easy visualization
import streamlit as st

# Pandas for data loading / transformation
import numpy as np
import pandas as pd

# for file download accessory function
import base64

# Transformers for sentiment analysis pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def read_data(file_path):
    '''Separate Function so this is cached and not
    reexecuted by streamlit.
    '''
    return pd.read_csv(file_path).iloc[:100]

def classify(df, target):
    '''Execute `classifier` function on data with target column,
    returning the score
    '''
    results = classifier(list(df[target].values))
    return list(map(lambda result: result['score'], results))

# from user godot23 @ https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806 - thanks!
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download csv file</a>'
    return href

st.title("Sentiment Analysis Demo")

st.write("Sentiment Analysis on your uploaded dataset using the Transformers Library.")

uploaded_file = st.file_uploader("Upload a CSV file - will be truncated to first 100 rows for this demo")

if uploaded_file is not None:
    data = read_data(uploaded_file)

    st.subheader("Data Preview")

    st.dataframe(data.head())

    st.subheader("Sentiment Analysis")

    target = st.selectbox("Choose the Text Column you wish to analyze", list(data.columns))
    
    if st.button("Run"):
        try: 
            with st.spinner():
                results = classify(data, target)
                data['results'] = results
            st.success("Done Classifying!")
            st.balloons()
            st.write('Download Here!')
            st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        except Exception as e:
            print(e)
            st.write("Error! Unable to Analyze. Please ensure your data has strings that can be analyzed!")

st.write("By Branden Ciranni, made possible by Streamlit and Transformers")
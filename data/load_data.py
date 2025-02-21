import pandas as pd
import streamlit as st


# Function to load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    return df

    # Load the dataset
df = load_data()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from src.Model_Comparison.model_comparison import model_comparison_page
from src.Home.home import home_page
from src.About.about import about_page
from src.Data_Insights.data_insight import data_insight_page
from src.Prediction.prediction import prediction_page
import pickle
from train_model import save_model_and_scaler


# Main app
def app():


    # Add navigation sidebar
    # Create sidebar navigation
    st.sidebar.image("https://img.icons8.com/color/100/000000/heart-health.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "Data Insights", "Model Comparison", "About"])

    if page == "Home":
        home_page()
    
    elif page == "Prediction":
        prediction_page()

    elif page == "Data Insights":
        data_insight_page()
    
    elif page == "Model Comparison":
        model_comparison_page()
    
    elif page == "About":
        about_page()

    else:
        st.error("Invalid page selected")


    # Other pages would be handled in the main app file

if __name__ == "__main__":
    app()

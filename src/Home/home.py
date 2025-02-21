import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from data.load_data import df



def home_page():
    st.markdown("<h1 class='main-header'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to the Heart Disease Prediction System</h3>
        <p>This application helps predict the risk of heart disease based on various health parameters. It uses a machine learning model trained on clinical data to provide risk assessments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='sub-header'>What You Can Do Here:</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        - **Get Predictions**: Enter your health metrics to receive a heart disease risk assessment
        - **Explore Data Insights**: View visualizations and statistics about heart disease factors
        - **Learn About Heart Health**: Access information about preventive measures and risk factors
        """)
        
        st.markdown("<h3 class='sub-header'>Key Features:</h3>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            🔍 **Accurate Predictions**
            - Built on clinical data
            - Uses machine learning
            - Instant results
            
            📊 **Data Visualization**
            - Interactive charts
            - Risk factor analysis
            - Demographic insights
            """)
        
        with col_b:
            st.markdown("""
            🏥 **Health Metrics Analysis**
            - Blood pressure impact
            - Cholesterol assessment
            - Age and gender factors
            
            💡 **Educational Content**
            - Risk factor awareness
            - Prevention strategies
            - Health recommendation
            """)
    
    with col2:
        # Create a heart health progress chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 75,
            title = {'text': "Heart Health Index"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [0, 30], 'color': "#ffcccb"},
                    {'range': [30, 70], 'color': "#ffd700"},
                    {'range': [70, 100], 'color': "#90ee90"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #000000; border-radius: 10px;">
            <h4>Start Your Assessment</h4>
            <p>Navigate to the Prediction page to check your heart disease risk.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display some key statistics
    st.markdown("<h3 class='sub-header'>Heart Disease: Key Statistics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Patients in Dataset", value=f"{len(df)}")
    
    with col2:
        disease_count = df['target'].sum()
        st.metric(label="Cases Detected", value=f"{disease_count}")
    
    with col3:
        male_count = df[df['sex'] == 1].shape[0]
        st.metric(label="Male Patients", value=f"{male_count}")
    
    with col4:
        female_count = df[df['sex'] == 0].shape[0]
        st.metric(label="Female Patients", value=f"{female_count}")
    
    # Sample visualization
    st.markdown("<h3 class='sub-header'>Quick Insight: Heart Disease by Age Groups</h3>", unsafe_allow_html=True)
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[25, 35, 45, 55, 65, 80], labels=['25-35', '36-45', '46-55', '56-65', '66+'])
    
    # Create plotly visualization
    fig = px.histogram(df, x='age_group', color='target', 
                       barmode='group', 
                       color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                       labels={'target': 'Heart Disease', 'age_group': 'Age Group', 'count': 'Number of Patients'},
                       title='Heart Disease Distribution by Age Groups')
    
    fig.update_layout(
        legend_title_text='Heart Disease',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update legend labels
    fig.data[0].name = 'No Disease'
    fig.data[1].name = 'Disease'
    
    st.plotly_chart(fig, use_container_width=True)

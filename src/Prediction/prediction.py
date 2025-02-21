import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from train_model import save_model_and_scaler
from data.load_data import df




def prediction_page():
    # Prediction page
    st.markdown("<h1 class='main-header'>Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>Enter your health parameters below to get a prediction. All fields are required for an accurate assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    # Try to load the model and scaler, or create and save them if they don't exist
    try:
        with open('model/heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.info("First-time setup: Training model and saving it for future use...")
        save_model_and_scaler()
        with open('model/heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    # Create three columns for input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.subheader("Demographics")
        age = st.slider("Age", 25, 80, 45)
        sex = st.radio("Sex", ["Female", "Male"])
        sex_value = 1 if sex == "Male" else 0
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.subheader("Blood Work")
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 220)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_value = 1 if fbs == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.subheader("Cardiac Symptoms")
        cp_options = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp = st.selectbox("Chest Pain Type", list(cp_options.keys()))
        cp_value = cp_options[cp]
        
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        exang_value = 1 if exang == "Yes" else 0
        
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
        st.subheader("Other Measurements")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        
        restecg_options = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }
        restecg = st.selectbox("Resting ECG", list(restecg_options.keys()))
        restecg_value = restecg_options[restecg]
        
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
        
        slope_options = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_options.keys()))
        slope_value = slope_options[slope]
        st.markdown("</div>", unsafe_allow_html=True)
        
    # New column for advanced features
    st.markdown("<div class='feature-container'>", unsafe_allow_html=True)
    st.subheader("Advanced Cardiac Measurements")
    col4, col5 = st.columns(2)
    
    with col4:
        ca_options = [0, 1, 2, 3]
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", ca_options)
        
    with col5:
        thal_options = {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3
        }
        thal = st.selectbox("Thalassemia", list(thal_options.keys()))
        thal_value = thal_options[thal]
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a dictionary of input features
    input_data = {
        'age': age,
        'sex': sex_value,
        'cp': cp_value,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_value,
        'restecg': restecg_value,
        'thalach': thalach,
        'exang': exang_value,
        'oldpeak': oldpeak,
        'slope': slope_value,
        'ca': ca,
        'thal': thal_value
    }
    
    # Button to predict
    predict_button = st.button("Predict Heart Disease Risk", type="primary")
    
    if predict_button:
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        # Display results
        st.markdown("<h3 class='sub-header'>Prediction Result</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.markdown(f"<div class='result-danger'><strong>Risk Detected</strong><br>The model predicts you may have a risk of heart disease.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-success'><strong>Low Risk</strong><br>The model predicts you are at a lower risk of heart disease.</div>", unsafe_allow_html=True)
        
        with col2:
            # Create gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[0][1] * 100,
                title = {'text': "Risk Probability"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if probability[0][1] > 0.5 else "#4CAF50"},
                    'steps': [
                        {'range': [0, 25], 'color': "#4CAF50"},
                        {'range': [25, 50], 'color': "#FFC107"},
                        {'range': [50, 75], 'color': "#FF9800"},
                        {'range': [75, 100], 'color': "#F44336"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors analysis
        st.markdown("<h3 class='sub-header'>Risk Factor Analysis</h3>", unsafe_allow_html=True)
        
        # Get feature importance
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': np.abs(coefficients),
                'Value': [input_data[feat] for feat in input_df.columns],
                'Direction': ['Increases Risk' if coef > 0 else 'Decreases Risk' for coef in coefficients]
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Display top risk factors
            top_factors = feature_importance.head(5)
            
            fig = px.bar(
                top_factors,
                y='Feature',
                x='Importance',
                color='Direction',
                orientation='h',
                color_discrete_map={
                    'Increases Risk': '#F44336',
                    'Decreases Risk': '#4CAF50'
                },
                title='Your Top 5 Risk Factors'
            )
            
            fig.update_layout(
                xaxis_title="Relative Importance",
                yaxis_title="",
                legend_title="Impact",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on top risk factors
            st.markdown("<h3 class='sub-header'>Health Recommendations</h3>", unsafe_allow_html=True)
            
            recommendations = {
                'age': "Age is a non-modifiable risk factor. Regular check-ups become increasingly important as you age.",
                'sex': "Gender is a non-modifiable risk factor. Be aware of gender-specific risk patterns.",
                'cp': "The type of chest pain you experience can be significant. Report any chest discomfort to your doctor.",
                'trestbps': "Work on maintaining healthy blood pressure through diet, exercise, and possibly medication if prescribed.",
                'chol': "Aim to keep cholesterol levels in check through diet, exercise, and medication if prescribed.",
                'fbs': "Managing blood sugar levels is crucial. Follow a balanced diet and consider regular diabetes screening.",
                'restecg': "Abnormal ECG results should be followed up with your cardiologist.",
                'thalach': "Regular cardiovascular exercise can help improve your maximum heart rate capacity.",
                'exang': "Exercise-induced chest pain should be discussed with your doctor immediately.",
                'oldpeak': "ST depression noted during exercise stress tests should be evaluated by a specialist.",
                'slope': "The slope of the ST segment during exercise provides important diagnostic information.",
                'ca': "The number of major vessels with blockage is a significant indicator. Consider follow-up tests if recommended.",
                'thal': "Thalassemia-related issues should be monitored by your healthcare provider."
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Based on Your Results:")
                for factor in top_factors['Feature'][:3]:
                    st.markdown(f"**{factor}**: {recommendations[factor]}")
            
            with col2:
                st.subheader("General Advice:")
                st.markdown("""
                - 🏃‍♂️ Engage in regular physical activity (150+ minutes/week)
                - 🥗 Maintain a heart-healthy diet (low sodium, low fat)
                - 🚭 Avoid smoking and limit alcohol consumption
                - 😴 Prioritize quality sleep (7-8 hours nightly)
                - 🩺 Schedule regular check-ups with your physician
                """)

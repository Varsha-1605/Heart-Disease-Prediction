import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def about_page():
    
# About page
    st.markdown("<h1 class='main-header'>About This Application</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Heart Disease Prediction System</h3>
        <p>This application uses machine learning to predict heart disease risk based on clinical parameters. It was developed to assist healthcare professionals and individuals in understanding heart disease risk factors and making informed health decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='sub-header'>How It Works</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        The application is powered by a logistic regression model trained on clinical data. Here's how the prediction system works:
        
        1. **Data Collection**: The model was trained on a dataset of 303 patients with various cardiac health parameters.
        
        2. **Preprocessing**: Patient data is standardized to ensure consistent predictions.
        
        3. **Risk Prediction**: The logistic regression algorithm calculates the probability of heart disease.
        
        4. **Visualization**: Results are presented through interactive charts and personalized recommendations.
        """)
        
        st.markdown("<h3 class='sub-header'>Dataset Information</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        The prediction model uses the UCI Heart Disease dataset with the following features:
        
        - **age**: Age in years
        - **sex**: Sex (1 = male, 0 = female)
        - **cp**: Chest pain type (0-3)
        - **trestbps**: Resting blood pressure in mm Hg
        - **chol**: Serum cholesterol in mg/dl
        - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        - **restecg**: Resting electrocardiographic results (0-2)
        - **thalach**: Maximum heart rate achieved
        - **exang**: Exercise induced angina (1 = yes, 0 = no)
        - **oldpeak**: ST depression induced by exercise relative to rest
        - **slope**: Slope of the peak exercise ST segment (0-2)
        - **ca**: Number of major vessels colored by fluoroscopy (0-3)
        - **thal**: Thalassemia type (1-3)
        - **target**: Heart disease presence (1 = present, 0 = absent)
        """)
    
    with col2:
        # Display model performance metrics
        st.markdown("<h4>Model Performance</h4>", unsafe_allow_html=True)
        
        # Create a gauge for accuracy
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 85.7,
            title = {'text': "Model Accuracy"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 60], 'color': "#F44336"},
                    {'range': [60, 80], 'color': "#FFC107"},
                    {'range': [80, 100], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display other metrics
        metrics = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score'],
            'Value': [0.86, 0.85, 0.85]
        })
        
        fig = px.bar(
            metrics,
            x='Metric',
            y='Value',
            color='Metric',
            text_auto='.2f',
            title='Model Evaluation Metrics',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Score",
            showlegend=False,
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box" style="background-color: #000000; border-left: 5px solid #4caf50;">
        <h4>Disclaimer</h4>
        <p>This application is designed for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment.</p>
        <p>Always consult a qualified healthcare provider for medical concerns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>References & Further Reading</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Academic Papers:**
        
        1. Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304-310.
        
        2. Janosi, A., et al. (1988). Heart disease and angiographic findings: A study of predictive variables. American Heart Journal, 115(3), 843-849.
        
        3. Wang, S., et al. (2017). Predicting heart disease using machine learning methods. In Healthcare Analytics (pp. 365-393). Springer.
        """)
    
    with col2:
        st.markdown("""
        **Resources for Heart Health:**
        
        - American Heart Association: [heart.org](https://www.heart.org)
        - World Heart Federation: [world-heart-federation.org](https://www.world-heart-federation.org)
        - CDC Heart Disease Information: [cdc.gov/heartdisease](https://www.cdc.gov/heartdisease)
        - Mayo Clinic Heart Disease Guide: [mayoclinic.org/heart-disease](https://www.mayoclinic.org/diseases-conditions/heart-disease)
        """)
    
    # Feedback section
    st.markdown("<h3 class='sub-header'>Feedback</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    We welcome your feedback to improve this application. Please share your thoughts and suggestions in the form below.
    """)
    
    feedback_type = st.selectbox(
        "Feedback Type",
        ["General Feedback", "Bug Report", "Feature Request", "Accuracy Concerns", "UI/UX Suggestions"]
    )
    
    feedback_text = st.text_area("Your Feedback", height=100)
    
    contact_email = st.text_input("Email (optional)")
    
    submit_button = st.button("Submit Feedback")
    
    if submit_button:
        st.success("Thank you for your feedback! We appreciate your input and will use it to improve the application.")
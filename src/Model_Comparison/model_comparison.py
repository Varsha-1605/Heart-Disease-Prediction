import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def model_comparison_page():
    st.title("Model Comparison and Selection")
    
    st.markdown("""
    ## 🔍 Comparing Machine Learning Models for Heart Disease Prediction
    
    We evaluated multiple classification models to identify the most effective approach for heart disease prediction.
    Each model was assessed based on accuracy, precision, recall, and F1-score metrics.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "📈 Visualizations", "🏆 Conclusion"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Model comparison data
        model_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'KNN', 
                     'XGBoost', 'Gradient Boosting', 'LightGBM', 'CatBoost', 'Extra Trees'],
            'Accuracy': [0.78, 0.70, 0.74, 0.79, 0.79, 0.76, 0.70, 0.78, 0.72, 0.73],
            'Precision': [0.79, 0.75, 0.75, 0.75, 0.75, 0.79, 0.72, 0.78, 0.74, 0.77],
            'Recall': [0.82, 0.68, 0.68, 0.68, 0.68, 0.78, 0.74, 0.82, 0.76, 0.74],
            'F1-score': [0.80, 0.71, 0.71, 0.71, 0.71, 0.78, 0.73, 0.80, 0.75, 0.75]
        }
        
        df_models = pd.DataFrame(model_data)
        
        # Display models in DataFrame format
        st.dataframe(df_models.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-score']),
                   use_container_width=True)
        
        # Add explanation about metrics
        with st.expander("📖 Understanding the Metrics"):
            st.markdown("""
            ### Key Performance Metrics Explained
            
            - **Accuracy**: Overall correctness of the model (correct predictions / total predictions)
            - **Precision**: Ability to identify only relevant instances (true positives / (true positives + false positives))
            - **Recall**: Ability to find all relevant instances (true positives / (true positives + false negatives))
            - **F1-score**: Harmonic mean of precision and recall, providing a balanced metric
            
            **In the context of heart disease prediction:**
            - High recall is particularly important to minimize missed cases of heart disease
            - Precision helps minimize unnecessary anxiety or follow-up procedures
            """)
    
    with tab2:
        st.subheader("Performance Visualizations")
        
        # Extract data for visualization
        models = model_data['Model']
        accuracy = model_data['Accuracy']
        precision = model_data['Precision']
        recall = model_data['Recall']
        f1_score = model_data['F1-score']
        
        # Visualization type selector
        viz_type = st.radio(
            "Select visualization type:",
            ["Bar Chart Comparison", "Radar Chart", "Heatmap", "3D Performance Plot"]
        )
        
        if viz_type == "Bar Chart Comparison":
            # Create a Plotly bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(x=models, y=accuracy, name='Accuracy', marker_color='royalblue'))
            fig.add_trace(go.Bar(x=models, y=precision, name='Precision', marker_color='lightseagreen'))
            fig.add_trace(go.Bar(x=models, y=recall, name='Recall', marker_color='indianred'))
            fig.add_trace(go.Bar(x=models, y=f1_score, name='F1-score', marker_color='mediumpurple'))
            
            fig.update_layout(
                title='Model Performance Metrics Comparison',
                xaxis_title='Models',
                yaxis_title='Score',
                yaxis=dict(range=[0.65, 0.85]),
                barmode='group',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Radar Chart":
            # Create radar chart for top 5 models
            top_models = ['Logistic Regression', 'SVC', 'XGBoost', 'LightGBM', 'KNN']
            
            # Filter data for top models
            top_df = df_models[df_models['Model'].isin(top_models)]
            
            # Create radar chart
            fig = go.Figure()
            
            for index, row in top_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-score']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-score'],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.65, 0.85]
                    )),
                showlegend=True,
                title="Top 5 Models Performance Radar Comparison",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Heatmap":
            # Create heatmap DataFrame
            heatmap_df = df_models.set_index('Model')
            
            # Generate heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
            plt.title("Model Performance Heatmap", fontsize=14)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        elif viz_type == "3D Performance Plot":
            # Create 3D scatter plot
            fig = px.scatter_3d(
                df_models, x='Precision', y='Recall', z='Accuracy',
                color='F1-score', hover_name='Model', size='F1-score',
                color_continuous_scale='Viridis',
                size_max=15, opacity=0.7
            )
            
            fig.update_layout(
                title='3D Model Performance Analysis',
                height=700,
                scene=dict(
                    xaxis_title='Precision',
                    yaxis_title='Recall',
                    zaxis_title='Accuracy',
                    xaxis=dict(range=[0.7, 0.8]),
                    yaxis=dict(range=[0.65, 0.85]),
                    zaxis=dict(range=[0.7, 0.8])
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🏆 Final Model Selection & Conclusions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Best Model: Logistic Regression
            
            After evaluating multiple models based on their performance metrics, we selected **Logistic Regression** as the optimal model for heart disease prediction.
            
            ### Key Factors in Selection:
            - **Strong Overall Performance**: High accuracy (0.78) and F1-score (0.80)
            - **Superior Recall (0.82)**: Critical for medical applications to minimize missed heart disease cases
            - **Interpretability**: Provides clear feature importance insights that can be validated by medical professionals
            - **Efficiency**: Fast training and inference times, suitable for real-time predictions
            
            ### Clinical Significance:
            - Higher recall prioritizes patient safety by minimizing false negatives
            - Good precision (0.79) helps reduce unnecessary follow-up procedures
            - The model balances sensitivity and specificity effectively for clinical decision support
            
            ### Alternative Models:
            - **Support Vector Classifier (SVC)**: Highest accuracy (0.79) but lower recall
            - **LightGBM**: Matches Logistic Regression's recall but slightly more complex
            - **XGBoost**: Good balance of metrics but more computationally intensive
            """)
        
        with col2:
            st.image("https://storage.openvisualization.dev:443/public/default-images/logistic-regression-heart-disease.webp", 
                   caption="Logistic Regression Decision Boundary (Representative Image)")
            
            # Create accuracy comparison chart for top models
            top_models = ['Logistic Regression', 'SVC', 'LightGBM', 'XGBoost', 'KNN']
            top_accuracy = [0.78, 0.79, 0.78, 0.76, 0.79]
            top_recall = [0.82, 0.68, 0.82, 0.78, 0.68]
            
            comparison_df = pd.DataFrame({
                'Model': top_models,
                'Accuracy': top_accuracy,
                'Recall': top_recall
            })
            
            fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'Recall'],
                        title="Top Models: Accuracy vs Recall",
                        barmode='group',
                        color_discrete_sequence=['royalblue', 'crimson'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model deployment considerations
        st.markdown("""
        ## Deployment Considerations
        
        Our Logistic Regression model has been integrated into the prediction interface with the following implementation details:
        
        1. **Feature Standardization**: All numerical features are scaled to improve model performance
        2. **Threshold Optimization**: We've calibrated the prediction threshold to prioritize recall
        3. **Regular Retraining**: The model will be periodically retrained as new data becomes available
        4. **Interpretability Layer**: We provide feature importance and prediction explanations in the user interface
        
        ### Future Improvements:
        - Ensemble model combining Logistic Regression with SVC for potentially higher performance
        - Implementation of post-processing calibration for better probability estimates
        - Investigation of neural network approaches as dataset size increases
        """)
        
        # Add a note about the features used by the model
        st.info("""
        **Features Used by the Selected Model:**
        - age: Age in years
        - sex: Gender (1 = male, 0 = female)
        - cp: Chest pain type (0-3)
        - trestbps: Resting blood pressure (mm Hg)
        - chol: Serum cholesterol (mg/dl)
        - fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        - restecg: Resting electrocardiographic results (0-2)
        - thalach: Maximum heart rate achieved
        - exang: Exercise induced angina (1 = yes, 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: Slope of peak exercise ST segment (0-2)
        - ca: Number of major vessels colored by fluoroscopy (0-4)
        - thal: Thalassemia (0-3)
        """)


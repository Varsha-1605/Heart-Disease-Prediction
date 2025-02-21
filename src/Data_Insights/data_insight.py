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


def data_insight_page():

    st.markdown("<h1 class='main-header'>Heart Disease Data Insights</h1>", unsafe_allow_html=True)
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Key Indicators", "Correlations", "Advanced Analysis"])
    
    with tab1:
        st.markdown("<h3 class='sub-header'>Patient Demographics</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            fig = px.pie(
                df, 
                names=df['sex'].map({0: 'Female', 1: 'Male'}),
                title='Gender Distribution in Dataset',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age distribution by heart disease
            fig = px.histogram(
                df,
                x='age',
                color='target',
                nbins=20,
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'age': 'Age', 'count': 'Number of Patients'},
                title='Age Distribution by Heart Disease Status'
            )
            
            fig.data[0].name = 'No Disease'
            fig.data[1].name = 'Disease'
            
            fig.update_layout(barmode='overlay', bargap=0.1)
            fig.update_traces(opacity=0.7)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Gender vs. Heart Disease Analysis
        st.markdown("<h4>Heart Disease by Gender</h4>", unsafe_allow_html=True)
        
        # Create the crosstab
        cross_tab = pd.crosstab(df['sex'], df['target'], margins=True, margins_name='Total')
        cross_tab_df = cross_tab.reset_index().melt(id_vars='sex', var_name='Heart Disease', value_name='Count')
        cross_tab_df['sex'] = cross_tab_df['sex'].map({0: 'Female', 1: 'Male', 'Total': 'Total'})
        cross_tab_df['Heart Disease'] = cross_tab_df['Heart Disease'].map({0: 'Not Diseased', 1: 'Diseased', 'Total': 'Total'})
        
        # Filter out the totals for the bar chart
        plot_df = cross_tab_df[(cross_tab_df['sex'] != 'Total') & (cross_tab_df['Heart Disease'] != 'Total')]
        
        fig = px.bar(
            plot_df, 
            x='sex', 
            y='Count', 
            color='Heart Disease', 
            barmode='group',
            text='Count',
            color_discrete_map={'Not Diseased': '#1f77b4', 'Diseased': '#ff7f0e'},
            title='Heart Disease Distribution by Sex'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Gender",
            yaxis_title="Number of Patients",
            legend_title="Heart Disease Status"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>Key Demographics Insights:</h4>
            <ul>
                <li>The dataset contains more male patients (207) than female patients (96)</li>
                <li>Among women, 72 out of 96 (75%) have heart disease</li>
                <li>Among men, 93 out of 207 (45%) have heart disease</li>
                <li>Heart disease appears across all age groups but peaks in the 55-65 range</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box" style="background-color: #000000; border-left: 5px solid #ff9800;">
            <h4>Clinical Implications:</h4>
            <p>While men are more represented in this dataset, women show a significantly higher percentage of heart disease cases. This suggests that gender-specific risk assessment strategies may be beneficial.</p>
            <p>The age distribution shows heart disease risk increases with age, particularly after 45 years.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 class='sub-header'>Key Health Indicators</h3>", unsafe_allow_html=True)
        
        indicator = st.selectbox(
            "Select Health Indicator to Analyze",
            ["Cholesterol Levels", "Resting Blood Pressure", "Maximum Heart Rate"]
        )
        
        # if indicator == "Chest Pain Type":
        #     # Map chest pain types
        #     cp_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        #     df['cp_name'] = df['cp'].map(cp_map)
            
        #     fig = px.bar(
        #         df, 
        #         x='cp_name',
        #         color='target',
        #         barmode='group',
        #         color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
        #         labels={'cp_name': 'Chest Pain Type', 'count': 'Number of Patients', 'target': 'Heart Disease'},
        #         title='Heart Disease by Chest Pain Type'
        #     )
            
        #     fig.data[0].name = 'No Disease'
        #     fig.data[1].name = 'Disease'
            
        #     st.plotly_chart(fig, use_container_width=True)
            
        #     st.markdown("""
        #     <div class="info-box">
        #     <h4>Chest Pain Analysis:</h4>
        #     <p>Asymptomatic chest pain (type 3) shows the highest correlation with heart disease, despite patients not reporting typical symptoms. This highlights the importance of routine screenings even in the absence of obvious symptoms.</p>
        #     </div>
        #     """, unsafe_allow_html=True)
            
        if indicator == "Cholesterol Levels":
            fig = px.box(
                df,
                x='target',
                y='chol',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'chol': 'Cholesterol Level (mg/dl)'},
                title='Cholesterol Levels vs. Heart Disease',
                points="all"
            )
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['No Disease', 'Disease']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a histogram with density curve
            fig = px.histogram(
                df, 
                x='chol',
                color='target',
                marginal='rug',
                opacity=0.7,
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'chol': 'Cholesterol Level (mg/dl)'},
                title='Distribution of Cholesterol Levels'
            )
            
            fig.data[0].name = 'No Disease'
            fig.data[1].name = 'Disease'
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Cholesterol Analysis:</h4>
            <p>Contrary to what might be expected, there isn't a strong visual correlation between cholesterol levels and heart disease in this dataset. Heart disease cases are distributed across various cholesterol levels, suggesting that cholesterol alone may not be a definitive predictor without considering other factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif indicator == "Resting Blood Pressure":
            fig = px.box(
                df,
                x='target',
                y='trestbps',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'trestbps': 'Resting Blood Pressure (mm Hg)'},
                title='Resting Blood Pressure vs. Heart Disease',
                points="all"
            )
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['No Disease', 'Disease']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add scatter plot with density
            fig = px.scatter(
                df,
                x='age',
                y='trestbps',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                opacity=0.7,
                labels={'target': 'Heart Disease', 'trestbps': 'Resting Blood Pressure (mm Hg)', 'age': 'Age'},
                title='Blood Pressure by Age and Heart Disease Status',
                marginal_x='histogram',
                marginal_y='histogram'
            )
            
            fig.data[0].name = 'No Disease'
            fig.data[1].name = 'Disease'
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Blood Pressure Analysis:</h4>
            <p>Resting blood pressure shows some correlation with heart disease, with higher median values observed in patients with heart disease. However, there is significant overlap between both groups, suggesting blood pressure should be considered alongside other risk factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif indicator == "Maximum Heart Rate":
            fig = px.box(
                df,
                x='target',
                y='thalach',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'thalach': 'Maximum Heart Rate (bpm)'},
                title='Maximum Heart Rate vs. Heart Disease',
                points="all"
            )
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['No Disease', 'Disease']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add heart rate vs age scatter plot
            fig = px.scatter(
                df,
                x='age',
                y='thalach',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'thalach': 'Maximum Heart Rate (bpm)', 'age': 'Age'},
                title='Maximum Heart Rate vs. Age',
                trendline='ols'
            )
            
            fig.data[0].name = 'No Disease'
            fig.data[1].name = 'Disease'
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Heart Rate Analysis:</h4>
            <p>There's a notable inverse relationship between maximum heart rate and heart disease. Patients with heart disease tend to have lower maximum heart rates. This is an important clinical finding as decreased heart rate response to exercise (chronotropic incompetence) can be a sign of underlying heart disease.</p>
            <p>The age-related decline in maximum heart rate is steeper in individuals with heart disease, suggesting compounded cardiovascular deterioration.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h3 class='sub-header'>Feature Correlations</h3>", unsafe_allow_html=True)
        
        # Create a copy of the dataframe without categorical columns
        df_corr = df.copy()
        
        # Remove any categorical or non-numeric columns before correlation
        # Explicitly list numeric columns or drop known categorical ones
        numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
        df_corr = df_corr[numeric_columns]
        
        # Now calculate correlation
        corr = df_corr.corr()
        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Heatmap',
            aspect="auto"
        )
        
        fig.update_layout(
            height=700,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Important correlations with target
        target_corr = corr['target'].sort_values(ascending=False)
        target_corr = target_corr.drop('target')
        
        fig = px.bar(
            x=target_corr.index,
            y=target_corr.values,
            title='Feature Correlation with Heart Disease',
            labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
            color=target_corr.values,
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Correlation with Heart Disease"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>Key Correlation Insights:</h4>
        <ul>
            <li><strong>Positive correlations with heart disease:</strong> cp (chest pain type), thalach (maximum heart rate), slope (ST segment slope)</li>
            <li><strong>Negative correlations with heart disease:</strong> oldpeak (ST depression), ca (number of major vessels), thal (thalassemia type), exang (exercise-induced angina)</li>
            <li>The strongest predictor appears to be the number of major vessels colored by fluoroscopy (ca), which has a strong negative correlation with heart disease risk.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Pairplot selections
        st.subheader("Explore Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis Feature", df.columns[:-2])
        
        with col2:
            y_feature = st.selectbox("Select Y-axis Feature", df.columns[:-2], index=1)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color='target',
            color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
            title=f'{y_feature} vs {x_feature} by Heart Disease Status',
            labels={'target': 'Heart Disease'},
            opacity=0.7,
            trendline='ols'
        )
        
        fig.data[0].name = 'No Disease'
        fig.data[1].name = 'Disease'
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("<h3 class='sub-header'>Advanced Analysis</h3>", unsafe_allow_html=True)
        
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Feature Importance", "Risk Profile Analysis", "Age-Gender Interaction"]
        )
        
                # Fix for Feature Importance section
        if analysis_type == "Feature Importance":
        # Calculate feature importance using mutual information
        from sklearn.feature_selection import mutual_info_classif
        import numpy as np
        
        # Create a clean copy of the dataframe for analysis
        X_clean = df.drop(columns=["target"]).copy()
        y = df["target"].copy()
        
        # Ensure all columns are numeric and handle NaN values
        for col in X_clean.columns:
            # Convert to numeric and fill NaN with column mean
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            if X_clean[col].isna().any():
                # Replace NaN with mean of column
                X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
        
        # Convert to numpy array with explicit dtype
        X_array = X_clean.to_numpy(dtype=np.float64)
        
        try:
            # Compute Mutual Information Scores
            mi_scores = mutual_info_classif(X_array, y)
            mi_df = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': mi_scores
            }).sort_values('Importance', ascending=False)
            
            # Plot
            fig = px.bar(
                mi_df,
                y='Feature',
                x='Importance',
                orientation='h',
                title="Feature Importance (Mutual Information)",
                color='Importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                xaxis_title="Mutual Information Score",
                yaxis_title="",
                yaxis={'categoryorder':'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Feature Importance Analysis:</h4>
            <p>Mutual information measures how much information a feature provides about the target variable (heart disease). Key findings:</p>
            <ul>
                <li><strong>ca (number of major vessels):</strong> The most informative feature, indicating that coronary angiography results are crucial for diagnosis</li>
                <li><strong>thal (thalassemia):</strong> The second most important feature, showing that blood disorders play a significant role</li>
                <li><strong>cp (chest pain type):</strong> The nature of chest pain provides substantial diagnostic value</li>
                <li><strong>oldpeak (ST depression):</strong> ST segment depression during exercise is a strong indicator</li>
            </ul>
            <p>These results highlight the importance of cardiac-specific tests rather than general health metrics like cholesterol or blood pressure for accurate heart disease diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_type == "Risk Profile Analysis":
            # Create age groups
            df['age_group'] = pd.cut(df['age'], bins=[25, 40, 55, 70, 85], labels=['25-40', '41-55', '56-70', '71+'])
            
            # Calculate risk profiles
            risk_profile = df.groupby(['age_group', 'sex'])['target'].mean().reset_index()
            risk_profile['target'] = risk_profile['target'] * 100
            risk_profile['sex'] = risk_profile['sex'].map({0: 'Female', 1: 'Male'})
            
            # Plot risk profiles
            fig = px.bar(
                risk_profile,
                x='age_group',
                y='target',
                color='sex',
                barmode='group',
                title='Heart Disease Risk Profile by Age Group and Gender',
                labels={'target': 'Risk Percentage (%)', 'age_group': 'Age Group', 'sex': 'Gender'},
                text_auto='.1f'
            )
            
            fig.update_layout(
                xaxis_title="Age Group",
                yaxis_title="Percentage with Heart Disease (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate risk by multiple factors
            df['high_bp'] = df['trestbps'] > 140
            df['high_chol'] = df['chol'] > 240
            
            # Advanced risk analysis
            col1, col2 = st.columns(2)
            
            with col1:
                risk_factor = st.selectbox(
                    "Select Additional Risk Factor",
                    ["Chest Pain Type", "High Blood Pressure", "High Cholesterol", "Diabetes (FBS)"]
                )
            
            with col2:
                view_by = st.selectbox(
                    "View By",
                    ["Age Group", "Gender"]
                )
            
            if risk_factor == "Chest Pain Type":
                factor_col = 'cp'
                factor_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
                df['factor'] = df[factor_col].map(factor_map)
                factor_name = "Chest Pain Type"
            elif risk_factor == "High Blood Pressure":
                df['factor'] = df['high_bp'].map({True: 'High BP (>140)', False: 'Normal BP'})
                factor_name = "Blood Pressure Status"
            elif risk_factor == "High Cholesterol":
                df['factor'] = df['high_chol'].map({True: 'High Chol (>240)', False: 'Normal Chol'})
                factor_name = "Cholesterol Status"
            else:
                df['factor'] = df['fbs'].map({1: 'Diabetes', 0: 'No Diabetes'})
                factor_name = "Diabetes Status"
            
            if view_by == "Age Group":
                group_col = 'age_group'
                group_name = "Age Group"
            else:
                group_col = 'sex'
                df[group_col] = df[group_col].map({0: 'Female', 1: 'Male'})
                group_name = "Gender"
            
            # Calculate the advanced risk profile
            adv_risk = df.groupby([group_col, 'factor'])['target'].mean().reset_index()
            adv_risk['target'] = adv_risk['target'] * 100
            
            # Plot advanced risk profile
            fig = px.bar(
                adv_risk,
                x=group_col,
                y='target',
                color='factor',
                barmode='group',
                title=f'Heart Disease Risk by {group_name} and {factor_name}',
                labels={'target': 'Risk Percentage (%)', group_col: group_name, 'factor': factor_name},
                text_auto='.1f'
            )
            
            fig.update_layout(
                xaxis_title=group_name,
                yaxis_title="Percentage with Heart Disease (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Risk Profile Insights:</h4>
            <p>The analysis reveals complex interactions between demographic factors and heart disease risk:</p>
            <ul>
                <li>Women generally show higher heart disease rates than men in the same age groups</li>
                <li>Risk increases with age for both genders, but the rate of increase differs</li>
                <li>Asymptomatic chest pain carries the highest risk across age groups</li>
                <li>The combination of age, gender, and specific clinical factors creates unique risk profiles</li>
            </ul>
            <p>These findings highlight the importance of personalized risk assessment that takes into account multiple factors simultaneously.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_type == "Age-Gender Interaction":
            # Create 3D plot of age, max heart rate, and heart disease
            fig = px.scatter_3d(
                df,
                x='age',
                y='thalach',
                z='chol',
                color='target',
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                opacity=0.7,
                labels={'target': 'Heart Disease', 'thalach': 'Maximum Heart Rate', 'age': 'Age', 'chol': 'Cholesterol'},
                title='3D Visualization: Age, Heart Rate, and Cholesterol'
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Age',
                    yaxis_title='Maximum Heart Rate',
                    zaxis_title='Cholesterol'
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Age distribution by gender and heart disease
            fig = px.histogram(
                df,
                x='age',
                color='target',
                facet_col='sex',
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={0: '#1E88E5', 1: '#FF5252'},
                labels={'target': 'Heart Disease', 'age': 'Age', 'sex': 'Gender'},
                category_orders={"sex": [0, 1]},
                title='Age Distribution by Gender and Heart Disease Status'
            )
            
            fig.data[0].name = 'No Disease'
            fig.data[1].name = 'Disease'
            fig.data[2].name = 'No Disease'
            fig.data[3].name = 'Disease'
            
            # Update facet labels
            fig.for_each_annotation(lambda a: a.update(text=a.text.replace("sex=0", "Female").replace("sex=1", "Male")))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <h4>Age-Gender Interaction Analysis:</h4>
            <p>The 3D visualization and age distribution charts reveal important patterns:</p>
            <ul>
                <li>Women tend to develop heart disease at older ages compared to men</li>
                <li>Maximum heart rate shows a stronger relationship with heart disease than cholesterol</li>
                <li>The age distribution curves differ significantly between genders</li>
                <li>For women, heart disease cases peak in the 55-65 age range, while for men the distribution is broader</li>
            </ul>
            <p>These gender differences highlight the need for gender-specific screening protocols and risk assessment models.</p>
            </div>
            """, unsafe_allow_html=True)

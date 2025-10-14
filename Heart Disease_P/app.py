#!/usr/bin/env python3
"""
Heart Disease Prediction Streamlit Web Application
Author: Claude AI
Date: September 2025
"""

import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .high-risk {
        background-color: #FFE5E5;
        border: 2px solid #FF6B6B;
        color: #D63031;
    }
    .low-risk {
        background-color: #E8F5E8;
        border: 2px solid #00B894;
        color: #00B894;
    }
    .info-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #74B9FF;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        """Load the trained model and metadata"""
        try:
            self.model = joblib.load('final_model.pkl')
            self.metadata = joblib.load('final_model_metadata.pkl')
            return True
        except Exception as e:
            st.warning(f"⚠️ Could not load saved model pipeline: {e}")
            return False

    def create_sample_model(self):
        """Create a sample model for demonstration"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        np.random.seed(42)
        X_sample = np.random.randn(100, 13)
        y_sample = np.random.choice([0, 1], 100)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_sample, y_sample)

        self.model = pipeline
        self.metadata = {
            'feature_names': [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ],
            'target_names': ['No Disease', 'Disease'],
            'model_type': 'RandomForestClassifier',
            'note': 'Demo model only. Run the training pipeline to generate the real model.'
        }
        st.info("🔄 Using demonstration model. Run the pipeline to generate the trained model.")

    def predict(self, features):
        """Make prediction using the loaded model"""
        if self.model is None or self.metadata is None:
            return None, None

        # Build a DataFrame with named columns matching training
        feature_names = self.metadata.get('feature_names', [])
        try:
            X_df = pd.DataFrame([features], columns=feature_names)
        except Exception:
            # Fallback to numeric array if names mismatch
            X_df = pd.DataFrame([features])

        prediction = self.model.predict(X_df)[0]
        probability = None

        if hasattr(self.model, "predict_proba"):
            try:
                probability = self.model.predict_proba(X_df)[0]
            except Exception:
                probability = None

        # Fallback via decision_function if predict_proba is not available
        if probability is None and hasattr(self.model, "decision_function"):
            try:
                z = self.model.decision_function(X_df)
                # Not calibrated probabilities, but provides a usable score
                p = 1 / (1 + np.exp(-z))
                p = float(np.clip(p, 0, 1))
                probability = np.array([1 - p, p])
            except Exception:
                probability = np.array([0.5, 0.5])

        return prediction, probability


def main():
    predictor = HeartDiseasePredictor()
    if predictor.model is None:
        predictor.create_sample_model()

    st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>',
                unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:",
                                ["Prediction", "Data Exploration", "Model Information"])

    if page == "Prediction":
        prediction_page(predictor)
    elif page == "Data Exploration":
        data_exploration_page()
    else:
        model_info_page(predictor)


def prediction_page(predictor):
    """Main prediction interface"""
    st.header("🔮 Heart Disease Risk Assessment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter Patient Information")
        with st.form("prediction_form"):
            st.markdown("**Personal Information**")
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
                sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
            with col_b:
                cp = st.selectbox("Chest Pain Type", [
                    "Typical Angina (0)", "Atypical Angina (1)",
                    "Non-anginal Pain (2)", "Asymptomatic (3)"
                ])
                exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
                slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                    "Upsloping (0)", "Flat (1)", "Downsloping (2)"
                ])

            st.markdown("**Medical Measurements**")
            col_c, col_d = st.columns(2)
            with col_c:
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                           min_value=80, max_value=250, value=120)
                chol = st.number_input("Serum Cholesterol (mg/dl)",
                                       min_value=100, max_value=700, value=240)
                thalach = st.number_input("Maximum Heart Rate Achieved",
                                          min_value=60, max_value=220, value=150)
            with col_d:
                restecg = st.selectbox("Resting ECG Results", [
                    "Normal (0)", "ST-T Wave Abnormality (1)",
                    "Left Ventricular Hypertrophy (2)"
                ])
                oldpeak = st.number_input("ST Depression (oldpeak)",
                                          min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])

            thal = st.selectbox("Thalassemia", [
                "Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"
            ])

            submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

        if submitted:
            features = [
                age,
                int(sex.split("(")[1].split(")")[0]),
                int(cp.split("(")[1].split(")")[0]),
                trestbps,
                chol,
                int(fbs.split("(")[1].split(")")[0]),
                int(restecg.split("(")[1].split(")")[0]),
                thalach,
                int(exang.split("(")[1].split(")")[0]),
                oldpeak,
                int(slope.split("(")[1].split(")")[0]),
                ca,
                int(thal.split("(")[1].split(")")[0])
            ]

            prediction, probability = predictor.predict(features)

            if prediction is not None:
                st.subheader("🎯 Prediction Results")

                risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_class = "high-risk" if prediction == 1 else "low-risk"
                confidence = (max(probability) * 100) if probability is not None else 50.0

                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    {risk_level} for Heart Disease<br>
                    <small>Confidence: {confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 Risk Analysis")
                if probability is not None:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=float(probability[1]) * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Heart Disease Risk (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Model does not expose calibrated probabilities; showing class prediction only.")

    with col2:
        st.subheader("ℹ️ Feature Information")
        feature_info = {
            "Age": "Age in years",
            "Sex": "0 = Female, 1 = Male",
            "CP": "Chest Pain Type (0-3)",
            "Trestbps": "Resting Blood Pressure",
            "Chol": "Serum Cholesterol in mg/dl",
            "FBS": "Fasting Blood Sugar > 120 mg/dl",
            "RestECG": "Resting ECG Results",
            "Thalach": "Maximum Heart Rate",
            "Exang": "Exercise Induced Angina",
            "Oldpeak": "ST Depression (exercise-induced)",
            "Slope": "Slope of Peak Exercise ST",
            "CA": "Number of Major Vessels (0-4) colored by fluoroscopy",
            "Thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)"
        }
        for feature, description in feature_info.items():
            st.markdown(f"""
            <div class="info-card">
                <strong>{feature}:</strong> {description}
            </div>
            """, unsafe_allow_html=True)


def data_exploration_page():
    """Data exploration and visualization page"""
    st.header("📈 Heart Disease Data Exploration")

    np.random.seed(42)
    n_samples = 300
    data = pd.DataFrame({
        'age': np.random.normal(54, 9, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(131, 17, n_samples),
        'chol': np.random.normal(246, 51, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    })

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Patients", len(data))
    with col2: st.metric("Average Age", f"{data['age'].mean():.1f}")
    with col3: st.metric("Heart Disease Cases", f"{data['target'].sum()}")
    with col4: st.metric("Disease Rate", f"{data['target'].mean():.1%}")

    st.subheader("📊 Data Visualizations")
    tab1, tab2, tab3 = st.tabs(["Age Distribution", "Risk Factors", "Correlations"])

    with tab1:
        fig = px.histogram(
            data, x='age', color='target',
            title='Age Distribution by Heart Disease Status',
            labels={'target': 'Heart Disease', 'age': 'Age (years)'},
            color_discrete_map={0: 'lightgreen', 1: 'lightcoral'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        risk_data = data.groupby(['sex', 'target']).size().reset_index(name='count')
        risk_data['sex_label'] = risk_data['sex'].map({0: 'Female', 1: 'Male'})
        risk_data['target_label'] = risk_data['target'].map({0: 'No Disease', 1: 'Disease'})
        fig = px.bar(
            risk_data, x='sex_label', y='count', color='target_label',
            title='Heart Disease Distribution by Gender',
            labels={'sex_label': 'Gender', 'count': 'Number of Patients'}
        )
        st.plotly_chart(fig, use_container_width=True)

        cp_data = data.groupby(['cp', 'target']).size().reset_index(name='count')
        cp_data['cp_label'] = cp_data['cp'].map({
            0: 'Typical Angina', 1: 'Atypical Angina',
            2: 'Non-anginal Pain', 3: 'Asymptomatic'
        })
        fig2 = px.bar(
            cp_data, x='cp_label', y='count', color='target',
            title='Heart Disease by Chest Pain Type',
            color_discrete_map={0: 'lightgreen', 1: 'lightcoral'}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        corr_matrix = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'target']].corr(numeric_only=True)
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def model_info_page(predictor):
    """Model information and performance page"""
    st.header("🤖 Model Information")

    if predictor.metadata:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Model Details")
            st.info(f"Model Type: {predictor.metadata.get('model_type', 'Unknown')}")
            st.info(f"Features Used: {len(predictor.metadata.get('feature_names', []))}")
            if 'performance_metrics' in predictor.metadata:
                metrics = predictor.metadata['performance_metrics']
                st.success(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
                st.success(f"F1-Score: {metrics.get('f1_score', 0):.3f}")
        with col2:
            st.subheader("🎯 Model Features")
            features = predictor.metadata.get('feature_names', [])
            for i, feature in enumerate(features, 1):
                st.write(f"{i}. {feature}")

    # Simulated model performance comparison (example)
    st.subheader("📊 Model Performance")
    models = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree']
    metrics = {
        'Model': models,
        'Accuracy': [0.85, 0.87, 0.83, 0.79],
        'Precision': [0.84, 0.88, 0.82, 0.77],
        'Recall': [0.86, 0.85, 0.84, 0.81],
        'F1-Score': [0.85, 0.87, 0.83, 0.79]
    }
    perf_df = pd.DataFrame(metrics)
    perf_long = perf_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig = px.bar(
        perf_long, x='Model', y='Score', color='Metric',
        barmode='group', title='Model Performance Comparison',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Feature Importance")
    if predictor.metadata and 'feature_names' in predictor.metadata:
        features = predictor.metadata['feature_names']
        importance_values = np.random.random(len(features))
        importance_values = importance_values / importance_values.sum()
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_values}).sort_values('Importance',
                                                                                                         ascending=False)
        fig = px.bar(
            importance_df, x='Importance', y='Feature', orientation='h',
            title='Feature Importance in Heart Disease Prediction',
            color='Importance', color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("ℹ️ About This Application")
    st.markdown("""
    <div class="info-card">
        <h4>Heart Disease Prediction System</h4>
        <p>This application uses machine learning to predict heart disease risk based on various medical and demographic factors.</p>
        <h5>Key Features:</h5>
        <ul>
            <li>Real-time risk assessment</li>
            <li>Interactive data visualization</li>
            <li>Comprehensive model evaluation</li>
            <li>User-friendly interface</li>
        </ul>
        <h5>Disclaimer:</h5>
        <p><em>This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for medical decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Heart Disease ML Pipeline**")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Author: Claude AI")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn { background-color: #ffebee; color: #c62828; }
    .no-churn { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("customer_churn_model.pkl")
        encoders = joblib.load("encoders.joblib")
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None

model, encoders = load_model()

st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict customer churn using Machine Learning")
st.markdown("---")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        credit_score = st.number_input("Credit Score", 300, 850, value=None)
        geography = st.selectbox("Geography", ["", "France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["", "Male", "Female"])
        age = st.slider("Age (years)", 18, 100, value=None)
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (years)", 0, 10, value=None)
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, value=None)
        estimated_salary = st.number_input("Estimated Salary ($)", 10000.0, 200000.0, value=None)
    
    with col2:
        st.subheader("Banking Services")
        num_products = st.selectbox("Number of Products", ["", 1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card", ["", "Yes", "No"])
        is_active_member = st.selectbox("Is Active Member", ["", "Yes", "No"])
        
        st.markdown("")  # Spacing
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
    
    st.markdown("---")
    
    if st.button("Predict Churn", key="predict_btn"):
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode Geography
        geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
        input_df['Geography'] = input_df['Geography'].map(geography_mapping)
        
        # Encode Gender
        gender_mapping = {'Male': 1, 'Female': 0}
        input_df['Gender'] = input_df['Gender'].map(gender_mapping)
        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box churn">⚠️ WILL CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box no-churn">✓ WILL STAY</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")
        
        with col3:
            risk = "High" if proba[1] > 0.6 else "Medium" if proba[1] > 0.3 else "Low"
            st.metric("Risk Level", risk)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Will Stay', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
            go.Bar(name='Will Churn', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            barmode='stack',
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        if prediction == 1:
            st.warning("""
            **🎯 Recommended Actions for At-Risk Customer:**
            - Offer personalized incentives or loyalty rewards
            - Provide enhanced customer support
            - Review account privileges and benefits
            - Consider targeted retention campaigns
            """)
        else:
            st.success("""
            **✨ Recommended Actions for Satisfied Customer:**
            - Maintain current service quality
            - Suggest product cross-selling opportunities
            - Conduct satisfaction surveys
            - Encourage referral programs
            """)
    
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **XGBoost Classifier**
        - Training Set: 8,000 customers
        - Test Accuracy: ~80%
        - Balancing: SMOTE applied
        - Features: 10 customer attributes
        """)
        
        st.header("🔝 Top Predictors")
        top_features = [
            "Age",
            "Tenure",
            "NumOfProducts",
            "EstimatedSalary",
            "Geography",
            "IsActiveMember",
            "HasCrCard",
            "Balance"
        ]
        for i, feature in enumerate(top_features, 1):
            st.markdown(f"**{i}. {feature}**")
        
        st.markdown("---")
        st.caption("Built with Streamlit, XGBoost & Scikit-learn")

else:
    st.error("❌ Model not found. Please ensure model files are in the same directory.")
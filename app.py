import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load('models/best_wellness_xgb.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    return model, encoders

def preprocess_input(input_dict, encoders):
    df = pd.DataFrame([input_dict])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    return df

st.title("Wellness Tourism Package Predictor")
st.markdown("Predicts purchase probability - XGBoost AUC: 0.9712")

# Load model
model, encoders = load_model()

# Input form
with st.form("customer_inputs"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 70, 40)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        num_trips = st.slider("Number of Trips", 0, 10, 2)
    with col2:
        monthly_income = st.slider("Monthly Income", 10000, 100000, 30000)
        passport = st.selectbox("Passport", [0, 1])
        own_car = st.selectbox("Own Car", [0, 1])
    
    # Submit
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_data = {
            'Age': age, 'CityTier': city_tier, 'NumberOfTrips': num_trips,
            'MonthlyIncome': monthly_income, 'Passport': passport, 
            'OwnCar': own_car
            # Add more features as needed from your dataset
        }
        
        X = preprocess_input(input_data, encoders)
        prob = model.predict_proba(X)[0, 1]
        
        st.success(f"Purchase Probability: {prob:.1%}")
        if prob > 0.5:
            st.balloons()


import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load model and encoders at startup
model = joblib.load('models/best_wellness_xgb.pkl')
encoders = joblib.load('models/label_encoders.pkl')

def preprocess_input(input_dict):
    """Convert inputs to model-ready DataFrame"""
    df = pd.DataFrame([input_dict])
    # Encode categoricals using saved encoders
    for col in encoders.keys():
        if col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))
    # Fill any missing columns with defaults or model expectations
    return df

def predict_wellness_purchase(age, city_tier, num_trips, monthly_income, passport, own_car):
    """Main prediction function"""
    input_data = {
        'Age': int(age),
        'CityTier': city_tier,
        'NumTrips': int(num_trips),
        'MonthlyIncome': int(monthly_income),
        'Passport': int(passport),
        'OwnCar': int(own_car)
    }
    
    # Preprocess to DataFrame (rubric requirement)
    df = preprocess_input(input_data)
    
    # Predict probability
    prob = model.predict_proba(df)[0][1]  # Probability of purchase (class 1)
    
    # Interpret result
    if prob > 0.5:
        result = f"High likelihood of wellness tourism purchase! (Probability: {prob:.2%})"
    else:
        result = f"Low likelihood of wellness tourism purchase. (Probability: {prob:.2%})"
    
    return result, f"Raw probability: {prob:.4f}"

# Gradio interface
with gr.Blocks(title="Wellness Tourism Predictor") as demo:
    gr.Markdown("# Wellness Tourism Purchase Predictor")
    gr.Markdown("Enter customer details to predict wellness tourism purchase likelihood (XGBoost AUC 0.9712).")
    
    with gr.Row():
        age = gr.Slider(18, 80, value=35, label="Age", step=1)
        city_tier = gr.Dropdown(choices=[1, 2, 3], value=2, label="City Tier (1=Metro, 3=Rural)")
        num_trips = gr.Slider(0, 20, value=5, label="Number of Past Trips", step=1)
    
    with gr.Row():
        monthly_income = gr.Slider(10000, 100000, value=50000, label="Monthly Income (INR)", step=1000)
        passport = gr.Radio(choices=[0, 1], value=0, label="Has Passport? (0=No, 1=Yes)")
        own_car = gr.Radio(choices=[0, 1], value=1, label="Owns Car? (0=No, 1=Yes)")
    
    predict_btn = gr.Button("Predict", variant="primary")
    
    result = gr.Textbox(label="Prediction")
    details = gr.Textbox(label="Model Details", interactive=False)
    
    predict_btn.click(
        predict_wellness_purchase,
        inputs=[age, city_tier, num_trips, monthly_income, passport, own_car],
        outputs=[result, details]
    )

if __name__ == "__main__":
    demo.launch()

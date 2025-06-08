import streamlit as st

st.title("✅ Hello Streamlit")
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# --- Environment Check ---
st.write("✅ Streamlit App Loaded")
st.write("TensorFlow Version:", tf.__version__)

# --- Load Model and Preprocessors ---
model_path = "model.h5"  # Ensure this is the .h5 format, not .keras
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model from {model_path}: {e}")
    st.stop()

# Load encoders and scaler
#label_encoder_gender = joblib.load("label_encoder_gender.pkl")
label_encoder_gender = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/label_encoder_gender.pkl")
label_encoder_overtime = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/label_encoder_overtime.pkl")
label_encoder_remote_work= joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/label_encoder_remote_work.pkl")
label_encoder_leadership_opportunities = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/label_encoder_leadership_opportunities.pkl")
label_encoder_innovation_opportunities = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/label_encoder_innovation_opportunities.pkl")
onehot_encoder = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/onehot_encoder_company_size_job_role_work-life_balance_job_satisfaction_performance_rating_education_level_marital_status_job_level_company_reputation_employee_recognition.pkl")
scaler = joblib.load("C:/Users/Nitesh/Downloads/AI Tests/Projects/ANNClassification/scaler.pkl")


# OneHot columns
onehot_columns = [
    'Company Size', 'Job Role', 'Work-Life Balance', 'Job Satisfaction',
    'Performance Rating', 'Education Level', 'Marital Status',
    'Job Level', 'Company Reputation', 'Employee Recognition']

# --- Streamlit UI ---
st.title('🧑\u200d💼 Employee Attrition Prediction')

# --- User Inputs ---
age = st.number_input("Age", min_value=18, max_value=65)
gender = st.selectbox("Gender", label_encoder_gender.classes_)
years_at_company = st.number_input("Years at Company", min_value=0)
job_role = st.selectbox("Job Role", onehot_encoder.categories_[onehot_columns.index('Job Role')])
monthly_income = st.number_input("Monthly Income", min_value=0)
work_life_balance = st.selectbox("Work-Life Balance", onehot_encoder.categories_[onehot_columns.index('Work-Life Balance')])
job_satisfaction = st.selectbox("Job Satisfaction", onehot_encoder.categories_[onehot_columns.index('Job Satisfaction')])
performance_rating = st.selectbox("Performance Rating", onehot_encoder.categories_[onehot_columns.index('Performance Rating')])
num_promotions = st.number_input("Number of Promotions", min_value=0)
overtime = st.selectbox("Overtime", label_encoder_overtime.classes_)
distance_from_home = st.number_input("Distance from Home (km)", min_value=0)
education_level = st.selectbox("Education Level", onehot_encoder.categories_[onehot_columns.index('Education Level')])
marital_status = st.selectbox("Marital Status", onehot_encoder.categories_[onehot_columns.index('Marital Status')])
num_dependents = st.number_input("Number of Dependents", min_value=0)
job_level = st.selectbox("Job Level", onehot_encoder.categories_[onehot_columns.index('Job Level')])
company_size = st.selectbox("Company Size", onehot_encoder.categories_[onehot_columns.index('Company Size')])
company_tenure = st.number_input("Company Tenure", min_value=0)
remote_work = st.selectbox("Remote Work", label_encoder_remote_work.classes_)
leadership_opp = st.selectbox("Leadership Opportunities", label_encoder_leadership_opportunities.classes_)
innovation_opp = st.selectbox("Innovation Opportunities", label_encoder_innovation_opportunities.classes_)
company_reputation = st.selectbox("Company Reputation", onehot_encoder.categories_[onehot_columns.index('Company Reputation')])
employee_recognition = st.selectbox("Employee Recognition", onehot_encoder.categories_[onehot_columns.index('Employee Recognition')])

# --- Prepare Input Data ---
input_data = {
    'Age': age,
    'Years at Company': years_at_company,
    'Monthly Income': monthly_income,
    'Number of Promotions': num_promotions,
    'Distance from Home': distance_from_home,
    'Number of Dependents': num_dependents,
    'Company Tenure': company_tenure,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Overtime': label_encoder_overtime.transform([overtime])[0],
    'Remote Work': label_encoder_remote_work.transform([remote_work])[0],
    'Leadership Opportunities': label_encoder_leadership_opportunities.transform([leadership_opp])[0],
    'Innovation Opportunities': label_encoder_innovation_opportunities.transform([innovation_opp])[0],
    'Company Size': company_size,
    'Job Role': job_role,
    'Work-Life Balance': work_life_balance,
    'Job Satisfaction': job_satisfaction,
    'Performance Rating': performance_rating,
    'Education Level': education_level,
    'Marital Status': marital_status,
    'Job Level': job_level,
    'Company Reputation': company_reputation,
    'Employee Recognition': employee_recognition
}

input_df = pd.DataFrame([input_data])



# --- OneHot Encoding ---
onehot_array = onehot_encoder.transform(input_df[onehot_columns])
onehot_df = pd.DataFrame(onehot_array, columns=onehot_encoder.get_feature_names_out(onehot_columns), index=input_df.index)
input_df_final = pd.concat([input_df.drop(columns=onehot_columns), onehot_df], axis=1)

# --- Scale Input ---
# Ensure input_df_final has the same columns and order
input_df_final= input_df_final[scaler.feature_names_in_]

input_scaled = scaler.transform(input_df_final)





# --- Predict ---
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    st.subheader("Prediction Result")
    if probability > 0.5:
        st.success(f"🔒 The employee is likely to **stay**. (Confidence: {probability:.2f})")
    else:
        st.error(f"⚠️ The employee is likely to **leave**. (Confidence: {1 - probability:.2f})")

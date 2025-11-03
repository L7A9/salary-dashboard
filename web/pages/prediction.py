import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).parent  # folder of this script
model_path = BASE_DIR / "salary_pipeline.pkl"

pipeline = joblib.load(model_path)

st.title("Salary Prediction App")  # Example title

st.header("Enter Employee Details:")

# Input widgets
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.text_input("Job Title", "Software Engineer")  # could also be a selectbox if you have known titles
years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# Make a DataFrame for the input (pipeline expects a DataFrame)
input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "education level": [education_level],
    "job title": [job_title],
    "years of experience": [years_of_experience]
})

# Button to predict
if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted value: {prediction[0]:.2f}")

st.subheader("Go to the Home Page")
st.page_link("home.py", label="Home")


st.markdown("---")
st.markdown("Developed by Amine EL Hanine | Data Science student | Salary Prediction Dashboard | 2025, [Github](https://github.com/L7A9)")

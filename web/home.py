import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Salary Dashboard", layout="centered")
st.title("Salary Data Analysis (Test Dataset)")


BASE_DIR = Path(__file__).parent  # folder of this script
df_path = BASE_DIR / "model_results.csv"

st.markdown("### Data Preprocessing Explanation")
st.markdown("""
1. **Gender and Education Level:** These columns are categorical but have a natural order (education) or binary options (gender). We used **Ordinal Encoding** to convert them into numeric values that the model can understand. For education level, this preserves the hierarchy (Bachelor < Master < PhD), and for gender, it converts it to 0/1.
""")
st.markdown("""
2. **Job Title:** Since job titles have many unique categories, we used **Target Encoding**. This replaces each job title with the **average salary** for that role. It helps the model understand the impact of each job on salary without creating hundreds of columns.
""")
st.markdown("""
3. **Age and Years of Experience:** These are numeric features, but on different scales. We applied **Standard Scaling** to transform them to have a **mean of 0 and standard deviation of 1**. This ensures the model treats all numeric features equally and improves training stability.
""")

st.markdown("### Before Scaling and Encoding")

st.markdown("#### Dataset preview Before")
st.dataframe(df[['age','gender','job title','years of experience','education level']].head())
st.markdown("---")

st.markdown("#### Describe the Dataset")
st.dataframe(df[['age','gender','job title','years of experience','education level']].describe(include='all'))
st.markdown("---")

st.markdown("#### After Scaling and Encoding")
st.subheader("Dataset preview After Scaling and Encoding")
st.dataframe(df[['age_scaled','gender_code','job_code','experience_scaled','education_level_code']].head())
st.markdown("---")

st.markdown("#### Describe the Dataset after Scaling and Encoding")
st.dataframe(df[['age_scaled','gender_code','job_code','experience_scaled','education_level_code']].describe())
st.markdown("---")



# ---------- 1. Salary Distribution ----------
st.markdown("**Salary Distribution:** Shows how salaries are spread across the test dataset. The KDE curve highlights density, helping identify common salary ranges.")

fig, ax = plt.subplots(figsize=(5,3))
sns.histplot(df['actual_salary'], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_title("Salary Distribution (Test Data)",fontsize=10)
ax.set_xlabel("Salary ($)",fontsize=7.5)
ax.set_ylabel("Count",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")


# ---------- 2. Salary vs Years of Experience ----------
st.markdown("**Salary vs Years of Experience:** Shows the relationship between experience and salary, colored by education level. Helps see how experience and education affect salary.")

fig, ax = plt.subplots(figsize=(5,3))
sns.scatterplot(
    x=df['years of experience'], 
    y=df['actual_salary'], 
    hue=df['education level'], 
    palette='Set2', 
    ax=ax
)
ax.set_title("Salary vs Years of Experience",fontsize=10)
ax.set_xlabel("Years of Experience",fontsize=7.5)
ax.set_ylabel("Salary ($)",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# ---------- 3. Salary vs Age by Gender ----------
st.markdown("**Salary vs Age by Gender:** Shows how salaries vary with age, separated by gender. Helps understand age and gender impact on salaries.")

fig, ax = plt.subplots(figsize=(5,3))
sns.scatterplot(
    x=df['age'], 
    y=df['actual_salary'], 
    hue=df['gender'], 
    palette='cool', 
    alpha=0.7, 
    ax=ax
)
ax.set_title("Salary vs Age by Gender",fontsize=10)
ax.set_xlabel("Age",fontsize=7.5)
ax.set_ylabel("Salary ($)",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# ---------- 4. Average Salary by Education Level ----------
st.markdown("**Average Salary by Education Level:** Bar chart showing the mean salary for each education level. Helps compare how education impacts salary.")

fig, ax = plt.subplots(figsize=(5,3))
edu_avg = df.groupby("education level")['actual_salary'].mean().sort_values()
sns.barplot(x=edu_avg.index, y=edu_avg.values, palette='pastel', ax=ax)
ax.set_title("Average Salary by Education Level",fontsize=10)
ax.set_xlabel("Education Level",fontsize=7.5)
ax.set_ylabel("Average Salary ($)",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# ---------- 5. Top 10 Job Titles by Average Salary ----------
st.markdown("**Top 10 Jobs by Average Salary:** Shows which job titles have the highest average salaries. Useful for career insights and comparison.")

fig, ax = plt.subplots(figsize=(5,3))
top_jobs = df.groupby("job title")['actual_salary'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='magma', ax=ax)
ax.set_title("Top 10 Jobs by Average Salary",fontsize=10)
ax.set_xlabel("Average Salary ($)",fontsize=7.5)
ax.set_ylabel("Job Title",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# ---------- 6. Actual vs Predicted Salary ----------
st.markdown("**Actual vs Predicted Salary:** Compares the model predictions to actual salaries. Dots close to the red line indicate accurate predictions.")

fig, ax = plt.subplots(figsize=(5,3))
sns.scatterplot(x=df['actual_salary'], y=df['predicted_salary'], color='green', alpha=0.7, ax=ax)
ax.plot(
    [df['actual_salary'].min(), df['actual_salary'].max()],
    [df['actual_salary'].min(), df['actual_salary'].max()],
    'r--', lw=2
)
ax.set_title("Actual vs Predicted Salary",fontsize=10)
ax.set_xlabel("Actual Salary ($)",fontsize=7.5)
ax.set_ylabel("Predicted Salary ($)",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# ---------- 7. Residuals vs Predicted Salary ----------
st.markdown("**Residuals vs Predicted Salary:** Shows prediction errors (residuals) across predicted salaries. Helps detect bias, patterns, or heteroscedasticity in predictions.")

df['error'] = df['actual_salary'] - df['predicted_salary']
fig, ax = plt.subplots(figsize=(5,3))
sns.scatterplot(x=df['predicted_salary'], y=df['error'], alpha=0.7, color='purple', ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_title("Residuals vs Predicted Salary",fontsize=10)
ax.set_xlabel("Predicted Salary ($)",fontsize=7.5)
ax.set_ylabel("Residual ($)",fontsize=7.5)
ax.tick_params(axis='both', which='major', labelsize=6)
st.pyplot(fig, use_container_width=False)
st.markdown("---")


st.subheader("Go to the Prediction Page")
st.page_link("pages/prediction.py", label="Open Salary Predictor")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>

    <div class="footer">
        Developed by Amine EL Hanine | Data Science student & Salary Prediction Dashboard | 2025
        <a href="https://github.com/L7A9" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)


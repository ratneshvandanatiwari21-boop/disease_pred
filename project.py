import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Backend: Data aur Model load karna
@st.cache_resource
def load_model():
    with open(BASE_DIR / 'Heart_Disease.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / 'heart_dataset.csv')

model = load_model()
df = load_data()

# Sidebar Setup
st.sidebar.title("Navigation Menu")
st.sidebar.info("Yahan se Prediction aur Graphs ke beech switch karein.")
page = st.sidebar.radio("Go to:", ["Heart Disease Prediction", "Data Graphs & Insights"])

# ----------------- SECTION 1: PREDICTION -----------------
if page == "Heart Disease Prediction":
    st.title("Heart Disease Prediction System")
    st.write("Apne symptoms enter karein aur check karein ki heart risk kaisa hai.")

    # 3 Columns ka layout banate hain taaki UI clean dikhe
    col1, col2, col3 = st.columns(3)

    # Actual meaning wale Inputs
    with col1:
        age = st.number_input("Age (Umar)", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Gender", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type (Seene me dard ka prakar)", 
                          ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

    with col2:
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl ?", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", 
                               ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)

    with col3:
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
        st.info("Prediction model uses the 9 features selected during training.")

    # Dictionary banate hain in actual texts ko wapas numbers me convert karne ke liye (model ke liye)
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"No": 0, "Yes": 1}
    restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}

    st.markdown("---")
    
    # Prediction Button
    if st.button("Predict Heart Disease", use_container_width=True):
        # User ke text input ko wapas numeric array me convert karna
        input_data = [
            age, sex_map[sex], cp_map[cp], trestbps, chol, fbs_map[fbs], 
            restecg_map[restecg], thalach, oldpeak
        ]
        
        # Model ka prediction
        prediction = model.predict([input_data])
        
        # Result Show karna
        if prediction[0] == 1:
            st.error("**Prediction:** High Risk of Heart Disease. Kripya doctor se consult karein.")
        else:
            st.success("**Prediction:** Low Risk (Healthy). Aapka heart condition safe lag raha hai.")

# ----------------- SECTION 2: GRAPHS -----------------
elif page == "Data Graphs & Insights":
    st.title("Dataset Graphs & Analysis")
    st.write("Yahan aap check kar sakte hain ki purane dataset me trends kaise hain.")

    st.markdown("### 1. Healthy vs Defective Heart Count")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2', ax=ax1)
    ax1.set_xticklabels(['Healthy (0)', 'Heart Disease (1)'])
    st.pyplot(fig1)

    st.markdown("### 2. Age vs Maximum Heart Rate (Thalach)")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 3. Chest Pain Type ka Asar")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='cp', hue='target', data=df, palette='viridis', ax=ax3)
    ax3.set_xticklabels(['Typical', 'Atypical', 'Non-anginal', 'Asymptomatic'])
    st.pyplot(fig3)

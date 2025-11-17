import streamlit as st
import pandas as pd
import streamlit as st
import joblib

import os

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        model = None
else:
    st.error(f"Model file not found at {MODEL_PATH}")
    model = None





st.title("Titanic Survival Prediction")




# User input
st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", value=32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    df_input = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    # Make prediction
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0, 1]

    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.success(f"Prediction: **{result}** (Probability of survival: {proba:.2f})")

import streamlit as st
import pandas as pd
import pickle
import os

st.title("Titanic Survival Prediction")

# ✅ Make sure path is correct
MODEL_PATH = os.path.join("models", "model.pkl")

# Load model
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_PATH)


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

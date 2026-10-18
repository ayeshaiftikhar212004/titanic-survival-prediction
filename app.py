
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
gender = st.selectbox("Gender", ["Male", "Female"])
siblings = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival"):
    gender_value = 1 if gender == "Female" else 0

    input_data = pd.DataFrame(
        [[pclass, gender_value, siblings, embarked]],
        columns=["PClass", "Gender", "Sibling", "Embarked"]
    )

    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Prediction: Passenger may survive.")
        else:
            st.warning("Prediction: Passenger may not survive.")
    except Exception as error:
        st.error("The model's expected input format differs from this app.")
        st.write(str(error))

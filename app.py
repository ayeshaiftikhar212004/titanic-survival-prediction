import streamlit as st
import pickle

# Load the trained model
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Codes match the notebook's LabelEncoders (alphabetical order)
PCLASS_CODES = {"First": 0, "Second": 1, "Third": 2}
GENDER_CODES = {"Female": 0, "Male": 1}
SIBLING_CODES = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
EMBARKED_CODES = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Male", "Female"])
siblings = st.selectbox("Number of Siblings/Spouses Aboard", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict Survival"):
    # Feature order must match training: Pclass, Gender, Sibling, Embarked
    input_data = [[
        PCLASS_CODES[pclass],
        GENDER_CODES[gender],
        SIBLING_CODES[siblings],
        EMBARKED_CODES[embarked],
    ]]

    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Prediction: Passenger may survive.")
        else:
            st.warning("Prediction: Passenger may not survive.")
    except Exception as error:
        st.error("Prediction failed.")
        st.write(str(error))

import streamlit as st
import pandas as pd
import pickle
import os
import joblib
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    with open("med_knn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_data():
    return joblib.load("le_medicine.pkl")


model = load_model()
pred_model = load_label_data()

# Pre-defined categories (based on dataset)
gender_list = ['Male', 'Female']
symptom_list = ['Fever', 'Headache', 'Shortness of Breath' ,'Nausea' ,'Sore Throat','Joint Pain', 'Chest Pain' ,'Itching' ,'Abdominal Pain' ,'Fatigue' ,'Cough','Stomach Pain' ,'Back Pain' ,'Anxiety' ,'Skin Rash' ,'Dizziness','Muscle Pain' ,' Back Pain']
cause_list = ['Viral Infection', 'Stress', 'Pollution', 'Food Poisoning','Bacterial Infection', 'Rheumatoid Arthritis','High Blood Pressure', 'Allergies', 'Poor Diet', 'Depression','Cold Weather', 'Motion Sickness', 'Smoking', 'Migraine Triggers','Spicy Food', 'Autoimmune Response', 'Herniated Disc', 'Pregnancy','Anemia', 'Obesity', 'Osteoarthritis', 'Dehydration', 'Tension','vereating            ', 'Menstrual Cycle', 'Heart Disease','Hypothyroidism', 'Infection', 'COVID-19 Exposure', 'Overexertion','Eye Strain', 'Sciatica', 'Chronic Fatigue Syndrome','Physical Exertion', 'COVID-19 exposure', 'Overeating','Chronic Fatigue', ' COVID-19 Exposure','Chronic Fatigue Syndrome', 'Bacterial Infection ','Anemia              ', 'Stress              ','Obesity             ', 'Allergies           ','Viral Infection     ', 'Rheumatoid Arthritis ','Dehydration         ', 'Tension             ','Overeating          ', 'Menstrual Cycle    ','Heart Disease       ', 'Hypothyroidism      ','Infection           ', 'COVID-19 Exposure   ','Overexertion        ', 'Food Poisoning  ', 'Eye Strain          ']
disease_list = ['Common Cold', 'Migraine', 'Asthma', 'Gastroenteritis','Strep Throat', 'Arthritis', 'Hypertension', 'Allergic Reaction','Indigestion', 'Major Depressive', 'Influenza', 'Motion Sickness','Chronic Bronchitis', 'Gastritis', 'Rheumatoid Arthritis','Tonsillitis', 'Sciatica', 'Morning Sickness', 'Iron Deficiency','Panic Disorder', 'Sleep Apnea', 'Dermatitis','Respiratory infection', 'Heat Exhaustion', 'Tension Headache','Menstrual Cramps', 'Coronary ArteryDisease', 'Thyroid Disorder','Pneumonia', 'COVID-19', 'Muscle Strain', 'Vision Fatigue','Herniated Disc', 'Chronic Fatigue Syndrome', 'Anxiety Disorder','Muscle Overuse', 's  Arthritis', 'Major Depressive Disorder','Allergic Reacti', 'Chronic Fatigue', ' Arthritis','RespiratoryInfection', 'Coronary Artery', 'is  Arthritis','Respiratory Infection', 'Coronary Artery Disease', 'Respiratory','Strep Throat     ', 'Iron Deficiency ', 'Panic Disorder  ','Sleep Apnea     ', 'Dermatitis      ','Respiratory     AntiInfection', 'Arthritis     ','Heat Exhaustion Hydr', 'Tension Headache Rel','Indigestion     Anta', 'Menstrual Cramps Pai','Coronary Artery Disease   ', 'Disease         ','Pheumonia      ', 'COVID-19      ', 'Allergic Reaction An','Muscle Strain  ']

# Encoders (simulate label encoding – replace with actual if needed)
def encode_inputs(gender, symptom, cause, disease):
    return [
        gender_list.index(gender),
        symptom_list.index(symptom),
        cause_list.index(cause),
        disease_list.index(disease)
    ]

# Save results to CSV
def save_result_to_csv(details, filename="medicine_predictions.csv"):
    new_entry = pd.DataFrame([details])
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        combined = pd.concat([existing, new_entry], ignore_index=True)
    else:
        combined = new_entry
    combined.to_csv(filename, index=False)

# Streamlit App UI
def main():
    st.title("💊 AI Medicine Predictor")
    st.markdown("Predict the right **medicine** based on symptoms, causes, and diagnosis.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Select Gender", gender_list)
        symptom = col2.selectbox("Select Symptom", symptom_list)

        cause = st.selectbox("Select Cause", cause_list)
        disease = st.selectbox("Select Disease", disease_list)

        submit = st.form_submit_button("Predict Medicine")

    if submit:
        encoded_input = encode_inputs(gender, symptom, cause, disease,)
        
        encoded_input = np.array(encoded_input).reshape(1, -1)
        prediction = model.predict(encoded_input)
        
        prediction_medicine = pred_model['Medicine'].inverse_transform(prediction)[0]
            
        st.success(f"🧾 **Predicted Medicine**: `{prediction_medicine}`")

        patient_data = {
            "Gender": gender,
            "Symptom": symptom,
            "Cause": cause,
            "Disease": disease,
            "Predicted Medicine": prediction_medicine
        }

        save_result_to_csv(patient_data)

        st.markdown("### 📋 Patient Summary")
        st.json(patient_data)

        st.markdown("✅ Result saved to `medicine_predictions.csv`.")

if __name__ == "__main__":
    main()

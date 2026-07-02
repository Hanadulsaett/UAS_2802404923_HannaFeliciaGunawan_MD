import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_components():
    with open('model.pkl', 'rb') as f:
        components = pickle.load(f)
    return components

components = load_components()
model = components['model']
scaler = components['scaler']
label_encoders = components['label_encoders']

def preprocess_input(df_input):
    df_clean = df_input.copy()
    cols_to_drop = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Credit_Score']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

    numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                    'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('_', '')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median()) # Imputasi darurat

    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            df_clean[col] = df_clean[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            df_clean[col] = le.transform(df_clean[col])
            
    df_clean = df_clean.fillna(0)
    return scaler.transform(df_clean)

#streamlit 
st.title("Prediksi Performa Kredit Nasabah")
st.write("Aplikasi untuk menilai Credit Score menggunakan Machine Learning.")
st.divider()

st.header("Test Case Generator")
st.write("Karena jumlah fitur sangat banyak, kita akan menggunakan data yang sudah ada sebagai test case.")

@st.cache_data
def load_data():
    return pd.read_csv('data_B.csv')

df_raw = load_data()

row_index = st.number_input("Pilih Index Data (0 - 24000) untuk di-test:", min_value=0, max_value=len(df_raw)-1, value=0)
selected_data = df_raw.iloc[[row_index]]

st.subheader("Data Nasabah Terpilih:")
st.dataframe(selected_data)

if st.button("Prediksi Credit Score"):
    with st.spinner("Memproses..."):
        X_input = preprocess_input(selected_data)
        prediction = model.predict(X_input)[0]

        target_encoder = label_encoders.get('Credit_Score', None)
        
        if prediction == 0:
            hasil_teks = "Good"
            st.success(f"Prediksi: **{hasil_teks}**")
        elif prediction == 1:
            hasil_teks = "Poor"
            st.error(f"Prediksi: **{hasil_teks}**")
        else:
            hasil_teks = "Standard"
            st.warning(f"Prediksi: **{hasil_teks}**")

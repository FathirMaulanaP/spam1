import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Judul aplikasi
st.set_page_config(page_title="Deteksi Spam SMS")
st.title("📩 Deteksi Spam SMS")

st.write("Masukkan pesan SMS untuk mengetahui apakah termasuk Spam atau Ham.")

# Input pengguna
pesan = st.text_area("Masukkan Pesan")

if st.button("Prediksi"):
    if pesan.strip() == "":
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:
        # Gunakan transform(), JANGAN fit_transform()
        input_vec = vectorizer.transform([pesan])

        pred = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)

        if pred == "spam":
            st.error("Hasil Prediksi: SPAM")
            st.write(f"Probabilitas Spam: {prob.max()*100:.2f}%")
        else:
            st.success("Hasil Prediksi: HAM")
            st.write(f"Probabilitas Ham: {prob.max()*100:.2f}%")

import joblib
import streamlit as st

model = joblib.load ("model_logistic_regression.pkl")
tfidf = joblib.load ("tfidf_vectorizer.pkl")

st.title("Aplikasi klasifikasi komentar publik")
st.write("Aplikasi ini dibuat menggunakan Teknologi NLP dengan menfaatkan model machenin learning Logistic Regresion")
input = st.text_input("Masukan Komentar Anda")
if st.button("Sumbit"):
    if input.strip() == "":
        st.warning("Komentar tidak boleh kosong")
    else:
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "NEGATIF",
            1: "POSITIF"
        }
        st.subheader("Hasil Analisa Komentar")
        st.write("**Komentar :**", label_map.get(prediksi,prediksi))

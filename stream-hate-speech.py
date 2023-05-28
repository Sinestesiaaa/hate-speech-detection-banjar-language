import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


# membaca model
hate_speech_detection = pickle.load(open('ensemble-method.pkl', 'rb'))
cv = pickle.load(open('feature-extraction.pkl', 'rb'))

# judul web
st.title("Deteksi Ujaran Kebencian bahasa Banjar")
st.text("Menggunakan N-Estimator dengan Nilai 5")

text = st.text_area("Masukan Kalimat bahasa Banjar", "Masukan Kalimat Disini")

prediction_labels = {'Ujaran Kebencian': 1, 'Bukan Ujaran Kebencian': 0}
if st.button("Klasifikasikan"):
    if text.strip():  # Memeriksa apakah teks tidak kosong setelah tombol diklik
        vect_text = cv.transform([text])

        prediction = hate_speech_detection.predict(vect_text)

        final_result = get_key(prediction, prediction_labels)

        if final_result == 'Bukan Ujaran Kebencian':
            st.success("Kalimat Termasuk: {}".format(final_result))
        elif final_result == 'Ujaran Kebencian':
            st.error("Kalimat Termasuk: {}".format(final_result))
    else:
        st.warning("Masukkan teks sebelum melakukan klasifikasi.")

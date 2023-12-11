import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

#call Model
model = pickle.load(open('catboost_model_no_outlier.pkl', 'rb'))

#call data
file_data = "pollutant-standards-index-southtangerang-2020-2022.csv"
try:
    df = pd.read_csv(file_data)
except FileNotFoundError:
    st.error("File CSVnya tidak ditemukan. Pastikan file ada dalam direktori yang benar atau ganti nama file sesuai dengan yang Anda miliki.")
    st.stop()

#open file css
with open('style.css')as f:
    st.markdown(f'<style>{f.read()}<style>',unsafe_allow_html = True)


#membuat tab
with st.sidebar:
    menuweb = st.radio("Menu Website",["Udara di Jakarta","Topik Diangkat","App","About"])
if menuweb == "Udara di Jakarta":
    st.title("Polusi Udara di Indonesia yang kian memburuk")
    st.markdown("""Jakarta, CNBC Indonesia - Berdasarkan data IQAir pada pagi hari ini Selasa (5/9/2023) pukul 06.00 WIB kualitas udara di Jakarta kembali memburuk dari status sedang menjadi tidak sehat dengan indeks kualitas udara AQI US 156 dan polutan utama PM2.5. Konsentrasi PM2.5 di Jakarta saat ini 12,9 kali nilai panduan kualitas udara tahunan WHO.
Angka AQI US ini lebih besar dibandingkan dengan angka kualitas udara hari sebelumnya di AQI US 95.

Cuaca Jakarta pagi ini berkabut dengan suhu 23 derajat celcius, kelembapan 75%, angin 7,4 hm/h dan tekanan 1.012 mbar.
Kembali memburuknya polusi Jakarta ini justru terjadi di tengah perhelatan Konferensi Tingkat Tinggi (KTT) ASEAN. KTT resmi dibuka pada hari ini dengan dihadiri puluhan pejabat selevel presiden.
Selain pemimpin ASEAN, hadir pula pejabat tinggi dari Amerika Serikat (AS), China, India, Korea Selatan, hingga Jepang sebagai mitra ASEAN.

Dalam rangking kota AQI langsung dari beberapa kota di Indonesia, hari ini pukul 06.00 WIB kota di Provinsi Kalimantan Barat kembali masuk urutan pertama dari 10 rangking kota berpolusi tidak sehat.

Pada pagi hari ini Kalimantan Barat kembali menjadi provinsi berpolusi buruk, sebelumnya kota Mempawah, Kalimantan Barat, lalu hari ini kota Terentang, Kalimantan Barat. Dan kota Jakarta hari ini tidak masuk dalam jajaran 10 rangking kota berpolusi tidak sehat. Namun, Provinsi Banten dan Jawa Barat masih mendominasi provinsi berpolusi buruk.
                Buruknya kualitas udara di beberapa wilayah Indonesia juga dipengaruhi kebakaran hutan di banyak titik.

Fenomena El Nino juga ikut memperparah kondisi kebakaran hutan dan lahan (karhutla). Beberapa pihak memprediksi karhutla tahun ini akan lebih parah dibandingkan dua tahun sebelumnya (2021-2022).

Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) telah menyerukan dampak dari iklim ekstrem El Nino di Indonesia dapat mengurangi curah hujan dan memicu terjadinya kekeringan. Pada tahun 2024 mendatang diprediksi akan menjadi tahun terpanas di dunia.""")

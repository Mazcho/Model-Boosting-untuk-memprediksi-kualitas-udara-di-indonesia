import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

# Memuat model
model = pickle.load(open('catboost_model_no_outlier_no_Max.pkl', 'rb'))

# Fungsi untuk normalisasi data pengguna
scaler = MinMaxScaler()

def normalisasi_data(data_pengguna):
    return scaler.transform(data_pengguna)

# Memuat data
file_data = "pollutant-standards-index-southtangerang-2020-2022.csv"

try:
    df = pd.read_csv(file_data)
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan file ada dalam direktori yang benar atau ganti nama file sesuai dengan yang Anda miliki.")
    st.stop()

# Membuka file css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

# Membuat tab
with st.sidebar:
    menuweb = st.radio("Menu Website", ["Udara di Jakarta", "Topik Diangkat", "App", "About"])

if menuweb == "Udara di Jakarta":
    st.title("Polusi Udara di Indonesia yang kian memburuk")
    st.markdown("""Jakarta, CNBC Indonesia - Berdasarkan data IQAir pada pagi hari ini Selasa (5/9/2023) kualitas udara di Jakarta kembali memburuk dari status sedang menjadi tidak sehat dengan indeks kualitas udara AQI US 156 dan polutan utama PM2.5. Konsentrasi PM2.5 di Jakarta saat ini 12,9 kali nilai panduan kualitas udara tahunan WHO.
                Angka AQI US ini lebih besar dibandingkan dengan angka kualitas udara hari sebelumnya di AQI US 95.

                Cuaca Jakarta pagi ini berkabut dengan suhu 23 derajat Celsius, kelembapan 75%, angin 7,4 hm/h, dan tekanan 1.012 mbar.
                Kembali memburuknya polusi Jakarta ini justru terjadi di tengah perhelatan Konferensi Tingkat Tinggi (KTT) ASEAN. KTT resmi dibuka pada hari ini dengan dihadiri puluhan pejabat selevel presiden.
                Selain pemimpin ASEAN, hadir pula pejabat tinggi dari Amerika Serikat (AS), China, India, Korea Selatan, hingga Jepang sebagai mitra ASEAN.

                Dalam rangking kota AQI langsung dari beberapa kota di Indonesia, hari ini pukul 06.00 WIB kota di Provinsi Kalimantan Barat kembali masuk urutan pertama dari 10 rangking kota berpolusi tidak sehat.

                Pada pagi hari ini Kalimantan Barat kembali menjadi provinsi berpolusi buruk, sebelumnya kota Mempawah, Kalimantan Barat, lalu hari ini kota Terentang, Kalimantan Barat. Dan kota Jakarta hari ini tidak masuk dalam jajaran 10 rangking kota berpolusi tidak sehat. Namun, Provinsi Banten dan Jawa Barat masih mendominasi provinsi berpolusi buruk.
                Buruknya kualitas udara di beberapa wilayah Indonesia juga dipengaruhi kebakaran hutan di banyak titik.

                Fenomena El Nino juga ikut memperparah kondisi kebakaran hutan dan lahan (karhutla). Beberapa pihak memprediksi karhutla tahun ini akan lebih parah dibandingkan dua tahun sebelumnya (2021-2022).

                Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) telah menyerukan dampak dari iklim ekstrem El Nino di Indonesia dapat mengurangi curah hujan dan memicu terjadinya kekeringan. Pada tahun 2024 mendatang diprediksi akan menjadi tahun terpanas di dunia.""")

if menuweb == "Topik Diangkat":
    st.write('Kuy')

if menuweb == "App":
    st.title("Halaman Prediksi udara hari ini")
    st.markdown("Halo! Sekarang kamu ada pada halaman prediksi udara yang telah dibuat oleh penulis kode ini. Silahkan masukkan aspek-aspek yang ada pada kolom di bawah ini. Setelah kalian memasukkan data yang dibutuhkan oleh prediksi cuaca ini, silahkan kalian tekan tombol prediksi cuaca. Nanti hasil prediksi akan muncul di sebelah kanan pada halaman ini. Selamat mencoba")

    col8, col9 = st.columns(2)

    # Fungsi untuk melakukan prediksi berdasarkan data pengguna
    def predict_air_quality(data_pengguna):
        # Load model
        with open('catboost_model_with_smote.pkl', 'rb') as model_file:
            catboost_classifier = pickle.load(model_file)

        # Load LabelEncoder
        with open('label_encoder.pkl', 'rb') as label_encoder_file:
            label_encoder = pickle.load(label_encoder_file)

        # Melakukan prediksi
        data_pengguna_encoded = label_encoder.transform(catboost_classifier.predict(data_pengguna))
        prediksi_cuaca = label_encoder.inverse_transform(data_pengguna_encoded)
        
        return prediksi_cuaca[0]

    # Menambahkan widget untuk upload file CSV
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca data dari file CSV yang diunggah
        df_uploaded = pd.read_csv(uploaded_file)

        with col8:
            # Menampilkan tabel data yang diunggah
            st.subheader("Data yang Diunggah")
            st.write(df_uploaded)

        with col9:
            # Menampilkan hasil prediksi untuk setiap baris di dataset yang diunggah
            st.subheader("Hasil Prediksi")
            predictions = []

            for index, row in df_uploaded.iterrows():
                data_pengguna = pd.DataFrame({
                    'PM2.5': [row['PM2.5']],
                    'PM10': [row['PM10']],
                    'SO2': [row['SO2']],
                    'CO': [row['CO']],
                    'O3': [row['O3']],
                    'NO2': [row['NO2']]
                })
                result = predict_air_quality(data_pengguna)
                predictions.append(result)

            df_uploaded['Prediksi Kualitas Udara'] = predictions
            st.write(df_uploaded)
            

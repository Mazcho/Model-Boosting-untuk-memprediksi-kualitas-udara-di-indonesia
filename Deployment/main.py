import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

#call Model

model = pickle.load(open('catboost_model_no_outlier_no_Max.pkl', 'rb'))

# Fungsi untuk normalisasi data pengguna
scaler = MinMaxScaler()
def normalisasi_data(data_pengguna):
    return scaler.transform(data_pengguna)



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
if menuweb == "Topik Diangkat":
    st.write('Kuy')
if menuweb == "App":
    st.write("Yuk")
    st.title("Halaman Prediksi udara hari ini")
    st.markdown("Halo! Sekarang kamu ada pada halaman prediksi udara yang telah dibuat oleh penulis kode ini. Silahkan masukkan aspek-aspek yang ada pada kolom di bawah ini. Setelah kalian memasukkan data yang dibutuhkan oleh prediksi cuaca ini, silahkan kalian tekan tombol prediksi cuaca. Nanti hasil prediksi akan muncul di sebelah kanan pada halaman ini. Selamat mencoba")
    
    col8, col9 = st.columns(2)
    
    with col8:
        pM25 = st.number_input("Masukkan PM25: ", value=0.0, step=0.1)
        pM10 = st.number_input("Masukkan PM10: ", value=0.0, step=0.1)
        sO2 = st.number_input("Masukkan SO2: ", value=0.0, step=0.1)
        cO = st.number_input("Masukkan CO: ", value=0.0, step=0.1)
        o3 = st.number_input("Masukkan O3: ", value=0.0, step=0.1)
        nO2 = st.number_input("Masukkan NO2: ", value=0.0, step=0.1)
        
        prediksi_cuaca = ""
        if st.button("Prediksi Udara"):
            data_pengguna = [[pM25, pM10, sO2, cO, o3, nO2]]

            # Pastikan untuk memanggil fit sebelum transform jika belum difit
            if not scaler._get_tags().get('fitted', False):
                scaler.fit(data_pengguna)

            data_pengguna_normalisasi = normalisasi_data(data_pengguna)
            prediksi_cuaca = model.predict(data_pengguna_normalisasi)

    with col9:
        if prediksi_cuaca == 0:
            st.title("Kualitas Udara: Sehat")
        elif prediksi_cuaca == 1:
            st.title("Kualitas Udara: Cukup Baik")
        elif prediksi_cuaca == 2:
            st.title("Kualitas Udara: Tidak Sehat")
            

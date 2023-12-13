import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Memuat model
try:
    model = pickle.load(open('catboost_model_revisi.pkl', 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan file ada dalam direktori yang benar atau ganti nama file sesuai dengan yang Anda miliki.")
    st.stop()

file_data = ""

# Membuka file css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html=True)

# Membuat tab
with st.sidebar:
    menuweb = st.radio("Menu Website", ["Udara di Jakarta", "Topik Diangkat", "App", "About"])

if menuweb == "Udara di Jakarta":
    st.write("oke")

if menuweb == "Topik Diangkat":
    st.write("oke")

if menuweb == "App":
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

        prediksi_udara=""
        if st.button("Prediksi Udara : "):
            prediksi_udara = model.predict([[pM25,pM10,sO2,cO,o3,nO2]])
    
    with col9:
        if prediksi_udara==0:
            st.write(prediksi_udara)
            st.write("Baik")
        elif prediksi_udara==1:
            st.write(prediksi_udara)
            st.write("Sedang")
        elif prediksi_udara==2:
            st.write(prediksi_udara)
            st.write("Tidak sehat")

        # # Membuat DataFrame dari input pengguna
        # df_user = pd.DataFrame({
        #     'PM2.5': [pM25],
        #     'PM10': [pM10],
        #     'SO2': [sO2],
        #     'CO': [cO],
        #     'O3': [o3],
        #     'NO2': [nO2]
        # })

        # # # Normalisasi data user (gunakan scaler yang telah di-fit pada data pelatihan)
        # # df_user_normalized = normalisasi_data(df_user)

        # prediksi_udara = ""
        # if st.button("Prediksi Udara"):
        #     prediksi_udara = prediksi_udara(df_user)

        # # Tampilkan hasil prediksi di dalam aplikasi
        # st.write(prediksi_udara)

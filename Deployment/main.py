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


# Fungsi untuk normalisasi data pengguna
scaler = MinMaxScaler()

def normalisasi_data(data_pengguna):
    return scaler.transform(data_pengguna)

# # Fungsi untuk prediksi udara
# def prediksi_udara(data_pengguna):
#     # Lakukan prediksi dengan model CatBoost yang telah dilatih
#     prediction = model.predict(data_pengguna)

#     # Tentukan hasil prediksi berdasarkan nilai yang diberikan oleh model
#     if prediction[0] == 1:
#         return "Hasil Prediksi: Baik"
#     elif prediction[0] == 2:
#         return "Hasil Prediksi: Sedang"
#     elif prediction[0] == 3:
#         return "Hasil Prediksi: Tidak Sehat"
#     else:
#         return "Error: Hasil prediksi tidak valid"

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
        pM25 = st.number_input("Masukkan PM25: ", value=0)
        pM10 = st.number_input("Masukkan PM10: ", value=0)
        sO2 = st.number_input("Masukkan SO2: ", value=0)
        cO = st.number_input("Masukkan CO: ", value=0)
        o3 = st.number_input("Masukkan O3: ", value=0)
        nO2 = st.number_input("Masukkan NO2: ", value=0)

        # Membuat DataFrame dari input pengguna
        df_user = pd.DataFrame({
            'PM2.5': [pM25],
            'PM10': [pM10],
            'SO2': [sO2],
            'CO': [cO],
            'O3': [o3],
            'NO2': [nO2]
        })

        # # Normalisasi data user (gunakan scaler yang telah di-fit pada data pelatihan)
        # df_user_normalized = normalisasi_data(df_user)

        prediksi_udara = ""
        if st.button("Prediksi Udara"):
            prediksi_udara = prediksi_udara(df_user)

# Tampilkan hasil prediksi di dalam aplikasi
st.write(prediksi_udara)

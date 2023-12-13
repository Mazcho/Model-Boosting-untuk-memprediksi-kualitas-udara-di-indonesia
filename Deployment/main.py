import numpy as np
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

# Memuat model
# Simpan model ke dalam file pickle
# Memuat kembali model dari file pickle
with open('catboost_model_revisi.pkl', 'rb') as model_file:
    loaded_catboost_model = pickle.load(model_file)

file_data1 = "pollutant-standards-index-southtangerang-2020-2022.csv"
file_data2 = "Data_SMOTE_normalisasi.csv"

file_data1salinan = pd.read_csv(file_data1)
file_data1salinan.drop(["Category","Date","Max","Critical Component"],axis=1,inplace=True)

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

        prediksi_udara = ""
        if st.button("Prediksi Udara"):

            x_user = pd.DataFrame({"PM2.5": [pM25],
                        "PM10": [pM10],
                        "SO2": [sO2],
                        "CO": [cO],
                        "O3": [o3],
                        "NO2": [nO2]})



            
            data_baru_user = np.vstack((file_data1salinan,x_user))
            data_baru_user = pd.DataFrame(data_baru_user)
            st.write(data_baru_user)

            # Data X yang akan dinormalisasi (kecuali 'IE EXP (%)')
            X_data_salinan = data_baru_user[[0,1,2,3,4,5]]

            # Inisialisasi MinMaxScaler untuk X
            scaler_X = MinMaxScaler()
            X_normalized_salinan = scaler_X.fit_transform(X_data_salinan)

            # Mengganti kolom-kolom dalam data_model dengan data yang sudah dinormalisasi
            data_baru_user[[0,1,2,3,4,5]] = X_normalized_salinan


            # Kolom 'IE EXP (%)' tidak perlu dinormalisasi
            # Sekarang Anda dapat menggabungkan data yang sudah dinormalisasi dengan 'IE EXP (%)'
            df_salinan_ku = data_baru_user



            df_coba_baru = df_salinan_ku.tail(1)
            st.write(df_coba_baru)

            # Memuat kembali model dari file pickle
            with open('catboost_model_revisi.pkl', 'rb') as model_file:
                    loaded_catboost_model = pickle.load(model_file)

            # Gunakan model yang telah dimuat kembali untuk membuat prediksi
            

            df_coba_baru.rename(columns={0:"PM2.5",1:"PM10",2:"SO2",3:"CO",4:"O3",5:"NO2"}, inplace=True)
            df_coba_baru

            y_new_pred = loaded_catboost_model.predict(df_coba_baru)
            st.write(y_new_pred)

            if y_new_pred==0:
                st.write(y_new_pred)
                st.write("Baik")
            elif y_new_pred==1:
                st.write(y_new_pred)
                st.write("Sedang")
            elif y_new_pred==2:
                st.write(y_new_pred)
                st.write("Tidak sehat")


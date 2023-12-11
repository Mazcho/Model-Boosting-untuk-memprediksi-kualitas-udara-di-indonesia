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



#membuat tab
with st.sidebar:
    menuweb = st.radio("Menu Website",["Ringkasan","Topik Diangkat","App","About"])
if menuweb == "Ringkasan":
    st.title("Polusi Udara di Indonesia yang kian memburuk")
    st.markdown("""Jakarta, CNBC Indonesia -  Tangerang Selatan dan Tangerang, Banten, menjadi dua wilayah kota dengan kualitas udara terburuk di Indonesia, bahkan beberapa kali melebihi DKI Jakarta. Menurut aplikasi penyedia data dan kualitas udara, Nafas Indonesia, hal itu diduga berasal dari dua sumber, yakni hyperlocal dan lintas batas. Apa maksudnya?
Co-Founder Nafas Indonesia, Piotr Jakubowski, mengatakan bahwa hyperlocal adalah sumber polusi yang berasal dari wilayah tercemar alias lokal, sementara lintas batas berasal dari wilayah lain di luar lokasi tercemar.

"Sumber hyperlocal adalah sesuatu yang terjadi di dekat lokasi kita, misalnya kawasan industri, mobil dan motor, pembakaran sampah oleh masyarakat, dan pabrik-pabrik yang emisinya tinggi," papar Piotr kepada CNBC Indonesia, Selasa (15/8/2023).""")

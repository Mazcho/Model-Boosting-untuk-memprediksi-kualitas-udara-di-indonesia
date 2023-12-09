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
    st.error("File CSV tidak ditemukan. Pastikan file ada dalam direktori yang benar atau ganti nama file sesuai dengan yang Anda miliki.")
    st.stop()


st.title("Pollutan Standart Index Clasification with CatBoost")
#membuat tab
with st.sidebar:
    menuweb = st.radio("Menu Website",["Home","App","About"])

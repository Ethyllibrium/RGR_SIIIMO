import streamlit as st
import tensorflow as tf

import pandas as pd
import numpy  as np
import pickle
import json

import sklearn
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

with open('scalerNormX.pickle', mode='rb') as filestream:
    scalerNormX = pickle.load(filestream)
with open('scalerNormY.pickle', mode='rb') as filestream:
    scalerNormY = pickle.load(filestream)

with open('linear_model.pickle', mode='rb') as filestream:
    lin_model = pickle.load(filestream)
nn_model = tf.keras.models.load_model(   
     r"neural_model.h5")

with open('linear_model_description.json', mode='r') as filestream:
    m1_dict = json.load(filestream)
with open('neural_model_description.json', mode='r') as filestream:
    m2_dict = json.load(filestream)


st.sidebar.write("Введите параметры ниже:")
cement = st.sidebar.number_input("Цемент (кг/м^3)", min_value=scalerNormX.data_min_[0], max_value=scalerNormX.data_max_[0], value = scalerNormX.data_min_[0], step=0.5)
slag = st.sidebar.number_input("Шлак (кг/м^3)", min_value=scalerNormX.data_min_[1], max_value=scalerNormX.data_max_[1], value = scalerNormX.data_min_[1], step=0.5)
ash = st.sidebar.number_input("Зола (кг/м^3)", min_value=scalerNormX.data_min_[2], max_value=scalerNormX.data_max_[2], value = scalerNormX.data_min_[2], step=0.5)
water = st.sidebar.number_input("Вода (кг/м^3)", min_value=scalerNormX.data_min_[3], max_value=scalerNormX.data_max_[3], value = scalerNormX.data_min_[3], step=0.5)
super = st.sidebar.number_input("Суперпластификатор (кг/м^3)", min_value=scalerNormX.data_min_[4], max_value=scalerNormX.data_max_[4], value = scalerNormX.data_min_[4], step=0.5)
coarse = st.sidebar.number_input("Крупный заполнитель (кг/м^3)", min_value=scalerNormX.data_min_[5], max_value=scalerNormX.data_max_[5], value = scalerNormX.data_min_[5], step=0.5)
fine = st.sidebar.number_input("Мелкий заполнитель (кг/м^3)", min_value=scalerNormX.data_min_[6], max_value=scalerNormX.data_max_[6], value = scalerNormX.data_min_[6], step=0.5)
age = st.sidebar.number_input("Возраст бетона (дней)", min_value=scalerNormX.data_min_[7], max_value=scalerNormX.data_max_[7], value = scalerNormX.data_min_[7], step=0.5)

dfx_custom = pd.DataFrame(data=[[cement, slag, ash, water, super, coarse, fine, age]], columns=scalerNormX.feature_names_in_)
dfx_custom
col1, col2 = st.columns([2,2])
col1.header("Линейная регрессия")
col1.write(f"R^2 = {m1_dict['R2']}")
col1.write(f"RMSE = {m1_dict['RMSE']}")
col1.write(f"Прочность бетона равна:\n{lin_model.predict(dfx_custom)[0][0]:.3} МПа")

col2.header("Нейронная сеть")
col2.write(f"R^2 = {m2_dict['R2']}")
col2.write(f"RMSE = {m2_dict['RMSE']}")
nnY = nn_model.predict(scalerNormX.transform(dfx_custom))
col2.write(f"Прочность бетона равна:")
col2.write(f"{nnY[0][0]:.3} (Нормализованное значение)")
col2.write(f"{scalerNormY.inverse_transform(nnY)[0][0]:.3} Мпа (В исходной шкале)")
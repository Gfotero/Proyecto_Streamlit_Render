import streamlit as st
import pandas as pd
from pickle import load

# Cargar el modelo
model = load(open("../models/random_forest_regressor_42_sin_scal.sav", "rb"))

# Definir la interfaz de usuario
st.title("Predicción de Precio de Viviendas")
st.write("Ingresa los valores de las características:")

crim = st.number_input("CRIM", min_value=0.0, max_value=100.0, value=0.0)
rm = st.number_input("RM", min_value=1.0, max_value=10.0, value=5.0)
dis = st.number_input("DIS", min_value=0.0, max_value=10.0, value=1.0)
lstat = st.number_input("LSTAT", min_value=0.0, max_value=50.0, value=10.0)

# Crear un DataFrame con los valores ingresados
data = pd.DataFrame([[crim, rm, dis, lstat]], columns=['CRIM', 'RM', 'DIS', 'LSTAT'])

# Realizar la predicción
prediction = model.predict(data)[0]
pred_class = f"{prediction:.3f} M$"

# Mostrar el resultado
st.write(f"La predicción del precio de la vivienda es: {pred_class}")

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("MPG Prediction App")
st.write("Enter car specifications to predict MPG (Miles Per Gallon).")

# User inputs
cylinders = st.slider("Cylinders", 3, 12, 4)
displacement = st.number_input("Displacement", 60.0, 500.0, 150.0)
horsepower = st.number_input("Horsepower", 40.0, 250.0, 100.0)
weight = st.number_input("Weight (lbs)", 1500.0, 5500.0, 3000.0)
acceleration = st.number_input("Acceleration (0-60 mph)", 8.0, 25.0, 15.0)
model_year = st.slider("Model Year", 70, 82, 76)
origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x])

# Make prediction
input_data = np.array([cylinders, displacement, horsepower, weight, acceleration, model_year, origin]).reshape(1,-1)
input_scaled = scaler.transform(input_data)
#user_input = sc.fit(np.array([cylinders, displacement, horsepower, weight, acceleration, model_year, origin]).reshape(1,-1))
prediction = model.predict(input_scaled)

# Display prediction
st.subheader("Predicted MPG:")
st.success(f"{prediction[0]:.2f}")

# Optional: Add a plot
import matplotlib.pyplot as plt

st.subheader("📊 Visualization")
fig, ax = plt.subplots()
ax.bar(['Predicted MPG'], [prediction[0]], color='skyblue')
ax.set_ylabel("Miles per Gallon")
st.pyplot(fig)

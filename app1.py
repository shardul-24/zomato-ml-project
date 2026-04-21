import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Title
st.title(" Restaurant Rating Prediction System")

# -----------------------------
# Load and train model
# -----------------------------
df = pd.read_csv("Zomato-data-.csv")

# Clean rate column
df['rate'] = df['rate'].str.replace('/5', '')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df.dropna(inplace=True)

# Drop useless column
df.drop(['name'], axis=1, inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop('rate', axis=1)
y = df['rate']

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("Enter Restaurant Details")

votes = st.sidebar.slider("Votes", 0, 1000, 100)
cost = st.sidebar.slider("Cost for Two", 100, 3000, 500)
online_order = st.sidebar.selectbox("Online Order", ["Yes", "No"])
rest_type = st.sidebar.selectbox("Restaurant Type", ["Buffet", "Cafes", "Dining", "other"])

# Convert input to match model
input_dict = {
    'votes': votes,
    'approx_cost(for two people)': cost,
    'online_order_Yes': 1 if online_order == "Yes" else 0,
    'listed_in(type)_Buffet': 1 if rest_type == "Buffet" else 0,
    'listed_in(type)_Cafes': 1 if rest_type == "Cafes" else 0,
    'listed_in(type)_Dining': 1 if rest_type == "Dining" else 0,
    'listed_in(type)_other': 1 if rest_type == "other" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Align columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted Rating: {round(prediction,2)} / 5")
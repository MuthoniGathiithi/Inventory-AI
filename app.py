# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data import data  # your 10-row dataset

# --- Load and prepare data ---
df = pd.DataFrame(data)
df_numeric = pd.get_dummies(df, columns=['weather', 'day_type', 'special_event'])

X = df_numeric.drop('total_sold', axis=1)
y = df_numeric['total_sold']

# --- Train model ---
Model = RandomForestRegressor(n_estimators=100, random_state=42)
Model.fit(X, y)

# --- Streamlit UI ---
st.title("Inventory AI Predictor")

day = st.number_input("Day", min_value=1, max_value=365, value=11)
weather = st.selectbox("Weather", ["Cold", "Hot", "Sunny"])
day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
event = st.selectbox("Special Event", ["None", "Promotion", "Holiday"])

# --- Prepare input in one-hot format ---
input_df = pd.DataFrame([{
    'day': day,
    'weather_Cold': weather == "Cold",
    'weather_Hot': weather == "Hot",
    'weather_Sunny': weather == "Sunny",
    'day_type_Weekday': day_type == "Weekday",
    'day_type_Weekend': day_type == "Weekend",
    'special_event_Holiday': event == "Holiday",
    'special_event_None': event == "None",
    'special_event_Promotion': event == "Promotion"
}])

# --- Predict ---
prediction = int(round(Model.predict(input_df)[0]))
st.write(f"Predicted Total Sold: {prediction}")
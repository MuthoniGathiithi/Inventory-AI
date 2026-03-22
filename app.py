import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import gradio as gr
from data import data

# Prepare the data
df = pd.DataFrame(data)
df_numeric = pd.get_dummies(df, columns=['weather', 'day_type', 'special_event'])

X = df_numeric.drop('total_sold', axis=1)
y = df_numeric['total_sold']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict
def predict_sales(day, weather_Cold, weather_Hot, weather_Sunny,
                  day_type_Weekday, day_type_Weekend,
                  special_event_Holiday, special_event_None, special_event_Promotion):
    new_day = pd.DataFrame([{
        'day': day,
        'weather_Cold': weather_Cold,
        'weather_Hot': weather_Hot,
        'weather_Sunny': weather_Sunny,
        'day_type_Weekday': day_type_Weekday,
        'day_type_Weekend': day_type_Weekend,
        'special_event_Holiday': special_event_Holiday,
        'special_event_None': special_event_None,
        'special_event_Promotion': special_event_Promotion
    }])
    prediction = model.predict(new_day)
    return int(round(prediction[0]))

# Build Gradio UI
inputs = [
    gr.Number(label="Day"),
    gr.Checkbox(label="Weather Cold"),
    gr.Checkbox(label="Weather Hot"),
    gr.Checkbox(label="Weather Sunny"),
    gr.Checkbox(label="Day type Weekday"),
    gr.Checkbox(label="Day type Weekend"),
    gr.Checkbox(label="Special Event Holiday"),
    gr.Checkbox(label="Special Event None"),
    gr.Checkbox(label="Special Event Promotion")
]

output = gr.Number(label="Predicted Sales")

gr.Interface(fn=predict_sales, inputs=inputs, outputs=output, title="Inventory AI").launch()
```python
import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🍓",
    layout="wide"
)

# Custom CSS — Baby Pink Background + Strawberry Decorations
st.markdown(
    """
    <style>

    /* Baby pink background */
    .stApp {
        background-color: #FDE7F0;
    }

    /* Main content */
    .main {
        background-color: #FDE7F0;
    }

    /* Title */
    h1 {
        color: #C2185B;
        text-align: center;
        font-family: Georgia, serif;
        font-weight: bold;
    }

    /* Normal text */
    p, label {
        color: #6D214F !important;
    }

    /* Strawberry decorations */
    .strawberry {
        position: fixed;
        font-size: 55px;
        z-index: 999;
    }

    .strawberry1 {
        top: 20px;
        left: 25px;
    }

    .strawberry2 {
        top: 20px;
        right: 25px;
    }

    .strawberry3 {
        bottom: 20px;
        left: 25px;
    }

    .strawberry4 {
        bottom: 20px;
        right: 25px;
    }

    /* Button */
    .stButton > button {
        background-color: #E75480;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #C2185B;
        color: white;
    }

    /* Input boxes */
    div[data-baseweb="select"] > div {
        background-color: #FFF5F8;
    }

    </style>

    <div class="strawberry strawberry1">🍓</div>
    <div class="strawberry strawberry2">🍓</div>
    <div class="strawberry strawberry3">🍓</div>
    <div class="strawberry strawberry4">🍓</div>
    """,
    unsafe_allow_html=True
)

# Load the trained model
logi = joblib.load('logi.sav')

# App title
st.title('🍓 Delivery Delay Prediction App')

st.write(
    'Enter the details below to predict if there will be a delivery delay.'
)

# Input features
delivery_distance = st.slider(
    'Delivery Distance (km)', 1.0, 50.0, 25.0
)

traffic_congestion = st.selectbox(
    'Traffic Congestion (1-5, 5=High)',
    [1, 2, 3, 4, 5]
)

weather_condition = st.selectbox(
    'Weather Condition (1-5, 5=Bad)',
    [1, 2, 3, 4, 5]
)

delivery_slot = st.selectbox(
    'Delivery Slot (1-3)',
    [1, 2, 3]
)

driver_experience = st.slider(
    'Driver Experience (years)', 0, 20, 5
)

num_stops = st.slider(
    'Number of Stops', 1, 10, 3
)

vehicle_age = st.slider(
    'Vehicle Age (years)', 0, 15, 5
)

road_condition_score = st.selectbox(
    'Road Condition Score (1-5, 5=Good)',
    [1, 2, 3, 4, 5]
)

package_weight = st.slider(
    'Package Weight (kg)', 0.1, 150.0, 50.0
)

fuel_efficiency = st.slider(
    'Fuel Efficiency (km/l)', 5.0, 25.0, 15.0
)

warehouse_processing_time = st.slider(
    'Warehouse Processing Time (minutes)', 1, 120, 60
)

# Create DataFrame for input
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

# Prediction
if st.button('🍓 Predict Delivery Delay'):

    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(
            '⚠️ Prediction: **Delivery Delay is Likely!**'
        )
    else:
        st.success(
            '✅ Prediction: **No Delivery Delay is Expected.**'
        )

    st.write(
        f'🍓 Probability of Delay: '
        f'**{prediction_proba[0][1]:.2f}**'
    )

    st.write(
        f'🍓 Probability of No Delay: '
        f'**{prediction_proba[0][0]:.2f}**'
    )
```

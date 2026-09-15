
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🍓",
    layout="wide"
)

# =========================
# CUSTOM DESIGN
# =========================
st.markdown("""
<style>

    /* Entire app background */
    [data-testid="stAppViewContainer"] {
        background-color: #FDE6EF;
    }

    /* Sidebar background if present */
    [data-testid="stSidebar"] {
        background-color: #F8C8DC;
    }

    /* Main content */
    [data-testid="stMain"] {
        background-color: #FDE6EF;
    }

    /* Title */
    h1 {
        color: #C2185B !important;
        text-align: center;
        font-family: Georgia, serif;
        font-size: 42px !important;
    }

    /* Text */
    p {
        color: #6D214F;
    }

    /* Button */
    .stButton > button {
        background-color: #E75480;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #C2185B;
        color: white;
    }

    /* Strawberry decoration */
    .strawberry {
        text-align: center;
        font-size: 45px;
        margin: 0px;
    }

    /* Prediction boxes */
    [data-testid="stAlert"] {
        border-radius: 15px;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# STRAWBERRY HEADER
# =========================

st.markdown(
    '<div class="strawberry">🍓 🍓 🍓 🍓 🍓</div>',
    unsafe_allow_html=True
)


# =========================
# LOAD MODEL
# =========================

logi = joblib.load('logi.sav')


# =========================
# TITLE
# =========================

st.title('Delivery Delay Prediction App')

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Enter the details below to predict if there will be a delivery delay. 🍓"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")


# =========================
# INPUT FEATURES
# =========================

delivery_distance = st.slider(
    'Delivery Distance (km)',
    1.0, 50.0, 25.0
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
    'Driver Experience (years)',
    0, 20, 5
)

num_stops = st.slider(
    'Number of Stops',
    1, 10, 3
)

vehicle_age = st.slider(
    'Vehicle Age (years)',
    0, 15, 5
)

road_condition_score = st.selectbox(
    'Road Condition Score (1-5, 5=Good)',
    [1, 2, 3, 4, 5]
)

package_weight = st.slider(
    'Package Weight (kg)',
    0.1, 150.0, 50.0
)

fuel_efficiency = st.slider(
    'Fuel Efficiency (km/l)',
    5.0, 25.0, 15.0
)

warehouse_processing_time = st.slider(
    'Warehouse Processing Time (minutes)',
    1, 120, 60
)


# =========================
# CREATE INPUT DATA
# =========================

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


# =========================
# PREDICTION
# =========================

if st.button('🍓 Predict Delivery Delay'):

    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)

    if prediction[0] == 1:

        st.error(
            "⚠️ **Delivery Delay is Likely!**"
        )

    else:

        st.success(
            "✅ **No Delivery Delay is Expected.**"
        )

    st.write(
        f"🍓 Probability of Delay: "
        f"**{prediction_proba[0][1]:.2f}**"
    )

    st.write(
        f"🍓 Probability of No Delay: "
        f"**{prediction_proba[0][0]:.2f}**"
    )


# =========================
# FOOTER STRAWBERRIES
# =========================

st.markdown("---")

st.markdown(
    '<div class="strawberry">🍓 🍓 🍓</div>',
    unsafe_allow_html=True
)
```

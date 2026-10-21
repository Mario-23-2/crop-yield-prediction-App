
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = "crop_yield_prediction_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place the model in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f4f8f2;
    }

    .main-title {
        text-align: center;
        color: #1b5e20;
        font-size: 38px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        color: #607d66;
        font-size: 17px;
        margin-bottom: 25px;
    }

    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        height: 48px;
        font-size: 17px;
        font-weight: bold;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }

    .result-card {
        background-color: #e0f2e3;
        border-left: 6px solid #2e7d32;
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
    }

    .result-value {
        color: #1b5e20;
        font-size: 36px;
        font-weight: 800;
    }

    .result-label {
        color: #456b4a;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🌱 Crop Yield Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Powered Agricultural Yield Estimation'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the crop, soil, weather, and farming details "
    "to estimate the expected yield."
)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("🌾 Crop and Environmental Details")

with st.form("crop_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Region",
            ["North", "South", "East", "West", "Central"]
        )

        weather = st.selectbox(
            "Weather Condition",
            ["Sunny", "Rainy", "Cloudy", "Stormy"]
        )

        crop = st.selectbox(
            "Crop Type",
            ["Wheat", "Rice", "Maize", "Cotton",
             "Barley", "Soybean", "Sugarcane"]
        )

        soil = st.selectbox(
            "Soil Type",
            ["Sandy", "Clay", "Loamy", "Silt", "Peaty",
             "Chalky"]
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.5
        )

    with col2:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        fertilizer = st.selectbox(
            "Fertilizer Used",
            ["Yes", "No"]
        )

        irrigation = st.selectbox(
            "Irrigation Used",
            ["Yes", "No"]
        )

        days_to_harvest = st.number_input(
            "Days to Harvest",
            min_value=1,
            max_value=500,
            value=100,
            step=1
        )

    submitted = st.form_submit_button(
        "🌱 Predict Crop Yield"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if submitted:

    # Build input using EXACT feature names used in training
    input_data = pd.DataFrame([{
        "Region": region,
        "Weather_Condition": weather,
        "Crop": crop,
        "Soil_Type": soil,
        "Temperature_Celsius": temperature,
        "Rainfall_mm": rainfall,
        "Fertilizer_Used": fertilizer == "Yes",
        "Irrigation_Used": irrigation == "Yes",
        "Days_to_Harvest": days_to_harvest
    }])

    try:
        prediction = model.predict(input_data)[0]

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")

        elif prediction < 0:
            st.warning(
                f"The model predicted {prediction:.2f} tons/hectare. "
                "This is a negative estimate and should not be "
                "interpreted as a realistic crop yield."
            )

        else:
            st.success("Prediction completed successfully!")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Crop Yield
                    </div>
                    <div class="result-value">
                        {prediction:.2f} tons/hectare
                    </div>
                    <div class="result-label">
                        Estimated yield for the selected conditions
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display the selected inputs
            st.subheader("📋 Input Summary")

            st.dataframe(
                input_data.rename(columns={
                    "Region": "Region",
                    "Weather_Condition": "Weather",
                    "Temperature_Celsius": "Temperature (°C)",
                    "Rainfall_mm": "Rainfall (mm)",
                    "Fertilizer_Used": "Fertilizer Used",
                    "Irrigation_Used": "Irrigation Used",
                    "Days_to_Harvest": "Days to Harvest"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This is a machine learning estimate, not a "
                "guarantee of actual farm yield."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Crop Yield Prediction System | "
    "Powered by Python, Streamlit and Scikit-learn"
)

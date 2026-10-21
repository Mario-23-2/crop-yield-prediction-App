
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# VERDANT | CROP YIELD INTELLIGENCE
# =========================================================

st.set_page_config(
    page_title="VERDANT | Crop Yield Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "crop_yield_prediction_model.pkl"

FEATURES = [
    "Region",
    "Weather_Condition",
    "Crop",
    "Soil_Type",
    "Temperature_Celsius",
    "Rainfall_mm",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Days_to_Harvest",
]

CATEGORICAL_FEATURES = [
    "Region",
    "Weather_Condition",
    "Crop",
    "Soil_Type",
    "Fertilizer_Used",
    "Irrigation_Used",
]

# =========================================================
# DESIGN SYSTEM
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --forest: #092e25;
        --forest-light: #104838;
        --green: #25894e;
        --lime: #75c957;
        --sage: #dff2e7;
        --cream: #f5f8f5;
        --ink: #143d38;
        --muted: #718780;
        --border: #e1ebe5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f1f7f5, #f9fbf9);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 249, 246, 0.9);
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06271f, #0b3a2d 65%, #072820);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f4fff8;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.16);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #a8dfc4;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 11px 12px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid transparent;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(86, 181, 111, 0.2);
    }

    .brand {
        padding: 10px 3px 22px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        font-size: 38px;
    }

    .brand-name {
        font-size: 23px;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }

    .brand-tagline {
        color: #a5e4c4;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 18px;
    }

    /* HERO */
    .hero {
        position: relative;
        overflow: hidden;
        min-height: 255px;
        border-radius: 18px;
        padding: 34px;
        margin-bottom: 22px;
        background:
            linear-gradient(
                90deg,
                rgba(230, 246, 247, 0.97) 0%,
                rgba(231, 245, 239, 0.91) 43%,
                rgba(244, 248, 225, 0.18) 100%
            ),
            linear-gradient(
                180deg,
                #d9edf5 0%,
                #e8f1db 48%,
                #77ad70 72%,
                #286b42 100%
            );
        box-shadow: 0 8px 24px rgba(19, 69, 52, 0.08);
    }

    .hero::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 40%;
        background:
            repeating-linear-gradient(
                175deg,
                transparent 0 18px,
                rgba(255,255,255,0.12) 19px 21px
            ),
            linear-gradient(
                180deg,
                rgba(65,133,64,0.08),
                rgba(24,91,47,0.65)
            );
        clip-path: polygon(
            0 45%, 13% 24%, 25% 43%, 39% 13%,
            52% 37%, 66% 19%, 79% 42%, 91% 12%,
            100% 28%, 100% 100%, 0 100%
        );
    }

    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 720px;
    }

    .eyebrow {
        color: #22574a;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(30px, 3.5vw, 46px);
        font-weight: 800;
        color: #103d38;
        line-height: 1.13;
        margin: 0 0 13px;
    }

    .hero h1 span {
        color: #278c4c;
    }

    .hero p {
        font-size: 14px;
        line-height: 1.7;
        color: #315c55;
        margin: 0;
    }

    .hero-note {
        position: absolute;
        z-index: 1;
        right: 6%;
        top: 25%;
        text-align: center;
        color: #17483b;
        font-size: 17px;
        font-weight: 700;
        line-height: 1.65;
        transform: rotate(-7deg);
    }

    /* CARDS */
    .panel {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        margin-bottom: 16px;
        box-shadow: 0 5px 20px rgba(16, 61, 47, 0.045);
    }

    .panel-heading {
        font-size: 18px;
        font-weight: 800;
        color: #153e38;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    .form-heading {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(100deg, #effaf3, #f8fbf8);
        border-bottom: 1px solid #e4eee7;
        padding: 16px 19px;
        margin: -19px -19px 19px;
        border-radius: 17px 17px 0 0;
    }

    .form-icon {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #c8f2dc;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        flex-shrink: 0;
    }

    /* METRICS */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 15px 17px;
        box-shadow: 0 4px 14px rgba(16, 61, 47, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #718780;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #21854b;
        font-weight: 800;
    }

    /* INPUTS */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #fbfdfb;
        border-color: #dce8e1;
        border-radius: 10px;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    /* BUTTONS */
    .stButton button,
    .stFormSubmitButton button,
    .stDownloadButton button {
        background: linear-gradient(90deg, #70c653, #24894e);
        color: white !important;
        border: none;
        border-radius: 11px;
        min-height: 43px;
        font-weight: 800;
        box-shadow: 0 5px 14px rgba(35, 137, 76, 0.12);
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover,
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #57b440, #176c3d);
        border: none;
    }

    /* RESULT CARD */
    .yield-card {
        background: linear-gradient(135deg, #effaf3, #e3f6ed);
        border: 1px solid #d4ecdd;
        border-radius: 17px;
        padding: 20px;
        margin: 10px 0 15px;
    }

    .yield-label {
        font-size: 11px;
        color: #547b6d;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .yield-value {
        font-size: clamp(32px, 3vw, 44px);
        font-weight: 800;
        line-height: 1.3;
        color: #103b35;
        margin: 6px 0;
    }

    .yield-unit {
        font-size: 14px;
        font-weight: 600;
        color: #315c55;
    }

    .status-pill {
        display: inline-block;
        background: #d0f4dc;
        color: #187747;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
    }

    .tip-card {
        background: linear-gradient(120deg, #f8fcf8, #edf8f1);
        border: 1px solid #d9eee0;
        border-radius: 15px;
        padding: 16px;
        font-size: 12px;
        line-height: 1.9;
        color: #315c55;
    }

    .feature-strip {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 19px;
        box-shadow: 0 4px 18px rgba(16, 61, 47, 0.035);
        margin-top: 8px;
    }

    .feature-title {
        color: #153e38;
        font-size: 12px;
        font-weight: 800;
    }

    .feature-caption {
        color: #718780;
        font-size: 10px;
        margin-top: 4px;
    }

    .footer {
        text-align: center;
        color: #81948b;
        font-size: 11px;
        padding: 20px 0 5px;
    }

    @media (max-width: 850px) {
        .hero { padding: 24px; }
        .hero-note { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_categories(loaded_model):
    """Read category options from the fitted preprocessing pipeline."""
    preprocessor = loaded_model.named_steps["preprocessor"]
    categorical = preprocessor.named_transformers_["categorical"]

    # Support a pipeline nested inside the categorical transformer.
    if not hasattr(categorical, "categories_"):
        for step in reversed(
            list(categorical.named_steps.values())
        ):
            if hasattr(step, "categories_"):
                categorical = step
                break

    if not hasattr(categorical, "categories_"):
        raise ValueError(
            "Could not find the fitted OneHotEncoder. "
            "Check your saved model's preprocessing pipeline."
        )

    return {
        name: [
            item.item() if hasattr(item, "item") else item
            for item in values
        ]
        for name, values in zip(
            CATEGORICAL_FEATURES,
            categorical.categories_,
        )
    }


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Keep app.py and the model file in the same folder."
    )
    st.stop()

try:
    model = load_model()
    categories = get_categories(model)
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


def pretty_category(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def category_selector(label, values, key):
    return st.selectbox(
        label,
        values,
        format_func=pretty_category,
        key=key,
    )


def show_hero():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-content">
            <div class="eyebrow">CROP YIELD PREDICTION</div>
            <h1>Data-Driven Decisions<br>
            for a <span>Better Harvest</span></h1>
            <p>
              Enter your crop details and environmental factors
              to predict expected yield and make smarter farming decisions.
            </p>
          </div>
          <div class="hero-note">
            Healthy Soil<br>
            Happy Crops ♡<br>
            Higher Yields
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_prediction(values):
    data = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(data)[0])

    st.session_state["last_prediction"] = prediction
    st.session_state["last_crop"] = str(values["Crop"])
    st.session_state["last_inputs"] = values

    history = st.session_state.setdefault(
        "prediction_history", []
    )
    history.insert(
        0,
        {
            "Time": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            ),
            "Crop": str(values["Crop"]),
            "Predicted yield (tons/ha)": round(prediction, 3),
        },
    )
    st.session_state["prediction_history"] = history[:50]

    return prediction


def prediction_report():
    prediction = st.session_state["last_prediction"]
    inputs = st.session_state["last_inputs"]

    lines = [
        "VERDANT - Crop Yield Prediction Report",
        f"Generated: {datetime.now():%d %b %Y, %I:%M %p}",
        "",
        f"Crop: {inputs['Crop']}",
        f"Predicted yield: {prediction:.3f} tons per hectare",
        "",
        "INPUT DETAILS",
    ]

    lines.extend(
        f"{key}: {value}" for key, value in inputs.items()
    )
    lines.extend(
        [
            "",
            "Note: This is a machine-learning estimate, "
            "not a guarantee of actual harvest.",
        ]
    )
    return "\n".join(lines)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-row">
            <div class="brand-icon">🌿</div>
            <div class="brand-name">VERDANT<br>Crop Intelligence</div>
          </div>
          <div class="brand-tagline">
            Smarter Farming<br>
            for a Greener Tomorrow 🌱
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Predict Yield",
            "Farming Assistant",
            "Insights",
            "About the Model",
        ],
        label_visibility="collapsed",
        format_func=lambda value: {
            "Home": "🏠   Home",
            "Predict Yield": "🌱   Predict Yield",
            "Farming Assistant": "💬   Farming Assistant",
            "Insights": "📊   Insights",
            "About the Model": "ⓘ   About the Model",
        }[value],
        key="navigation",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="padding:12px 2px;line-height:1.9;font-size:13px">
          <div style="font-size:17px;font-weight:800">
            🌿 Our purpose
          </div>
          Helping farmers make informed decisions with machine learning.
          <br><br>
          <i>“Better predictions.<br>
          Healthier crops.<br>
          A sustainable future.”</i> 🌱
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Data-driven farming · Greener tomorrow")


# =========================================================
# HOME DASHBOARD
# =========================================================

if page == "Home":
    show_hero()

    col1, col2, col3 = st.columns(3)
    col1.metric("🌿 Prediction engine", "Ready")
    col2.metric("🧪 Model input features", "9")
    col3.metric("📈 Output unit", "Tons / hectare")

    st.write("")

    left, right = st.columns([1.45, 1], gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel">
              <div class="form-heading">
                <div class="form-icon">🌱</div>
                <div>
                  <div class="panel-heading">Enter Crop Details</div>
                  <div class="panel-subtitle">
                    Complete the form to estimate crop yield.
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("home_prediction"):
            c1, c2 = st.columns(2)

            with c1:
                region = category_selector(
                    "📍 Region", categories["Region"], "home_region"
                )
                crop = category_selector(
                    "🌱 Crop", categories["Crop"], "home_crop"
                )
                soil = category_selector(
                    "🪴 Soil type", categories["Soil_Type"], "home_soil"
                )
                rainfall = st.number_input(
                    "🌧️ Rainfall (mm)",
                    min_value=0.0,
                    value=100.0,
                    step=10.0,
                )
                temperature = st.number_input(
                    "🌡️ Temperature (°C)",
                    value=25.0,
                    step=1.0,
                )

            with c2:
                weather = category_selector(
                    "⛅ Weather condition",
                    categories["Weather_Condition"],
                    "home_weather",
                )
                fertilizer = category_selector(
                    "🧪 Fertilizer used",
                    categories["Fertilizer_Used"],
                    "home_fertilizer",
                )
                irrigation = category_selector(
                    "💧 Irrigation used",
                    categories["Irrigation_Used"],
                    "home_irrigation",
                )
                days = st.number_input(
                    "📅 Days to harvest",
                    min_value=1,
                    value=90,
                    step=1,
                )

                st.caption(
                    "The model uses all nine inputs, including "
                    "region, crop, soil, weather and farm conditions."
                )

            submitted = st.form_submit_button(
                "✨  Predict Yield   →",
                use_container_width=True,
            )

        if submitted:
            values = {
                "Region": region,
                "Weather_Condition": weather,
                "Crop": crop,
                "Soil_Type": soil,
                "Temperature_Celsius": temperature,
                "Rainfall_mm": rainfall,
                "Fertilizer_Used": fertilizer,
                "Irrigation_Used": irrigation,
                "Days_to_Harvest": days,
            }

            try:
                result = make_prediction(values)
                st.success(
                    f"Prediction ready: {result:.2f} tons per hectare."
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    with right:
        st.markdown(
            '<div class="panel-heading">🏅 Predicted Yield</div>',
            unsafe_allow_html=True,
        )

        result = st.session_state.get("last_prediction")
        crop_name = st.session_state.get("last_crop", "Your crop")

        if result is None:
            st.markdown(
                """
                <div class="yield-card">
                  <div class="yield-label">YOUR HARVEST ESTIMATE</div>
                  <div style="display:flex;align-items:center;gap:15px">
                    <div style="font-size:48px">🌱</div>
                    <div>
                      <div style="font-size:22px;font-weight:800;
                                  color:#103b35">
                        Ready when you are
                      </div>
                      <div class="yield-unit">tons per hectare</div>
                    </div>
                  </div>
                  <p style="color:#547b6d;font-size:13px">
                    Complete the farm details and predict your yield.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="yield-card">
                  <div class="yield-label">
                    PREDICTED YIELD · {crop_name}
                  </div>
                  <div style="display:flex;align-items:center;gap:14px">
                    <div style="font-size:48px">🌿</div>
                    <div>
                      <div class="yield-value">
                        {result:,.2f}
                        <span class="yield-unit">tons/ha</span>
                      </div>
                      <span class="status-pill">✓ Estimate ready</span>
                    </div>
                  </div>
                  <p style="color:#315c55;font-size:13px">
                    This estimate is based on the values you entered.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                "⬇️ Download prediction report",
                data=prediction_report(),
                file_name="verdant_prediction_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="panel">
              <div class="panel-heading">📈 Yield Prediction Trend</div>
              <div class="panel-subtitle">
                Predictions made during this app session.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        history = st.session_state.get("prediction_history", [])

        if history:
            chart = pd.DataFrame(history[::-1])
            chart = chart.rename(
                columns={"Predicted yield (tons/ha)": "Yield"}
            )
            st.bar_chart(
                chart.set_index("Crop")["Yield"],
                color="#55b878",
                height=190,
            )
        else:
            st.info(
                "Your chart will appear after you make your first prediction."
            )

        st.markdown(
            """
            <div class="tip-card">
              <b>💡 Quick Farming Tips</b><br>
              ✓ Use realistic local measurements.<br>
              ✓ Consider rainfall and irrigation together.<br>
              ✓ Match the soil and crop categories to your farm.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        """
        <div class="feature-strip">
          <div style="display:flex;justify-content:space-around;
                      gap:18px;flex-wrap:wrap">
            <div>🌿 <span class="feature-title">Accurate Predictions</span>
              <div class="feature-caption">Machine-learning estimates</div>
            </div>
            <div>◷ <span class="feature-title">Save Resources</span>
              <div class="feature-caption">Make informed decisions</div>
            </div>
            <div>🌱 <span class="feature-title">Improve Productivity</span>
              <div class="feature-caption">Plan each growing season</div>
            </div>
            <div>🌍 <span class="feature-title">Sustainable Farming</span>
              <div class="feature-caption">Grow a greener future</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PREDICT YIELD PAGE
# =========================================================

elif page == "Predict Yield":
    show_hero()

    st.markdown(
        '<div class="panel-heading">🌾 Predict Your Crop Yield</div>',
        unsafe_allow_html=True,
    )
    st.caption("Enter all nine fields required by your trained model.")

    with st.form("full_prediction"):
        c1, c2, c3 = st.columns(3)

        with c1:
            region = category_selector(
                "Region", categories["Region"], "predict_region"
            )
            crop = category_selector(
                "Crop", categories["Crop"], "predict_crop"
            )
            soil = category_selector(
                "Soil type", categories["Soil_Type"], "predict_soil"
            )

        with c2:
            weather = category_selector(
                "Weather condition",
                categories["Weather_Condition"],
                "predict_weather",
            )
            fertilizer = category_selector(
                "Fertilizer used",
                categories["Fertilizer_Used"],
                "predict_fertilizer",
            )
            irrigation = category_selector(
                "Irrigation used",
                categories["Irrigation_Used"],
                "predict_irrigation",
            )

        with c3:
            temperature = st.number_input(
                "Temperature (°C)", value=25.0, step=1.0
            )
            rainfall = st.number_input(
                "Rainfall (mm)", min_value=0.0, value=100.0, step=10.0
            )
            days = st.number_input(
                "Days to harvest", min_value=1, value=90, step=1
            )

        submitted = st.form_submit_button(
            "✨ Predict Yield", use_container_width=True
        )

    if submitted:
        values = {
            "Region": region,
            "Weather_Condition": weather,
            "Crop": crop,
            "Soil_Type": soil,
            "Temperature_Celsius": temperature,
            "Rainfall_mm": rainfall,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Days_to_Harvest": days,
        }

        try:
            result = make_prediction(values)
            st.success(
                f"Estimated yield for {crop}: "
                f"{result:.2f} tons per hectare."
            )
            st.download_button(
                "⬇️ Download prediction report",
                prediction_report(),
                "verdant_prediction_report.txt",
                "text/plain",
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# FARMING ASSISTANT
# =========================================================

elif page == "Farming Assistant":
    show_hero()

    st.markdown(
        '<div class="panel-heading">💬 Your Farming Assistant</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "A built-in FAQ chatbot. No API key or external AI service is required."
    )

    faq_answers = {
        "How do I predict crop yield?":
            "Open Home or Predict Yield, complete all nine fields, "
            "and press Predict Yield. The result is shown in tons per hectare.",

        "What does tons per hectare mean?":
            "It is the estimated crop mass harvested per hectare of land. "
            "One hectare equals 10,000 square metres.",

        "Is the prediction guaranteed?":
            "No. This is a machine-learning estimate. Actual harvest depends "
            "on pests, crop variety, farm management and other factors.",

        "Why does the model need soil and weather?":
            "These are among the conditions the model learned from its "
            "training data. Enter values that represent your farm.",

        "How can I improve the estimate?":
            "Enter accurate measurements and choose the closest matching "
            "crop, soil, region and weather categories.",

        "Where is the model file?":
            "Keep crop_yield_prediction_model.pkl in the same folder as app.py.",

        "Does this chatbot require an API?":
            "No. This assistant uses predefined answers stored in the app.",
    }

    for question, answer in faq_answers.items():
        with st.expander(question):
            st.write(answer)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello, farmer! 🌿 Ask me about using the predictor, "
                    "understanding yield estimates, or the model inputs."
                ),
            }
        ]

    st.markdown("---")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask your farming question...")

    if question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": question}
        )

        q = question.lower()

        if any(word in q for word in ["hello", "hi", "hey"]):
            answer = (
                "Hello! 🌱 I can help you use the crop predictor "
                "and understand its estimates."
            )
        elif any(word in q for word in ["how", "predict", "start", "use"]):
            answer = faq_answers["How do I predict crop yield?"]
        elif any(word in q for word in ["ton", "hectare", "unit"]):
            answer = faq_answers["What does tons per hectare mean?"]
        elif any(word in q for word in ["guarantee", "accurate", "reliable"]):
            answer = faq_answers["Is the prediction guaranteed?"]
        elif any(word in q for word in ["soil", "weather", "rain", "temperature"]):
            answer = faq_answers["Why does the model need soil and weather?"]
        elif any(word in q for word in ["improve", "better", "correct"]):
            answer = faq_answers["How can I improve the estimate?"]
        elif any(word in q for word in ["model", "file", "pkl", "error"]):
            answer = faq_answers["Where is the model file?"]
        elif "api" in q or "chatbot" in q:
            answer = faq_answers["Does this chatbot require an API?"]
        else:
            answer = (
                "I can help with crop yield predictions, model inputs, "
                "yield units and common app questions. Try asking about "
                "one of those topics."
            )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.chat_messages = []
        st.rerun()


# =========================================================
# INSIGHTS PAGE
# =========================================================

elif page == "Insights":
    show_hero()

    st.markdown(
        '<div class="panel-heading">📊 Prediction Insights</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.get("prediction_history", [])

    if history:
        history_df = pd.DataFrame(history)

        c1, c2 = st.columns(2)
        c1.metric("Predictions this session", len(history_df))
        c2.metric(
            "Average predicted yield",
            f"{history_df['Predicted yield (tons/ha)'].mean():.2f} tons/ha",
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            history_df.iloc[::-1].set_index("Time")[
                "Predicted yield (tons/ha)"
            ],
            color="#55b878",
        )

        st.download_button(
            "⬇️ Download history as CSV",
            history_df.to_csv(index=False),
            "verdant_prediction_history.csv",
            "text/csv",
        )
    else:
        st.info(
            "No predictions yet. Go to Home or Predict Yield "
            "to create your first prediction."
        )

    st.caption(
        "Prediction history is stored for the current session and "
        "may reset when the session ends."
    )


# =========================================================
# ABOUT THE MODEL PAGE
# =========================================================

elif page == "About the Model":
    show_hero()

    st.markdown(
        '<div class="panel-heading">ⓘ About VERDANT</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "VERDANT is a crop-yield prediction application that uses "
        "a trained machine-learning model to estimate yield from "
        "farm and environmental information."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Input features", "9")
    c2.metric("Prediction output", "Tons / hectare")
    c3.metric("Model status", "Loaded")

    st.markdown(
        """
        <div class="panel">
          <div class="panel-heading">How it works</div>
          <p>① Select your region, crop, soil and weather conditions.</p>
          <p>② Enter temperature, rainfall, fertilizer use,
             irrigation use and days to harvest.</p>
          <p>③ The trained preprocessing pipeline transforms your
             inputs before the regression model estimates yield.</p>
          <hr>
          <div class="panel-subtitle">
            Predictions are estimates, not guarantees. Actual crop
            yield can vary with farming practices, pests, crop variety
            and other conditions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
      🌿 VERDANT · Crop Yield Intelligence<br>
      Growing a smarter, more sustainable future.
    </div>
    """,
    unsafe_allow_html=True,
)

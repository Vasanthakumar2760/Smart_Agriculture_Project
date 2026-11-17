import joblib
import streamlit as st
import pandas as pd
import requests
import os


from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt



# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# CUSTOM CSS - LIGHT & DARK MODE SUPPORT
# ==========================================

st.markdown("""
<style>

/* Theme Variables for Light & Dark Mode */
:root {
    --background-color: #0e1117;
    --secondary-background-color: #262730;
    --text-color: #fafafa;
    --border-color: rgba(250, 250, 250, 0.18);
}

@media (prefers-color-scheme: light) {
    :root {
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #31333f;
        --border-color: rgba(49, 51, 63, 0.2);
    }
}

/* Prevent horizontal overflow across entire website */
html, body, .stApp {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* Main App */
.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

/* Page Container Spacing */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

@media (min-width: 768px) {
    .main .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
}

/* Headings Responsive Scaling */
h1 {
    color: var(--text-color) !important;
    font-size: clamp(1.6rem, 4vw, 2.2rem) !important;
    word-break: break-word !important;
}

h2 {
    color: var(--text-color) !important;
    font-size: clamp(1.3rem, 3.5vw, 1.8rem) !important;
    word-break: break-word !important;
}

h3 {
    color: var(--text-color) !important;
    font-size: clamp(1.1rem, 3vw, 1.4rem) !important;
    word-break: break-word !important;
}

h4 {
    color: var(--text-color) !important;
    font-size: clamp(1rem, 2.5vw, 1.2rem) !important;
    word-break: break-word !important;
}

/* Paragraphs and Labels */
.stApp p,
.stApp label {
    color: var(--text-color);
    word-break: break-word;
}

/* Sidebar with Solid Background */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
div[data-testid="stSidebarContent"] {
    background-color: var(--secondary-background-color) !important;
    opacity: 1 !important;
}

@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        max-width: 85vw !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5) !important;
    }
}

/* Responsive Columns Grid on Mobile */
@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 12px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: calc(50% - 6px) !important;
        flex: 1 1 calc(50% - 6px) !important;
    }
}

@media (max-width: 400px) {
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--border-color) !important;
    padding: 14px 16px !important;
    border-radius: 12px !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

/* Metric Labels */
div[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    font-size: 14px !important;
    word-break: break-word !important;
}

/* Metric Values */
div[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
    font-size: 24px !important;
    word-break: break-word !important;
}

/* Buttons */
.stButton > button {
    background-color: #2e7d32 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    box-sizing: border-box !important;
    transition: background-color 0.2s ease !important;
}

/* Button Hover */
.stButton > button:hover {
    background-color: #43a047 !important;
    color: white !important;
}

@media (max-width: 768px) {
    .stButton > button {
        width: 100% !important;
    }
}

/* Input Field Containers */
div[data-testid="stTextInputRootElement"],
div[data-testid="stNumberInputContainer"] {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease !important;
}

div[data-testid="stTextInputRootElement"]:hover,
div[data-testid="stNumberInputContainer"]:hover {
    border-color: #2e7d32 !important;
}

/* Input Inner Fields */
.stTextInput input,
.stNumberInput input,
input[data-testid="stTextInputField"],
input[data-testid="stNumberInputField"] {
    background-color: transparent !important;
    color: var(--text-color) !important;
    font-size: 14px !important;
}

/* NumberInput Controls */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    color: var(--text-color) !important;
    background: transparent !important;
}

button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover {
    color: #2e7d32 !important;
}

/* Dropdown (Selectbox) Trigger Container */
div[data-testid="stSelectbox"] > div:nth-of-type(1) > div,
div[data-testid="stSelectbox"] div[data-focus-within],
div[data-baseweb="select"] > div {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    min-height: 42px !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease !important;
}

div[data-testid="stSelectbox"] > div:nth-of-type(1) > div:hover,
div[data-baseweb="select"] > div:hover {
    border-color: #2e7d32 !important;
}

/* Dropdown Input / Value Text */
div[data-testid="stSelectbox"] input,
div[data-baseweb="select"] input {
    color: var(--text-color) !important;
    background-color: transparent !important;
    font-size: 14px !important;
}

/* Dropdown Chevron / Open Button */
div[data-testid="stSelectbox"] button[aria-label="Open"],
div[data-testid="stSelectbox"] button[aria-label="Open"] svg,
div[data-baseweb="select"] svg {
    color: var(--text-color) !important;
    fill: currentColor !important;
}

/* Dropdown Popover Menu Container with Solid Background */
div[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-testid="stSelectboxVirtualDropdown"] > div {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    z-index: 1000000 !important;
    max-width: calc(100vw - 20px) !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    opacity: 1 !important;
}

/* Dropdown Options List with Solid Background */
div[data-testid="stSelectboxVirtualDropdown"] ul,
div[data-baseweb="popover"] ul,
ul[aria-label="Selectbox options"],
ul[role="listbox"] {
    background-color: var(--secondary-background-color) !important;
    max-height: min(240px, 45vh) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    overscroll-behavior: contain !important;
    padding: 4px !important;
    margin: 0 !important;
    list-style: none !important;
    opacity: 1 !important;
}

/* Dropdown Option Items */
div[data-testid="stSelectboxVirtualDropdown"] li[role="option"],
div[data-baseweb="popover"] li[role="option"],
li[role="option"] {
    min-height: 42px !important;
    padding: 8px 12px !important;
    display: flex !important;
    align-items: center !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    border-radius: 6px !important;
    box-sizing: border-box !important;
    transition: background-color 0.15s ease !important;
}

/* Dropdown Option Highlight / Text */
div[data-testid="stSelectboxVirtualDropdown"] [data-item-hl],
div[data-baseweb="popover"] [data-item-hl],
li[role="option"] span,
li[role="option"] div {
    color: var(--text-color) !important;
    white-space: normal !important;
    word-break: break-word !important;
    text-align: left !important;
    line-height: 1.4 !important;
}

/* Dropdown Option Hover & Focus */
div[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
div[data-testid="stSelectboxVirtualDropdown"] li[role="option"][data-hovered],
div[data-testid="stSelectboxVirtualDropdown"] li[role="option"][data-focused],
div[data-baseweb="popover"] li[role="option"]:hover,
li[role="option"]:hover [data-item-hl] {
    background-color: rgba(46, 125, 50, 0.15) !important;
}

/* Dropdown Selected Option */
div[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"],
div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
    background-color: rgba(46, 125, 50, 0.25) !important;
    font-weight: 600 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Charts */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    word-break: break-word !important;
}

/* Horizontal Divider */
hr {
    border-color: var(--border-color) !important;
    margin: 1.5rem 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "crops.csv"
    )

    try:
        data = pd.read_csv(csv_path)

    except FileNotFoundError:
        st.error(
            "crops.csv file not found. "
            "Please place crops.csv inside your project folder."
        )
        st.stop()

    required_columns = [
        "Crop",
        "Soil_Type",
        "Temperature",
        "Rainfall",
        "Humidity"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns in crops.csv: {missing_columns}"
        )
        st.stop()

    if data.empty:
        st.error("The crops.csv file is empty.")
        st.stop()

    # Convert numerical columns
    numerical_columns = [
        "Temperature",
        "Rainfall",
        "Humidity"
    ]

    for column in numerical_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    )

    if data.empty:
        st.error(
            "No valid crop records found in crops.csv."
        )
        st.stop()

    return data


# Load the dataset
data = load_data()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🌱 Smart Agriculture")

st.sidebar.markdown("---")

menu = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "Crop Information",
        "Crop Recommendation",
        "Weather Information",
        "ML Crop Prediction",
        "Model Evaluation Dashboard"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Smart Agriculture System\n\n"
    "A project for crop information, "
    "recommendations, weather, and machine learning."
)

# ==========================================
# DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.title("🌱 Smart Agriculture Dashboard")

    st.write(
        "Welcome to the Smart Agriculture System."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Crops", data["Crop"].nunique())

    with col2:
        st.metric("Soil Types", data["Soil_Type"].nunique())

    with col3:
        st.metric(
            "Average Temperature",
            f"{data['Temperature'].mean():.1f} °C"
        )

    with col4:
        st.metric(
            "Average Rainfall",
            f"{data['Rainfall'].mean():.1f} mm"
        )

    st.markdown("---")

    st.subheader("📊 Crop Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )

    st.markdown("---")

    # Temperature Chart
    st.subheader("🌡️ Average Temperature by Crop")

    temperature_chart = data[
        ["Crop", "Temperature"]
    ].copy()

    temperature_chart = temperature_chart.set_index("Crop")

    st.bar_chart(
        temperature_chart,
        y="Temperature"
    )

    st.markdown("---")

    # Rainfall Chart
    st.subheader("🌧️ Rainfall by Crop")

    rainfall_chart = data[
        ["Crop", "Rainfall"]
    ].copy()

    rainfall_chart = rainfall_chart.set_index("Crop")

    st.bar_chart(
        rainfall_chart,
        y="Rainfall"
    )
# ==========================================
# CROP RECOMMENDATION
# ==========================================

elif menu == "Crop Recommendation":

    st.title("🌱 Smart Crop Recommendation")
    st.write(
        "Find suitable crops based on soil and weather conditions."
    )

    st.markdown("---")

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        soil_type = st.selectbox(
            "Select Soil Type",
            sorted(data["Soil_Type"].unique())
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0
        )

    with col2:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=700.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    st.markdown("---")

    if st.button("🌾 Recommend Crop", use_container_width=True):

        # Filter crops based on soil type
        soil_data = data[
            data["Soil_Type"] == soil_type
        ].copy()

        if soil_data.empty:

            st.warning(
                "No crops are available for the selected soil type."
            )

        else:

            # Calculate difference between input and crop conditions
            soil_data["Score"] = (
                abs(soil_data["Temperature"] - temperature) / 10
                + abs(soil_data["Rainfall"] - rainfall) / 1000
                + abs(soil_data["Humidity"] - humidity) / 100
            )

            # Sort crops by suitability score
            recommended_crops = soil_data.sort_values(
                by="Score",
                ascending=True
            )

            best_crop = recommended_crops.iloc[0]

            st.success("Crop Recommendation Generated!")

            st.subheader("🌿 Best Suitable Crop")

            st.metric(
                "Recommended Crop",
                best_crop["Crop"]
            )

            st.markdown("---")

            st.subheader("📊 Recommended Crops")

            st.dataframe(
                recommended_crops[
                    [
                        "Crop",
                        "Soil_Type",
                        "Temperature",
                        "Rainfall",
                        "Humidity"
                    ]
                ],
                use_container_width=True
            )

            st.info(
                "This recommendation is based on similarity "
                "to the crop conditions in your dataset. "
                "Actual suitability also depends on local "
                "agricultural conditions."
            )
# ==========================================
# CROP INFORMATION
# ==========================================

elif menu == "Crop Information":

    st.title("🌾 Crop Information")

    st.write(
        "Explore detailed information about different crops."
    )

    st.markdown("---")

    crop_list = sorted(
        data["Crop"].unique()
    )

    selected_crop = st.selectbox(
        "Select a Crop",
        crop_list,
        key="crop_information_select"
    )

    crop_details = data[
        data["Crop"] == selected_crop
    ]

    if not crop_details.empty:

        crop = crop_details.iloc[0]

        st.subheader(
            f"🌱 Information about {selected_crop}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🌍 Soil Type",
                crop["Soil_Type"]
            )

            st.metric(
                "🌡️ Temperature",
                f"{crop['Temperature']} °C"
            )

        with col2:

            st.metric(
                "🌧️ Rainfall",
                f"{crop['Rainfall']} mm"
            )

            st.metric(
                "💧 Humidity",
                f"{crop['Humidity']} %"
            )

        st.markdown("---")

        st.subheader("📋 Complete Crop Details")

        st.dataframe(
            crop_details,
            use_container_width=True
        )

    else:

        st.warning(
            "No information available for this crop."
        )


# ==========================================
# WEATHER INFORMATION
# ==========================================

elif menu == "Weather Information":

    st.title("🌦️ Weather Information")

    st.write(
        "Get current weather information for a city."
    )

    st.markdown("---")

    city = st.text_input(
        "Enter City Name",
        placeholder="Example: Tiruppur"
    )

    if st.button("🌤️ Get Weather", use_container_width=True):

        if city.strip() == "":

            st.warning(
                "Please enter a city name."
            )

        else:

            try:

                # Read API key from Streamlit secrets
                api_key = st.secrets[
                    "OPENWEATHER_API_KEY"
                ]

                weather_url = (
                    "https://api.openweathermap.org/data/2.5/weather"
                )

                params = {
                    "q": city,
                    "appid": api_key,
                    "units": "metric"
                }

                response = requests.get(
                    weather_url,
                    params=params,
                    timeout=10
                )

                if response.status_code == 200:

                    weather_data = response.json()

                    city_name = weather_data[
                        "name"
                    ]

                    temperature = weather_data[
                        "main"
                    ]["temp"]

                    humidity = weather_data[
                        "main"
                    ]["humidity"]

                    pressure = weather_data[
                        "main"
                    ]["pressure"]

                    weather_description = weather_data[
                        "weather"
                    ][0]["description"]

                    wind_speed = weather_data[
                        "wind"
                    ]["speed"]

                    st.success(
                        f"Weather information for {city_name}"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Temperature",
                            f"{temperature} °C"
                        )

                    with col2:

                        st.metric(
                            "Humidity",
                            f"{humidity} %"
                        )

                    with col3:

                        st.metric(
                            "Pressure",
                            f"{pressure} hPa"
                        )

                    st.markdown("---")

                    st.subheader("🌤️ Weather Details")

                    st.write(
                        f"**Weather:** "
                        f"{weather_description.title()}"
                    )

                    st.write(
                        f"**Wind Speed:** "
                        f"{wind_speed} m/s"
                    )

                elif response.status_code == 404:

                    st.error(
                        "City not found. "
                        "Please check the city name."
                    )

                elif response.status_code == 401:

                    st.error(
                        "Invalid API key. "
                        "Please check your OpenWeather API key."
                    )

                else:

                    st.error(
                        "Unable to retrieve weather information. "
                        f"Error code: {response.status_code}"
                    )

            except KeyError:

                st.error(
                    "OPENWEATHER_API_KEY is missing "
                    "from your Streamlit secrets."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The weather service took too long to respond. "
                    "Please try again."
                )

            except requests.exceptions.RequestException:

                st.error(
                    "Unable to connect to the weather service. "
                    "Check your internet connection."
                )

# ==========================================
# MACHINE LEARNING CROP PREDICTION
# ==========================================

elif menu == "ML Crop Prediction":

    st.title("🤖 Machine Learning Crop Prediction")

    st.write(
        "Predict a suitable crop using a Random Forest "
        "Machine Learning model."
    )

    st.markdown("---")

    # ==========================================
    # MODEL TRAINING
    # ==========================================

    X = data[
        [
            "Soil_Type",
            "Temperature",
            "Rainfall",
            "Humidity"
        ]
    ]

    y = data["Crop"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "soil",
                OneHotEncoder(handle_unknown="ignore"),
                ["Soil_Type"]
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X, y)

    # ==========================================
    # USER INPUTS
    # ==========================================

    st.subheader("🌾 Enter Agricultural Conditions")

    col1, col2 = st.columns(2)

    with col1:

        soil_type = st.selectbox(
            "Select Soil Type",
            sorted(data["Soil_Type"].unique()),
            key="ml_soil_type"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0,
            key="ml_temperature"
        )

    with col2:

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=700.0,
            key="ml_rainfall"
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            key="ml_humidity"
        )

    st.markdown("---")

    # ==========================================
    # PREDICTION
    # ==========================================

    if st.button(
        "🤖 Predict Suitable Crop",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Soil_Type": [soil_type],
                "Temperature": [temperature],
                "Rainfall": [rainfall],
                "Humidity": [humidity]
            }
        )

        prediction = model.predict(input_data)

        predicted_crop = prediction[0]

        st.success("Machine Learning Prediction Completed!")

        st.markdown(
            f"""
            <div style="
                background-color: #14532d;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-sizing: border-box;
                max-width: 100%;
                word-break: break-word;
            ">
                <h3 style="color: white; margin-top: 0;">
                    🌱 Predicted Suitable Crop
                </h3>
                <h1 style="color: #fef3c7; margin-bottom: 0; font-size: clamp(1.8rem, 5vw, 2.5rem); word-break: break-word;">
                    {predicted_crop}
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("📋 Input Conditions")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.info(
            "This prediction is based on the available "
            "training dataset. The dataset is small, so "
            "the prediction is for demonstration purposes "
            "and is not a guaranteed agricultural recommendation."
        )

# ==========================================
# MODEL EVALUATION DASHBOARD
# ==========================================

elif menu == "Model Evaluation Dashboard":

    st.title("🤖 ML Model Evaluation Dashboard")

    st.write(
        "Analyze the performance of the Random Forest "
        "crop prediction model."
    )

    st.markdown("---")

    # Load dataset
    X = data[
        [
            "Soil_Type",
            "Temperature",
            "Rainfall",
            "Humidity"
        ]
    ]

    y = data["Crop"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "soil",
                OneHotEncoder(handle_unknown="ignore"),
                ["Soil_Type"]
            )
        ],
        remainder="passthrough"
    )

    # Model
    evaluation_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    # Train model
    evaluation_model.fit(X_train, y_train)

    # Predictions
    y_pred = evaluation_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # ==========================================
    # METRIC CARDS
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "📚 Training Records",
            len(X_train)
        )

    with col3:
        st.metric(
            "🧪 Testing Records",
            len(X_test)
        )

    st.markdown("---")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=evaluation_model.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=evaluation_model.classes_,
        columns=evaluation_model.classes_
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(2),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # MODEL PERFORMANCE CHART
    # ==========================================

    st.subheader("📈 Model Performance")

    performance = pd.DataFrame(
        {
            "Metric": ["Accuracy"],
            "Score": [accuracy * 100]
        }
    )

    st.bar_chart(
        performance.set_index("Metric")
    )

    st.success(
        "Model Evaluation Dashboard Loaded Successfully!"
    )

    st.info(
        "This evaluation uses synthetic demonstration data. "
        "The displayed accuracy is not a real-world "
        "agricultural performance guarantee."
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#1b5e20;">
        🌱 Smart Agriculture System |
        Developed using Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
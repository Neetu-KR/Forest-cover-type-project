import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>

.main {
    background-color: #f4f8f3;
}

.title {
    text-align: center;
    color: #176b3a;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    background-color: #e8f5e9;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------

rf_model = joblib.load("random_forest_model.pkl")
gb_model = joblib.load("gradient_boosting_model.pkl")
knn_model = joblib.load("knn_model.pkl")

columns = joblib.load("model_columns.pkl")
median_vals = joblib.load("median_values.pkl")

# ---------------- TITLE ----------------

st.markdown(
    '<div class="title">🌲 Forest Cover Type Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Forest Cover Classification</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- MODEL SELECTION ----------------

st.markdown("### 🤖 Select Machine Learning Model")

model_name = st.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "Gradient Boosting",
        "KNN"
    ]
)

if model_name == "Random Forest":
    model = rf_model

elif model_name == "Gradient Boosting":
    model = gb_model

else:
    model = knn_model

# ---------------- INPUTS ----------------

st.markdown("### 🌳 Terrain Details")

col1, col2 = st.columns(2)

with col1:

    elevation = st.slider(
        "🏔️ Elevation (meters)",
        0, 5000, 3000
    )

    aspect = st.slider(
        "🧭 Aspect (degrees)",
        0, 360, 180
    )

    slope = st.slider(
        "📐 Slope (degrees)",
        0, 90, 15
    )

    horiz_hydro = st.slider(
        "💧 Horizontal Distance to Hydrology",
        0, 5000, 200
    )

with col2:

    vert_hydro = st.slider(
        "💧 Vertical Distance to Hydrology",
        -500, 500, 50
    )

    horiz_road = st.slider(
        "🛣️ Horizontal Distance to Roadways",
        0, 10000, 1000
    )

    horiz_fire = st.slider(
        "🔥 Horizontal Distance to Fire Points",
        0, 10000, 1000
    )

# ---------------- CALCULATE FEATURES ----------------

distance_hydrology = np.sqrt(
    horiz_hydro**2 + vert_hydro**2
)

terrain_ruggedness = elevation * slope

# ---------------- DISPLAY VALUES ----------------

st.markdown("### 📊 Terrain Measurements")

chart_data = pd.DataFrame({
    "Feature": [
        "Elevation",
        "Slope",
        "Hydrology",
        "Roadways",
        "Fire Points"
    ],
    "Value": [
        elevation,
        slope,
        distance_hydrology,
        horiz_road,
        horiz_fire
    ]
})

st.bar_chart(
    chart_data.set_index("Feature")
)

# ---------------- PREDICT ----------------

st.write("")

if st.button(
    "🔮 Predict Forest Cover",
    type="primary",
    use_container_width=True
):

    # Start with median values
    row = median_vals.copy()

    # User values
    row["Elevation"] = elevation
    row["Aspect"] = aspect
    row["Slope"] = slope

    row["Horizontal_Distance_To_Hydrology"] = horiz_hydro
    row["Vertical_Distance_To_Hydrology"] = vert_hydro

    row["Horizontal_Distance_To_Roadways"] = horiz_road
    row["Horizontal_Distance_To_Fire_Points"] = horiz_fire

    # Engineered features
    row["Distance_To_Hydrology"] = distance_hydrology
    row["Terrain_Ruggedness"] = terrain_ruggedness

    # Arrange columns exactly like training
    input_df = pd.DataFrame([row])[columns]

    # Prediction
    prediction = model.predict(input_df)[0]

    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }

    result = cover_types.get(
        prediction,
        "Unknown"
    )

    # ---------------- RESULT ----------------

    st.markdown(
        f"""
        <div class="result">
        🌳 Predicted Forest Cover<br><br>
        <span style="font-size:32px;">
        {prediction} — {result}
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- CONFIDENCE ----------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        prob_df = pd.DataFrame({
            "Forest Cover": [
                cover_types[i + 1]
                for i in range(len(probabilities))
            ],
            "Probability": probabilities
        })

        prob_df = prob_df.sort_values(
            "Probability",
            ascending=False
        )

        st.markdown("### 📈 Prediction Confidence")

        st.bar_chart(
            prob_df.set_index("Forest Cover")
        )

        st.write(
            f"Selected Model: **{model_name}**"
        )

    # ---------------- INPUT SUMMARY ----------------

    st.markdown("### 📋 Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal Hydrology",
            "Vertical Hydrology",
            "Roadways",
            "Fire Points",
            "Distance to Hydrology",
            "Terrain Ruggedness"
        ],
        "Value": [
            elevation,
            aspect,
            slope,
            horiz_hydro,
            vert_hydro,
            horiz_road,
            horiz_fire,
            round(distance_hydrology, 2),
            terrain_ruggedness
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )
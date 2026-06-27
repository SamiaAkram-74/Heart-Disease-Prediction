import streamlit as st
import pandas as pd
import joblib

# =========================================================
#                     PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# =========================================================
#                     LOAD MODEL ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('Logistic_Regression_HD_prediction.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# =========================================================
#                     HEADER
# =========================================================
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Estimate heart disease risk based on clinical parameters, "
    "using a trained Logistic Regression model."
)
st.caption(
    "⚠️ For educational purposes only — not a substitute for professional "
    "medical advice, diagnosis, or treatment."
)
st.divider()

if not artifacts_loaded:
    st.error(
        "Could not load model files. Make sure `Logistic_Regression_HD_prediction.pkl`, "
        f"`scaler.pkl`, and `columns.pkl` are in the same folder as this script.\n\n{load_error}"
    )
    st.stop()

# =========================================================
#                     SIDEBAR — PATIENT INPUTS
# =========================================================
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.selectbox("Sex", ['M', 'F'])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
    help="ATA: Atypical Angina | NAP: Non-Anginal Pain | TA: Typical Angina | ASY: Asymptomatic"
)
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dL", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox(
    "Exercise-Induced Angina", ["Y", "N"],
    format_func=lambda x: "Yes" if x == "Y" else "No"
)
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# =========================================================
#                     PREDICTION
# =========================================================
if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBp': resting_bp,
        'Cholestrol': cholestrol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'OldPeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Risk Probability", f"{prob*100:.1f}%")
    col2.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")

    st.progress(min(int(prob * 100), 100))

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    with st.expander("View processed input data"):
        st.dataframe(input_df, use_container_width=True)

st.divider()

# =========================================================
#         MODEL TRANSPARENCY — REAL FEATURE IMPORTANCE
# =========================================================
with st.expander("📊 About the Model"):
    st.markdown(
        """
        - **Algorithm:** Logistic Regression
        - **Inputs:** 11 clinical features
        - **Output:** Probability of heart disease (0–100%)
        """
    )

    if hasattr(model, "coef_"):
        st.markdown("**Feature Importance (model coefficients)**")
        imp_df = pd.DataFrame({
            "Feature": expected_columns,
            "Coefficient": model.coef_.flatten()
        }).sort_values("Coefficient", key=abs, ascending=False).head(10)
        st.bar_chart(imp_df.set_index("Feature"))
        st.caption(
            "Positive values push the prediction toward higher risk; "
            "negative values push toward lower risk."
        )

st.caption("Built by Samia Akram")
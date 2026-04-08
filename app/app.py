import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.express as px

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ridge_model = pickle.load(open(os.path.join(BASE_DIR, "models/ridge_model.pkl"), "rb"))
lasso_model = pickle.load(open(os.path.join(BASE_DIR, "models/lasso_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "models/scaler.pkl"), "rb"))

df = pd.read_csv(os.path.join(BASE_DIR, "data/boston.csv"))

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="House Price Predictor", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")

theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])

# ---------- THEME ----------
if theme == "Dark":
    st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ---------- HEADER ----------
st.title("🏠 House Price Prediction Dashboard")
st.markdown("### Ridge vs Lasso Regression | Interactive ML App")

# ---------- HERO IMAGE ----------
st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994", use_container_width=True)

# ---------- DATASET INFO ----------
with st.expander("📊 About Dataset"):
    st.write("""
    Boston Housing dataset contains features like:
    - RM: Rooms
    - LSTAT: % lower income population
    - PTRATIO: Student-teacher ratio
    - INDUS: Industry area percentage
    
    Target:
    - MEDV: House price
    """)

# ---------- INPUT ----------
st.sidebar.header("📥 Input Features")

rm = st.sidebar.slider("Rooms (RM)", 1.0, 10.0, 5.0)
lstat = st.sidebar.slider("LSTAT (%)", 0.0, 40.0, 10.0)
ptratio = st.sidebar.slider("PTRATIO", 10.0, 25.0, 18.0)
indus = st.sidebar.slider("INDUS (Industry Area %)", 0.0, 30.0, 10.0)

input_data = np.array([[rm, lstat, ptratio, indus]])
input_scaled = scaler.transform(input_data)

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- PREDICTION ----------
if st.sidebar.button("🚀 Predict Price"):

    ridge_pred = ridge_model.predict(input_scaled)[0]
    lasso_pred = lasso_model.predict(input_scaled)[0]

    st.session_state.history.append({
        "RM": rm,
        "LSTAT": lstat,
        "PTRATIO": ptratio,
        "Ridge": ridge_pred,
        "Lasso": lasso_pred
    })

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📈 Ridge Prediction", f"${ridge_pred:.2f}K")

    with col2:
        st.metric("📉 Lasso Prediction", f"${lasso_pred:.2f}K")

# ---------- MODEL COMPARISON ----------
st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Ridge", "Lasso"],
    "Prediction": [
        ridge_model.predict(input_scaled)[0],
        lasso_model.predict(input_scaled)[0]
    ]
})

fig = px.bar(comparison_df, x="Model", y="Prediction", color="Model", title="Model Comparison")
st.plotly_chart(fig, use_container_width=True)

# ---------- FEATURE IMPORTANCE (COEFFICIENTS) ----------
st.subheader("📌 Feature Importance (Ridge Coefficients)")

features = ["RM", "LSTAT", "PTRATIO", "INDUS"]
coeffs = ridge_model.coef_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": coeffs
})

fig2 = px.bar(importance_df, x="Feature", y="Importance", color="Feature")
st.plotly_chart(fig2, use_container_width=True)

# ---------- DATA VISUALIZATION ----------
st.subheader("📈 RM vs Price")

fig3 = px.scatter(df, x="RM", y="MEDV", title="Rooms vs Price")
st.plotly_chart(fig3, use_container_width=True)

# ---------- PREDICTION HISTORY ----------
st.subheader("🕘 Prediction History")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.write("No predictions yet.")

# ---------- ANIMATION ----------
st.markdown("### 🎉 Thank you for using the app!")

st.markdown("""
<div style="text-align:center;">
    <img src="https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif" width="300">
</div>
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Project by You")
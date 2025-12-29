# ===============================
# PRIMARY EDUCATION DASHBOARD
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Primary Education Prediction",
    layout="wide"
)

st.title("📘 Primary Education Prediction Dashboard")
st.markdown("Predicting **Share of Population with Basic Education** using Machine Learning")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")

    return df

df = load_data()

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("🔎 Filters")

country = st.sidebar.selectbox(
    "Select Country",
    df["Entity"].unique()
)

filtered_df = df[df["Entity"] == country]

# -------------------------------
# DATA PREPARATION
# -------------------------------
X = filtered_df[['Year', 'Share of population with no education']]
y = filtered_df['Share of population with at least some basic education']

# Handle missing values safely
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# METRICS
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", round(mse, 2))
col2.metric("R² Score", round(r2, 2))

# -------------------------------
# ACTUAL VS PREDICTED (DYNAMIC GRAPH)
# -------------------------------
st.subheader("📊 Actual vs Predicted Trend")

# Sort by year for clean plotting
filtered_df = filtered_df.sort_values("Year")

X_all = filtered_df[['Year', 'Share of population with no education']]
y_all = filtered_df['Share of population with at least some basic education']

y_all_pred = model.predict(X_all)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(filtered_df["Year"], y_all, label="Actual", linewidth=2)
ax.plot(filtered_df["Year"], y_all_pred, label="Predicted", linestyle="--")

ax.set_xlabel("Year")
ax.set_ylabel("Education Share (%)")
ax.set_title(f"Actual vs Predicted Education Share - {country}")
ax.legend()

st.pyplot(fig)

# -------------------------------
# USER INPUT PREDICTION
# -------------------------------
st.subheader("🧮 Predict New Value")

year_input = st.number_input(
    "Enter Year",
    min_value=1900,
    max_value=2100,
    value=2020
)

no_edu_input = st.number_input(
    "Share of population with NO education (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

if st.button("Predict"):
    input_data = np.array([[year_input, no_edu_input]])
    prediction = model.predict(input_data)

    st.success(
        f"✅ Predicted Share with Basic Education: **{prediction[0]:.2f}%**"
    )

    # -------------------------------
    # UPDATE GRAPH WITH USER POINT
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(filtered_df["Year"], y_all, label="Actual", linewidth=2)
    ax2.plot(filtered_df["Year"], y_all_pred, label="Predicted", linestyle="--")

    ax2.scatter(
        year_input,
        prediction[0],
        color="red",
        s=100,
        label="User Prediction"
    )

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Education Share (%)")
    ax2.set_title(f"Prediction Result - {country}")
    ax2.legend()

    st.pyplot(fig2)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🎓 **Primary Education Prediction Project using Machine Learning**")

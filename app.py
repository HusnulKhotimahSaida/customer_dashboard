import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Customer Subscription & Churn Prediction",
    layout="centered"
)

# =========================
# JUDUL
# =========================
st.title("📊 Customer Subscription & Churn Prediction")

st.markdown("""
Aplikasi ini menggunakan **input data pelanggan secara manual**  
dan melakukan **training serta prediksi langsung** menggunakan
model **Random Forest**, **tanpa file `.pkl` dan tanpa file CSV**.

Pendekatan ini digunakan untuk **pembelajaran dan demonstrasi konsep ML**.
""")

st.markdown("---")

# =========================
# DATA TRAINING (HARDCODE)
# =========================
data = {
    "age": [25, 30, 45, 35, 50, 28, 40, 60],
    "income": [3000000, 5000000, 8000000, 6000000, 10000000, 4000000, 7000000, 12000000],
    "credit_score": [600, 650, 720, 680, 750, 620, 700, 780],
    "total_spent": [1000000, 2000000, 5000000, 3000000, 7000000, 1500000, 4000000, 9000000],
    "subscription": [0, 1, 1, 1, 1, 0, 1, 1],
    "churn_risk": [0.7, 0.4, 0.2, 0.3, 0.1, 0.6, 0.25, 0.05]
}

df = pd.DataFrame(data)

features = ["age", "income", "credit_score", "total_spent"]

X = df[features]
y_class = df["subscription"]
y_reg = df["churn_risk"]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN MODEL
# =========================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_scaled, y_class)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_scaled, y_reg)

# =========================
# INPUT MANUAL USER
# =========================
st.subheader("✍️ Input Data Pelanggan")

age = st.number_input("Umur", min_value=0, max_value=100, value=30)
income = st.number_input("Pendapatan", min_value=0.0, value=5000000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
total_spent = st.number_input("Total Pengeluaran", min_value=0.0, value=2000000.0)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame(
        [[age, income, credit_score, total_spent]],
        columns=features
    )

    input_scaled = scaler.transform(input_df)

    subscription_pred = clf.predict(input_scaled)[0]
    churn_risk_pred = reg.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader("📌 Hasil Prediksi")

    st.write("**Status Subscription:**")
    if subscription_pred == 1:
        st.success("Berlangganan")
    else:
        st.warning("Tidak Berlangganan")

    st.write("**Churn Risk:**")
    st.info(f"{churn_risk_pred:.2f}")

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
Aplikasi ini melakukan **prediksi secara langsung** menggunakan
model **Random Forest** tanpa menyimpan model ke file `.pkl`.

Model akan **dilatih ulang setiap aplikasi dijalankan**.
""")

st.markdown("---")

# =========================
# LOAD DATA TRAINING
# =========================
df = pd.read_csv("data_pelanggan.csv")

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

age = st.number_input("Umur", 0, 100, 30)
income = st.number_input("Pendapatan", min_value=0.0, value=5000000.0)
credit_score = st.number_input("Credit Score", 300, 900, 650)
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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Customer Subscription & Churn Dashboard",
    layout="centered"
)

# =========================
# JUDUL & DESKRIPSI
# =========================
st.title("📊 Customer Subscription & Churn Analysis")

st.markdown("""
Aplikasi ini digunakan untuk **memasukkan hasil prediksi pelanggan secara manual**
dan menampilkan visualisasi:

- **Klasifikasi Subscription** (0 = Tidak, 1 = Ya)
- **Regresi Churn Risk** (nilai risiko churn)

Model **Random Forest** dilatih di **Google Colab**,  
sedangkan aplikasi ini berfungsi sebagai **dashboard input & visualisasi hasil prediksi**.
""")

st.markdown("---")

# =========================
# INPUT MANUAL
# =========================
st.subheader("✍️ Input Data Prediksi Manual")

subscription_pred = st.selectbox(
    "Prediksi Subscription",
    options=[0, 1],
    format_func=lambda x: "Tidak Berlangganan (0)" if x == 0 else "Berlangganan (1)"
)

churn_risk_pred = st.slider(
    "Nilai Churn Risk",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

# =========================
# SIMPAN DATA KE DATAFRAME
# =========================
data = {
    "subscription_pred": [subscription_pred],
    "churn_risk_pred": [churn_risk_pred]
}

df = pd.DataFrame(data)

st.markdown("---")

# =========================
# TAMPIL DATA
# =========================
st.subheader("📄 Data Prediksi")
st.dataframe(df)

st.markdown("---")

# =========================
# VISUALISASI KLASIFIKASI
# =========================
st.subheader("✅ Hasil Klasifikasi Subscription")

fig1, ax1 = plt.subplots()
df["subscription_pred"].value_counts().plot(
    kind="bar",
    ax=ax1
)
ax1.set_xlabel("Subscription (0 = Tidak, 1 = Ya)")
ax1.set_ylabel("Jumlah Data")
st.pyplot(fig1)

st.info("""
**Interpretasi Klasifikasi:**

Nilai **0** menunjukkan pelanggan diprediksi **tidak berlangganan**,  
sedangkan nilai **1** menunjukkan pelanggan diprediksi **berlangganan**.  

Hasil ini merupakan keluaran model Random Forest berdasarkan karakteristik pelanggan
yang sebelumnya dipelajari pada proses training di Google Colab.
""")

st.markdown("---")

# =========================
# VISUALISASI REGRESI
# =========================
st.subheader("📈 Hasil Regresi Churn Risk")

fig2, ax2 = plt.subplots()
ax2.plot(df["churn_risk_pred"], marker="o")
ax2.set_ylabel("Churn Risk")
ax2.set_xlabel("Data ke-")
st.pyplot(fig2)

st.info("""
**Interpretasi Regresi:**

Nilai **churn risk** menunjukkan tingkat risiko pelanggan untuk berhenti berlangganan.
Semakin mendekati **1**, semakin tinggi risiko churn.

Model Random Forest digunakan karena mampu menangkap hubungan non-linear
dan memberikan prediksi yang lebih stabil terhadap data pelanggan.
""")

st.markdown("---")

# =========================
# DOWNLOAD
# =========================
st.download_button(
    "⬇️ Download Data Prediksi",
    df.to_csv(index=False),
    file_name="hasil_prediksi_manual.csv",
    mime="text/csv"
)

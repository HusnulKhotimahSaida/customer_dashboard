import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Analysis Dashboard",
    layout="centered"
)

st.title("📊 Customer Subscription & Churn Analysis")
st.markdown(
    "Metode: **Random Forest (Ensemble Method)**  \n"
    "Aplikasi ini menampilkan hasil **klasifikasi** dan **regresi** "
    "berdasarkan output model yang dijalankan di Google Colab."
)

st.markdown("---")

# =========================
# TAB
# =========================
tab1, tab2 = st.tabs(["🟥 Klasifikasi", "🟦 Regresi"])

# ======================================================
# TAB 1 — KLASIFIKASI
# ======================================================
with tab1:
    st.header("Klasifikasi Subscription")

    st.markdown("### Input Manual (Simulasi 1 Pelanggan)")

    subscription_pred = st.selectbox(
        "Prediksi Subscription",
        options=[0, 1],
        format_func=lambda x: "Tidak Berlangganan" if x == 0 else "Berlangganan"
    )

    subscription_prob = st.slider(
        "Probabilitas Subscription",
        0.0, 1.0, 0.5
    )

    st.markdown("### Hasil Klasifikasi")

    if subscription_pred == 1:
        st.success(
            f"Kategori: **BERLANGGANAN**  \n"
            f"Probabilitas: **{subscription_prob:.2f}**"
        )
    else:
        st.warning(
            f"Kategori: **TIDAK BERLANGGANAN**  \n"
            f"Probabilitas: **{subscription_prob:.2f}**"
        )

    # Visualisasi sederhana
    fig1, ax1 = plt.subplots()
    ax1.bar(["Tidak", "Ya"], [1 - subscription_prob, subscription_prob])
    ax1.set_ylabel("Probabilitas")
    st.pyplot(fig1)

    st.info("""
**Interpretasi:**
Hasil klasifikasi menunjukkan prediksi status subscription pelanggan.
Probabilitas menggambarkan tingkat keyakinan model terhadap hasil klasifikasi.
""")

# ======================================================
# TAB 2 — REGRESI
# ======================================================
with tab2:
    st.header("Regresi Churn Risk")

    st.markdown("### Input Manual (Simulasi 1 Pelanggan)")

    churn_risk = st.slider(
        "Nilai Churn Risk",
        0.0, 1.0, 0.3
    )

    st.markdown("### Hasil Regresi")

    st.metric(
        label="Churn Risk",
        value=f"{churn_risk:.2f}"
    )

    # Visualisasi
    fig2, ax2 = plt.subplots()
    ax2.plot([churn_risk], marker="o")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Churn Risk")
    ax2.set_xlabel("Index Data")
    st.pyplot(fig2)

    st.info("""
**Interpretasi:**
Nilai churn risk menunjukkan tingkat risiko pelanggan berhenti berlangganan.
Semakin mendekati 1, semakin tinggi risiko churn.
""")

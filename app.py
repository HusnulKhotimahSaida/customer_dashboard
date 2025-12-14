import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Customer Analysis Dashboard",
    layout="centered"
)

st.title("📊 Customer Subscription & Churn Analysis")
st.markdown("""
Metode yang digunakan adalah **Random Forest (Ensemble Method)**.  
Model dijalankan di **Google Colab**, sedangkan aplikasi ini berfungsi sebagai
**dashboard visualisasi, simulasi, dan interpretasi hasil**.
""")

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

    mode_klas = st.radio(
        "Pilih metode input:",
        ["✍️ Input Manual", "📂 Upload CSV"],
        key="mode_klas"
    )

    # ---------- INPUT MANUAL ----------
    if mode_klas == "✍️ Input Manual":
        st.subheader("Input Manual (Simulasi 1 Pelanggan)")

        subscription_pred = st.selectbox(
            "Prediksi Subscription",
            [0, 1],
            format_func=lambda x: "Tidak Berlangganan" if x == 0 else "Berlangganan"
        )

        subscription_prob = st.slider(
            "Probabilitas Subscription",
            0.0, 1.0, 0.5
        )

        st.markdown("### Hasil Klasifikasi")
        if subscription_pred == 1:
            st.success(
                f"Pelanggan diprediksi **BERLANGGANAN** "
                f"dengan probabilitas **{subscription_prob:.2f}**"
            )
        else:
            st.warning(
                f"Pelanggan diprediksi **TIDAK BERLANGGANAN** "
                f"dengan probabilitas **{subscription_prob:.2f}**"
            )

        # Visualisasi probabilitas
        fig, ax = plt.subplots()
        ax.bar(["Tidak", "Ya"], [1 - subscription_prob, subscription_prob])
        ax.set_ylabel("Probabilitas")
        st.pyplot(fig)

        st.info("""
**Interpretasi:**
Hasil klasifikasi menunjukkan status subscription pelanggan.
Probabilitas menggambarkan tingkat keyakinan model terhadap prediksi.
""")

    # ---------- UPLOAD CSV ----------
    else:
        st.subheader("Upload CSV Hasil Prediksi")

        file = st.file_uploader(
            "Upload CSV hasil prediksi dari Google Colab",
            type=["csv"],
            key="csv_klasifikasi"
        )

        if file is not None:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            if "subscription_pred" in df.columns:
                fig, ax = plt.subplots()
                df["subscription_pred"].value_counts().plot(kind="bar", ax=ax)
                ax.set_xlabel("Subscription (0 = Tidak, 1 = Ya)")
                ax.set_ylabel("Jumlah Pelanggan")
                st.pyplot(fig)

                st.info("""
**Interpretasi:**
Grafik menunjukkan distribusi pelanggan yang diprediksi
berlangganan dan tidak berlangganan berdasarkan model Random Forest.
""")
        else:
            st.info("Silakan upload file CSV hasil prediksi.")

# ======================================================
# TAB 2 — REGRESI
# ======================================================
with tab2:
    st.header("Regresi Churn Risk")

    mode_reg = st.radio(
        "Pilih metode input:",
        ["✍️ Input Manual", "📂 Upload CSV"],
        key="mode_reg"
    )

    # ---------- INPUT MANUAL ----------
    if mode_reg == "✍️ Input Manual":
        st.subheader("Input Manual (Simulasi 1 Pelanggan)")

        churn_risk = st.slider(
            "Nilai Churn Risk",
            0.0, 1.0, 0.3
        )

        st.markdown("### Hasil Regresi")
        st.metric(
            label="Churn Risk",
            value=f"{churn_risk:.2f}"
        )

        fig, ax = plt.subplots()
        ax.plot([churn_risk], marker="o")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Churn Risk")
        ax.set_xlabel("Index Data")
        st.pyplot(fig)

        st.info("""
**Interpretasi:**
Nilai churn risk menunjukkan tingkat risiko pelanggan berhenti berlangganan.
Semakin mendekati 1, semakin tinggi risiko churn.
""")

    # ---------- UPLOAD CSV ----------
    else:
        st.subheader("Upload CSV Hasil Prediksi")

        file = st.file_uploader(
            "Upload CSV hasil prediksi dari Google Colab",
            type=["csv"],
            key="csv_regresi"
        )

        if file is not None:
            df = pd.read_csv(file)
            st.dataframe(df.head())

            if "churn_risk_pred" in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df["churn_risk_pred"].values)
                ax.set_ylabel("Churn Risk")
                ax.set_xlabel("Index Data")
                st.pyplot(fig)

                st.info("""
**Interpretasi:**
Grafik line plot digunakan untuk melihat sebaran dan kestabilan
nilai churn risk pada seluruh data pelanggan.
""")
        else:
            st.info("Silakan upload file CSV hasil prediksi.")

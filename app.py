import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Subscription & Churn Dashboard",
    layout="centered"
)

st.title("📊 Customer Subscription & Churn Analysis")

st.markdown("""
Aplikasi ini menampilkan hasil **klasifikasi subscription** dan **regresi churn risk**
menggunakan metode **Random Forest (Ensemble Method)**.

Proses training model dilakukan di **Google Colab**, sedangkan aplikasi ini digunakan
sebagai **dashboard visualisasi dan interpretasi hasil**.
""")

st.markdown("---")

# =========================
# PILIH MODE INPUT
# =========================
mode = st.radio(
    "Pilih metode input data:",
    ("📂 Upload CSV", "✍️ Input Manual")
)

# =========================
# MODE 1: UPLOAD CSV
# =========================
if mode == "📂 Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV hasil prediksi dari Google Colab",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Hasil Prediksi")
        st.dataframe(df.head())

        st.markdown("---")

        # ===== KLASIFIKASI =====
        if "subscription_pred" in df.columns:
            st.subheader("✅ Hasil Klasifikasi Subscription")

            fig1, ax1 = plt.subplots()
            df["subscription_pred"].value_counts().plot(kind="bar", ax=ax1)
            ax1.set_xlabel("Subscription (0 = Tidak, 1 = Ya)")
            ax1.set_ylabel("Jumlah Pelanggan")
            st.pyplot(fig1)

            st.info("""
**Interpretasi:**
Grafik menunjukkan distribusi pelanggan yang diprediksi berlangganan dan tidak berlangganan.
""")

        st.markdown("---")

        # ===== REGRESI =====
        if "churn_risk_pred" in df.columns:
            st.subheader("📈 Hasil Regresi Churn Risk")

            fig2, ax2 = plt.subplots()
            ax2.plot(df["churn_risk_pred"].values)
            ax2.set_ylabel("Churn Risk")
            ax2.set_xlabel("Index Data")
            st.pyplot(fig2)

            st.info("""
**Interpretasi:**
Nilai churn risk menunjukkan tingkat risiko pelanggan berhenti berlangganan.
Kepadatan grafik disebabkan oleh jumlah data yang besar.
""")

        st.markdown("---")

        st.download_button(
            "⬇️ Download Data Hasil Prediksi",
            df.to_csv(index=False),
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    else:
        st.info("Silakan upload file CSV hasil prediksi.")

# =========================
# MODE 2: INPUT MANUAL
# =========================
else:
    st.subheader("✍️ Input Manual (Simulasi 1 Pelanggan)")

    subscription_pred = st.selectbox(
        "Prediksi Subscription",
        options=[0, 1],
        format_func=lambda x: "Tidak Berlangganan" if x == 0 else "Berlangganan"
    )

    subscription_prob = st.slider(
        "Probabilitas Subscription",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    churn_risk_pred = st.slider(
        "Churn Risk",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    st.markdown("---")
    st.subheader("📌 Hasil Interpretasi")

    if subscription_pred == 1:
        st.success(
            f"Pelanggan diprediksi **BERLANGGANAN** "
            f"dengan probabilitas **{subscription_prob:.2f}** "
            f"dan churn risk **{churn_risk_pred:.2f}**."
        )
    else:
        st.warning(
            f"Pelanggan diprediksi **TIDAK BERLANGGANAN** "
            f"dengan probabilitas **{subscription_prob:.2f}** "
            f"dan churn risk **{churn_risk_pred:.2f}**."
        )

    st.info("""
**Catatan:**
Mode input manual digunakan untuk simulasi dan interpretasi hasil prediksi
pada satu pelanggan secara individual.
""")

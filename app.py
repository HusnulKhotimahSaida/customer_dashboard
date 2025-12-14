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
Aplikasi ini digunakan untuk **menampilkan hasil analisis data pelanggan** yang meliputi:

- **Klasifikasi Subscription** (berlangganan / tidak berlangganan)
- **Regresi Churn Risk** (tingkat risiko pelanggan berhenti berlangganan)

Model yang digunakan adalah **Random Forest (Ensemble Method)**.  
Proses pelatihan dan evaluasi model dilakukan di **Google Colab**, sedangkan aplikasi ini berfungsi sebagai **dashboard visualisasi hasil prediksi**.
""")

st.markdown("---")

# =========================
# UPLOAD CSV
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload CSV hasil prediksi dari Google Colab",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =========================
    # TAMPIL DATA
    # =========================
    st.subheader("📄 Data Hasil Prediksi")
    st.write(
        "Tabel berikut menampilkan sebagian data pelanggan beserta "
        "hasil prediksi klasifikasi dan regresi."
    )
    st.dataframe(df.head())

    st.markdown("---")

    # =========================
    # KLASIFIKASI
    # =========================
    if "subscription_pred" in df.columns:
        st.subheader("✅ Hasil Klasifikasi Subscription")

        fig1, ax1 = plt.subplots()
        df["subscription_pred"].value_counts().plot(
            kind="bar",
            ax=ax1
        )
        ax1.set_xlabel("Subscription (0 = Tidak, 1 = Ya)")
        ax1.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig1)

        # INTERPRETASI KLASIFIKASI
        st.info("""
**Interpretasi Klasifikasi:**

Grafik menunjukkan distribusi pelanggan yang diprediksi **berlangganan (1)** dan 
**tidak berlangganan (0)**. Perbedaan jumlah pada masing-masing kelas menunjukkan 
bagaimana model Random Forest mengelompokkan pelanggan berdasarkan karakteristiknya.  
Nilai probabilitas (jika tersedia pada data) menunjukkan tingkat keyakinan model 
terhadap hasil prediksi tersebut.
""")

    st.markdown("---")

    # =========================
    # REGRESI
    # =========================
    if "churn_risk_pred" in df.columns:
        st.subheader("📈 Hasil Regresi Churn Risk")

        fig2, ax2 = plt.subplots()
        ax2.plot(df["churn_risk_pred"].values)
        ax2.set_ylabel("Churn Risk")
        ax2.set_xlabel("Index Data")
        st.pyplot(fig2)

        # INTERPRETASI REGRESI
        st.info("""
**Interpretasi Regresi:**

Grafik menampilkan sebaran nilai **churn risk** untuk seluruh pelanggan.  
Nilai churn risk menunjukkan tingkat risiko pelanggan untuk berhenti berlangganan,
di mana nilai yang lebih tinggi menandakan risiko churn yang lebih besar.

Kepadatan grafik disebabkan oleh jumlah data yang besar, sehingga visualisasi ini 
digunakan untuk melihat pola dan variasi churn risk secara keseluruhan.  
Model Random Forest dipilih karena mampu menangani hubungan non-linear dan 
lebih robust terhadap outlier.
""")

    st.markdown("---")

    # =========================
    # DOWNLOAD
    # =========================
    st.download_button(
        "⬇️ Download Data Hasil Prediksi",
        df.to_csv(index=False),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Silakan upload file CSV hasil prediksi dari Google Colab "
        "untuk melihat hasil analisis dan interpretasi."
    )

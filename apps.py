import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Penjualan Southwind", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Membaca file CSV yang diunggah
    url = "https://github.com/seiba1/tugas_visdat/blob/main/Southwind.csv"
    df = pd.read_csv(url)
    # Membersihkan kolom (opsional, menyesuaikan dengan data Southwind)
    df = df[['Penjualan', 'jumlah', 'diskon', 'keuntungan']].dropna()
    return df

# --- TRAINING MODEL ---
def train_model(df):
    # Fitur: Penjualan, Jumlah, Diskon | Target: Keuntungan
    X = df[['Penjualan', 'jumlah', 'diskon']]
    y = df['keuntungan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2, (y_test, y_pred)

# --- MAIN APP ---
def main():
    st.title("📊 Southwind Sales Dashboard & Predictor")
    st.write("Aplikasi untuk menganalisis data pesanan dan memprediksi keuntungan.")

    try:
        df_orders = load_data()
        model, r2, eval_data = train_model(df_orders)

        # Sidebar untuk Input Prediksi
        st.sidebar.header("📥 Input Data Pesanan Baru")
        input_penjualan = st.sidebar.number_input("Total Penjualan ($)", min_value=0.0, value=100.0)
        input_jumlah = st.sidebar.slider("Jumlah Barang (Quantity)", 1, 20, 3)
        input_diskon = st.sidebar.slider("Diskon (0.0 - 0.8)", 0.0, 0.8, 0.0, step=0.1)

        # Layout Utama
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔮 Prediksi Keuntungan")
            new_data = [[input_penjualan, input_jumlah, input_diskon]]
            prediction = model.predict(new_data)[0]
            
            st.metric(label="Estimasi Keuntungan (Profit)", value=f"$ {prediction:.2f}")
            st.info(f"Akurasi Model (R² Score): {r2:.4f}")

            st.write("---")
            st.subheader("📝 Ringkasan Data")
            st.write(df_orders.describe())

        with col2:
            st.subheader("📈 Visualisasi: Aktual vs Prediksi")
            y_test, y_pred = eval_data
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Keuntungan Aktual')
            ax.set_ylabel('Keuntungan Prediksi')
            st.pyplot(fig)

        # Menampilkan tabel data
        if st.checkbox("Lihat Mentah Data Pesanan"):
            st.dataframe(df_orders.head(50))

    except FileNotFoundError:
        st.error("File 'Southwind.xlsx - Orders.csv' tidak ditemukan. Pastikan file ada di folder yang sama dengan apps.py.")

if __name__ == "__main__":
    main()

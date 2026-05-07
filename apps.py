import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Southwind Sales Predictor", layout="wide")

# 2. Fungsi Load Data (Menggunakan Link Raw Anda)
@st.cache_data
def load_data():
    # Link Raw yang Anda berikan
    url = "https://github.com/seiba1/tugas_visdat/raw/refs/heads/main/Southwind.csv"
    
    try:
        # Membaca CSV dengan penanganan baris bermasalah (on_bad_lines)
        # quotechar='"' penting karena nama produk di Southwind mengandung koma
        df = pd.read_csv(
            url, 
            sep=',', 
            quotechar='"', 
            on_bad_lines='skip', 
            engine='python'
        )
        
        # Nama kolom harus persis sesuai file (Penjualan pakai P besar)
        cols_needed = ['Penjualan', 'jumlah', 'diskon', 'keuntungan']
        
        # Pastikan kolom tersedia di dataset
        if not set(cols_needed).issubset(df.columns):
            st.error(f"Kolom tidak lengkap! Kolom yang ditemukan: {df.columns.tolist()}")
            return pd.DataFrame()

        # Ambil kolom yang diperlukan dan bersihkan data non-numerik
        df = df[cols_needed].copy()
        for col in cols_needed:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
    
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# 3. Fungsi Training Model
def train_model(df):
    # Fitur (X) dan Target (y)
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

# 4. Aplikasi Utama
def main():
    st.title("📊 Southwind Sales Dashboard & Predictor")
    st.write("Aplikasi prediksi keuntungan berdasarkan data Penjualan, Jumlah, dan Diskon.")

    df_orders = load_data()

    if not df_orders.empty:
        model, r2, eval_data = train_model(df_orders)

        # --- SIDEBAR INPUT ---
        st.sidebar.header("📥 Input Data Baru")
        val_penjualan = st.sidebar.number_input("Total Penjualan ($)", min_value=0.0, value=150.0)
        val_jumlah = st.sidebar.slider("Jumlah Barang", 1, 50, 5)
        val_diskon = st.sidebar.slider("Diskon (0.0 - 0.8)", 0.0, 0.8, 0.1, step=0.05)

        # --- LAYOUT UTAMA ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔮 Hasil Prediksi Keuntungan")
            # Prediksi berdasarkan input user
            input_features = np.array([[val_penjualan, val_jumlah, val_diskon]])
            prediction = model.predict(input_features)[0]
            
            st.metric(label="Estimasi Profit", value=f"$ {prediction:.2f}")
            st.info(f"Akurasi Model (R² Score): {r2:.4f}")
            
            st.write("---")
            st.subheader("📝 Statistik Deskriptif")
            st.write(df_orders.describe())

        with col2:
            st.subheader("📈 Grafik: Aktual vs Prediksi")
            y_test, y_pred = eval_data
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5, color='seagreen', edgecolors='k')
            # Garis diagonal prediksi sempurna
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--', lw=2, label="Prediksi Ideal")
            
            ax.set_xlabel('Keuntungan Aktual ($)')
            ax.set_ylabel('Keuntungan Prediksi ($)')
            ax.legend()
            st.pyplot(fig)

        # Bagian tabel data
        if st.checkbox("Tampilkan Tabel Data (50 Baris Pertama)"):
            st.dataframe(df_orders.head(50), use_container_width=True)
    else:
        st.warning("Data tidak tersedia. Pastikan file CSV di GitHub sudah benar dan URL Raw valid.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Southwind Sales Predictor", layout="centered")

# 2. Fungsi Load Data dengan Penanganan Semicolon (Titik Koma)
@st.cache_data
def load_data():
    url = "https://github.com/seiba1/tugas_visdat/raw/refs/heads/main/Southwind.csv"
    
    encodings = ['utf-8', 'windows-1252', 'latin1']
    
    df = None
    for enc in encodings:
        try:
            # Menggunakan sep=';' karena file Anda menggunakan titik koma
            df = pd.read_csv(
                url, 
                encoding=enc,
                sep=';', 
                quotechar='"', 
                on_bad_lines='skip', 
                engine='python'
            )
            
            # Cek apakah kolom utama ada, jika tidak coba pakai pemisah koma
            if 'Penjualan' not in df.columns:
                df = pd.read_csv(url, encoding=enc, sep=',', on_bad_lines='skip', engine='python')
                
            break
        except Exception:
            continue
    
    if df is None or df.empty:
        return pd.DataFrame()

    try:
        # Nama kolom yang dibutuhkan
        cols_needed = ['Penjualan', 'jumlah', 'diskon', 'keuntungan']
        
        if not set(cols_needed).issubset(df.columns):
            st.error(f"Kolom tidak ditemukan! Kolom yang terbaca: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df[cols_needed].copy()
        
        # Konversi data ke numerik (menangani format desimal koma)
        for col in cols_needed:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
    
    except Exception as e:
        st.error(f"Kesalahan proses data: {e}")
        return pd.DataFrame()

# 3. Fungsi Training Model
def train_model(df):
    X = df[['Penjualan', 'jumlah', 'diskon']]
    y = df['keuntungan']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

# 4. Aplikasi Utama
def main():
    st.title("📊 Southwind Sales Predictor")
    st.write("Prediksi keuntungan berdasarkan Penjualan, Jumlah, dan Diskon.")

    df_orders = load_data()

    if not df_orders.empty:
        model, r2 = train_model(df_orders)

        # --- BAGIAN INPUT ---
        st.divider()
        st.subheader("📥 Input Data Pesanan")
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        
        with col_inp1:
            val_penjualan = st.number_input("Total Penjualan ($)", min_value=0.0, value=150.0)
        with col_inp2:
            val_jumlah = st.number_input("Jumlah Barang", min_value=1, value=5)
        with col_inp3:
            val_diskon = st.slider("Diskon", 0.0, 0.8, 0.1, step=0.05)

        # --- HASIL PREDIKSI ---
        st.divider()
        input_features = np.array([[val_penjualan, val_jumlah, val_diskon]])
        prediction = model.predict(input_features)[0]
        
        c1, c2 = st.columns(2)
        c1.metric(label="Estimasi Profit", value=f"$ {prediction:.2f}")
        c2.metric(label="Akurasi Model (R²)", value=f"{r2:.4f}")

        # --- INFORMASI DATA ---
        st.divider()
        tab1, tab2 = st.tabs(["📝 Statistik Deskriptif", "📄 Tabel Data"])
        
        with tab1:
            st.write(df_orders.describe())
            
        with tab2:
            st.dataframe(df_orders.head(100), use_container_width=True)
            
    else:
        st.warning("Data tidak tersedia. Periksa URL Raw GitHub atau pemisah (separator) pada file CSV Anda.")

if __name__ == "__main__":
    main()

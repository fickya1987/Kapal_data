import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import subprocess
import os
import warnings

# Install required libraries if not already installed
try:
    import openai
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "openai"])
    import openai

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key exists
if not openai.api_key:
    raise ValueError("API Key OpenAI tidak ditemukan. Harap tambahkan ke file .env.")

warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title="Analytical Dashboard Petikemas Pelindo", layout="wide")

# Header with logo and title
st.image("pelindo_logo.jfif", use_column_width=True)  # Replace with the actual path to the Pelindo logo
st.title("Analytical Dashboard Petikemas Pelindo")
st.markdown("---")


# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Dashboard", "Analisis Terminal", "Visualisasi Berdasarkan Kategori", "Prediction", "Preprocessing"])


# Fungsi untuk membaca data dari file
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")
        return None

# Fungsi untuk preprocessing data
def preprocess_data(data):
    try:
        # Debug awal: Menampilkan data asli
        st.write("Data awal yang diterima:")
        st.write(data.head())

        # Pastikan kolom 'Date' ada
        if 'Date' not in data.columns:
            raise ValueError("Kolom 'Date' tidak ditemukan pada data.")

        # Debug sebelum konversi
        st.write("Sebelum konversi 'Date':")
        st.write(data['Date'].head())

        # Konversi kolom 'Date' ke format datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M', errors='coerce')

        # Debug setelah konversi
        st.write("Setelah konversi 'Date':")
        st.write(data.head())

        # Hapus baris dengan tanggal tidak valid
        data = data.dropna(subset=['Date'])

        # Debug setelah drop NA
        st.write("Setelah menghapus baris dengan 'Date' tidak valid:")
        st.write(data.head())

        # Isi nilai NaN di kolom lain dengan 0
        data.fillna(0, inplace=True)

        # Debug setelah pengisian NaN
        st.write("Setelah mengisi nilai kosong dengan 0:")
        st.write(data.head())

        return data
    except Exception as e:
        st.error(f"Kesalahan saat preprocessing data: {e}")
        return None


# Fungsi untuk analisis AI menggunakan GPT-4
def generate_ai_analysis(data, context):
    try:
        data_summary = data.to_string(index=False, max_rows=5)
        messages = [
            {"role": "system", "content": "Anda adalah seorang analis data yang mahir."},
            {"role": "user", "content": f"Berikan analisis naratif berdasarkan data berikut:\n\n{data_summary}\n\nKonsep: {context}. Tuliskan analisis dengan narasi yang jelas dan terstruktur."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses analisis AI: {e}"

# Dashboard
if menu == "Dashboard":
    st.title("Dashboard Data Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            data = preprocess_data(data)

            if 'Date' not in data.columns or 'Satuan' not in data.columns:
                st.error("Kolom 'Date' atau 'Satuan' tidak ditemukan pada file yang diunggah.")
            else:
                satuan_list = data['Satuan'].unique()
                selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)

                if selected_satuan:
                    data_filtered = data[data['Satuan'] == selected_satuan]
                    st.write(f"Data yang Difilter (Satuan: {selected_satuan}):")
                    st.write(data_filtered)

                    if st.button("Generate AI Analysis - Dashboard"):
                        ai_analysis = generate_ai_analysis(data_filtered, "Dashboard - Analisis Data Petikemas")
                        st.subheader("Hasil Analisis AI:")
                        st.write(ai_analysis)

# Analisis Terminal
elif menu == "Analisis Terminal":
    st.title("Analisis Terminal Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            data = preprocess_data(data)

            if 'Terminal' not in data.columns:
                st.error("Kolom 'Terminal' tidak ditemukan pada file yang diunggah.")
            else:
                terminal_list = data['Terminal'].unique()
                selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)

                if selected_terminal:
                    terminal_data = data[data['Terminal'] == selected_terminal]
                    st.write(f"Data untuk Terminal '{selected_terminal}':")
                    st.write(terminal_data)

                    if st.button("Generate AI Analysis - Terminal"):
                        ai_analysis = generate_ai_analysis(terminal_data, "Analisis Terminal Petikemas")
                        st.subheader("Hasil Analisis AI:")
                        st.write(ai_analysis)

# Visualisasi Berdasarkan Kategori
elif menu == "Visualisasi Berdasarkan Kategori":
    st.title("Visualisasi Berdasarkan Kategori Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            data = preprocess_data(data)

            kategori_list = ['JenisPerdagangan', 'JenisKegiatan', 'IsiKargo', 'TipePetiKemas', 'UkuranPetiKemas']
            selected_kategori = st.sidebar.selectbox("Pilih Kategori", kategori_list)

            if selected_kategori in data.columns:
                category_data = data.groupby(selected_kategori)['Value'].sum().reset_index()
                st.subheader(f"Volume Berdasarkan {selected_kategori}")
                fig = px.pie(category_data, values='Value', names=selected_kategori, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                if st.button(f"Generate AI Analysis - {selected_kategori}"):
                    ai_analysis = generate_ai_analysis(category_data, f"Visualisasi Berdasarkan {selected_kategori}")
                    st.subheader("Hasil Analisis AI:")
                    st.write(ai_analysis)

# Prediction
elif menu == "Prediction":
    st.title("Prediksi Data Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data berhasil dimuat!")
            data = preprocess_data(data)

            if 'Date' not in data.columns or 'Value' not in data.columns:
                st.error("Kolom 'Date' atau 'Value' tidak ditemukan pada file yang diunggah.")
            else:
                data_grouped = data.groupby('Date')['Value'].sum().reset_index()

                if len(data_grouped) < 12:
                    st.error("Data terlalu sedikit untuk prediksi. Tambahkan data dengan rentang waktu yang lebih panjang.")
                else:
                    forecast_period = st.number_input("Masukkan periode prediksi (dalam bulan)", min_value=1, max_value=24, value=6, step=1)

                    try:
                        model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                                        enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit()

                        future = results.get_forecast(steps=forecast_period)
                        forecast = future.predicted_mean
                        conf_int = future.conf_int()

                        forecast_dates = pd.date_range(start=data_grouped['Date'].iloc[-1], periods=forecast_period + 1, freq='M')[1:]
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Predicted Value': forecast.values,
                            'Lower Bound': conf_int.iloc[:, 0].values,
                            'Upper Bound': conf_int.iloc[:, 1].values
                        })

                        # Terapkan batasan pada nilai prediksi
                        forecast_df['Predicted Value'] = forecast_df['Predicted Value'].apply(lambda x: max(x, 0))
                        forecast_df['Lower Bound'] = forecast_df['Lower Bound'].apply(lambda x: max(x, 0))
                        forecast_df['Upper Bound'] = forecast_df['Upper Bound'].apply(lambda x: max(x, 0))

                        st.subheader("Hasil Prediksi")
                        st.write(forecast_df)

                        fig = px.line(forecast_df, x='Date', y='Predicted Value', title="Prediksi Data Petikemas")
                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
                        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
                        st.plotly_chart(fig, use_container_width=True)

                        if st.button("Generate AI Analysis - Prediction"):
                            ai_analysis = generate_ai_analysis(forecast_df, "Prediksi Data Petikemas")
                            st.subheader("Hasil Analisis AI:")
                            st.write(ai_analysis)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")


# Preprocessing
elif menu == "Preprocessing":
    st.title("Preprocessing Data Petikemas")
    uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Data asli:")
            st.write(data.head())

            try:
                preprocessed_data = preprocess_data(data)
                if preprocessed_data is not None:
                    st.write("Data setelah preprocessing:")
                    st.write(preprocessed_data.head())

                    # Unduh data hasil preprocessing
                    csv = preprocessed_data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Unduh Data Preprocessed", data=csv, file_name="preprocessed_data.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Kesalahan selama preprocessing: {e}")


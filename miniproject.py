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

st.set_page_config(page_title="Kapal Analysis", layout="wide")

# Sidebar menu
menu = st.sidebar.selectbox("Pilih Menu", ["Pilih Kategori", "Dashboard", "Prediction"])

# Function to load and preprocess data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")
        return None

# Function for AI analysis using GPT-4
def generate_ai_analysis(data, context):
    """
    Generate AI analysis using GPT-4 based on the provided data and context.
    """
    try:
        # Convert data to a summarized string
        data_summary = data.to_string(index=False, max_rows=5)  # Show only top 5 rows
        messages = [
            {"role": "system", "content": "Anda adalah seorang analis data yang mahir."},
            {"role": "user", "content": f"Berikan analisis naratif berdasarkan data berikut:\n\n{data_summary}\n\n"
                                             f"Konsep: {context}. Tuliskan analisis dengan narasi yang jelas dan terstruktur."}
        ]
        # Call OpenAI GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=2048,
            temperature=1.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses analisis AI: {e}"

# Main application logic
uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

data = None
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None and 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date'])

if data is None or data.empty:
    st.warning("Data tidak tersedia atau tidak valid.")
else:
    if menu == "Pilih Kategori":
        st.title("Pilih Kategori Data")

        category = st.radio("Pilih Kategori", ["Petikemas", "Non-Petikemas", "SPJM"])

        if category == "SPJM":
            st.title("Dashboard SPJM")

            if 'JenisServis' in data.columns:
                jenis_servis_list = data['JenisServis'].unique()
                selected_jenis_servis = st.sidebar.selectbox("Pilih Jenis Servis", jenis_servis_list)
                data = data[data['JenisServis'] == selected_jenis_servis]

            if 'Terminal' in data.columns:
                terminal_list = data['Terminal'].unique()
                selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)
                data = data[data['Terminal'] == selected_terminal]

            # Display filtered data
            st.write("Filtered Data")
            st.write(data.head())

            # Visualization
            if 'Value' in data.columns and 'Date' in data.columns:
                st.subheader("Trend Visualization SPJM")
                fig = px.line(data, x='Date', y='Value', title="Trend Data SPJM", markers=True)
                st.plotly_chart(fig)

                # AI Analysis in SPJM
                if st.button("Generate AI Analysis - SPJM"):
                    context = "Analisis SPJM"
                    ai_analysis = generate_ai_analysis(data, context)
                    st.subheader("Hasil Analisis AI SPJM:")
                    st.write(ai_analysis)

    elif menu == "Dashboard":
        st.title("Dashboard Kapal Analysis")

        # Filter options
        st.sidebar.subheader("Filter Data")

        if 'Satuan' in data.columns:
            satuan_list = data['Satuan'].unique()
            selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)
            data = data[data['Satuan'] == selected_satuan]

        if 'JenisServis' in data.columns:
            jenis_servis_list = data['JenisServis'].unique()
            selected_jenis_servis = st.sidebar.selectbox("Pilih Jenis Servis", jenis_servis_list)
            data = data[data['JenisServis'] == selected_jenis_servis]

        if 'JenisKegiatan' in data.columns:
            jenis_kegiatan_list = data['JenisKegiatan'].unique()
            selected_jenis_kegiatan = st.sidebar.selectbox("Pilih Jenis Kegiatan", jenis_kegiatan_list)
            data = data[data['JenisKegiatan'] == selected_jenis_kegiatan]

        if 'JenisVessel' in data.columns:
            jenis_vessel_list = data['JenisVessel'].unique()
            selected_jenis_vessel = st.sidebar.selectbox("Pilih Jenis Vessel", jenis_vessel_list)
            data = data[data['JenisVessel'] == selected_jenis_vessel]

        if 'JenisKapal' in data.columns:
            jenis_kapal_list = data['JenisKapal'].unique()
            selected_jenis_kapal = st.sidebar.selectbox("Pilih Jenis Kapal", jenis_kapal_list)
            data = data[data['JenisKapal'] == selected_jenis_kapal]

        if 'Terminal' in data.columns:
            terminal_list = data['Terminal'].unique()
            selected_terminal = st.sidebar.selectbox("Pilih Terminal", terminal_list)
            data = data[data['Terminal'] == selected_terminal]

        # Display filtered data
        st.write("Filtered Data")
        st.write(data.head())

        # Visualization
        if 'Value' in data.columns and 'Date' in data.columns:
            st.subheader("Trend Visualization")
            fig = px.line(data, x='Date', y='Value', title="Trend Data", markers=True)
            st.plotly_chart(fig)

            # AI Analysis in Dashboard
            if st.button("Generate AI Analysis - Dashboard"):
                context = "Analisis Dashboard"
                ai_analysis = generate_ai_analysis(data, context)
                st.subheader("Hasil Analisis AI Dashboard:")
                st.write(ai_analysis)

    elif menu == "Prediction":
        st.title("Prediction Kapal Analysis")

        # Ensure required columns are present
        required_columns = ['Date', 'Value']
        if all(col in data.columns for col in required_columns):
            st.write("Data for Prediction")
            st.write(data.head())

            forecast_period = st.number_input("Masukkan Periode Prediksi (bulan)", min_value=1, max_value=24, value=6)

            # Aggregate data for modeling
            data_grouped = data.groupby('Date')['Value'].sum().reset_index()
            data_grouped = data_grouped.sort_values('Date')

            if len(data_grouped) >= 12:
                try:
                    # Build SARIMAX model
                    model = SARIMAX(data_grouped['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
                    results = model.fit()

                    # Forecast
                    future = results.get_forecast(steps=forecast_period)
                    forecast = future.predicted_mean
                    conf_int = future.conf_int()

                    # Prepare forecast dataframe
                    forecast_dates = pd.date_range(start=data_grouped['Date'].iloc[-1], periods=forecast_period+1, freq='M')[1:]
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast.values,
                        'Lower Bound': conf_int.iloc[:, 0],
                        'Upper Bound': conf_int.iloc[:, 1]
                    })

                    st.write("Forecast Data")
                    st.write(forecast_df)

                    # Visualization
                    fig = px.line(forecast_df, x='Date', y='Forecast', title="Forecast Results")
                    fig.add_scatter(x=forecast_df['Date'], y='Lower Bound', mode='lines', name='Lower Bound', line=dict(dash='dot'))
                    fig.add_scatter(x=forecast_df['Date'], y='Upper Bound', mode='lines', name='Upper Bound', line=dict(dash='dot'))
                    st.plotly_chart(fig)

                    # AI Analysis in Prediction
                    if st.button("Generate AI Analysis - Prediction"):
                        context = "Analisis Prediksi"
                        ai_analysis = generate_ai_analysis(forecast_df, context)
                        st.subheader("Hasil Analisis AI Prediction:")
                        st.write(ai_analysis)
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
            else:
                st.error("Data insufficient for prediction. Minimum 12 data points required.")
        else:
            st.error("Required columns ('Date', 'Value') not found in the dataset.")

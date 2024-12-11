import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import warnings
import openai

# Set configurations
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SPJM Analysis", layout="wide")

# Function to load and preprocess data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")
        return None

def filter_and_prepare_data(data):
    required_columns = ['JenisKunjungan', 'JenisServis', 'JenisKegiatan', 'JenisVessel',
                        'JenisKapal', 'Terminal', 'Satuan', 'Value', 'Date']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_columns)}")
        return None

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    return data

# AI Analysis Function
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
            temperature=1.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses analisis AI: {e}"

# Main Application Logic
st.title("SPJM Analysis Dashboard")
uploaded_file = st.file_uploader("Unggah File Data (.csv atau .xlsx)", type=["csv", "xlsx"])

data = None
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        data = filter_and_prepare_data(data)

if data is None or data.empty:
    st.warning("Mohon unggah data yang sesuai!")
else:
    # Sidebar filters
    if 'JenisServis' in data.columns:
        selected_jenis_servis = st.sidebar.selectbox("Pilih Jenis Servis", data['JenisServis'].unique())
        data = data[data['JenisServis'] == selected_jenis_servis]

    if 'JenisKegiatan' in data.columns:
        selected_jenis_kegiatan = st.sidebar.selectbox("Pilih Jenis Kegiatan", data['JenisKegiatan'].unique())
        data = data[data['JenisKegiatan'] == selected_jenis_kegiatan]

    if 'JenisVessel' in data.columns:
        selected_jenis_vessel = st.sidebar.selectbox("Pilih Jenis Vessel", data['JenisVessel'].unique())
        data = data[data['JenisVessel'] == selected_jenis_vessel]

    if 'JenisKapal' in data.columns:
        selected_jenis_kapal = st.sidebar.selectbox("Pilih Jenis Kapal", data['JenisKapal'].unique())
        data = data[data['JenisKapal'] == selected_jenis_kapal]

    if 'Terminal' in data.columns:
        selected_terminal = st.sidebar.selectbox("Pilih Terminal", data['Terminal'].unique())
        data = data[data['Terminal'] == selected_terminal]

    if 'Satuan' in data.columns:
        selected_satuan = st.sidebar.selectbox("Pilih Satuan", data['Satuan'].unique())
        data = data[data['Satuan'] == selected_satuan]

    st.write("Filtered Data")
    st.write(data.head())

    # Aggregation and Visualization
    if 'Value' in data.columns and 'Date' in data.columns:
        aggregated_data = data.groupby('Date')['Value'].sum().reset_index()

        st.subheader("Trend Visualization")
        chart_type = st.selectbox("Pilih Jenis Chart", ["Line Chart", "Scatter Plot", "Histogram"])

        if chart_type == "Line Chart":
            fig = px.line(aggregated_data, x='Date', y='Value', title="Trend Data SPJM", markers=True)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(aggregated_data, x='Date', y='Value', title="Scatter Data SPJM")
        elif chart_type == "Histogram":
            fig = px.histogram(data, x='Value', title="Histogram Data SPJM")

        st.plotly_chart(fig)

        # AI Analysis Button
        if st.button("Generate AI Analysis - Visualization"):
            ai_analysis = generate_ai_analysis(aggregated_data, "Trend Visualization SPJM")
            st.subheader("Hasil Analisis AI:")
            st.write(ai_analysis)

    # Prediction
    st.subheader("Prediction SPJM")
    forecast_period = st.number_input("Masukkan Periode Prediksi (bulan)", min_value=1, max_value=24, value=6)

    if len(aggregated_data) >= 12:
        try:
            model = SARIMAX(aggregated_data['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            results = model.fit()

            future = results.get_forecast(steps=forecast_period)
            forecast = future.predicted_mean.clip(lower=0)
            conf_int = future.conf_int().clip(lower=0)

            forecast_dates = pd.date_range(start=aggregated_data['Date'].iloc[-1], periods=forecast_period+1, freq='M')[1:]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast.values,
                'Lower Bound': conf_int.iloc[:, 0].values,
                'Upper Bound': conf_int.iloc[:, 1].values
            })

            st.write("Forecast Data")
            st.write(forecast_df)

            fig_forecast = px.line(forecast_df, x='Date', y='Forecast', title="Prediction Results")
            fig_forecast.add_scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dot'))
            fig_forecast.add_scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dot'))
            st.plotly_chart(fig_forecast)

            # AI Analysis Button for Prediction
            if st.button("Generate AI Analysis - Prediction"):
                ai_prediction_analysis = generate_ai_analysis(forecast_df, "Prediction Analysis SPJM")
                st.subheader("Hasil Analisis AI Prediksi:")
                st.write(ai_prediction_analysis)

        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Data insufficient for prediction. Minimum 12 data points required.")





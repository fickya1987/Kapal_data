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
    st.warning("Silahkan Input Data format csv atau excel")
else:
    st.title("Dashboard SPJM")

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

    if 'Satuan' in data.columns:
        satuan_list = data['Satuan'].unique()
        selected_satuan = st.sidebar.selectbox("Pilih Satuan", satuan_list)
        data = data[data['Satuan'] == selected_satuan]

    # Display filtered data
    st.write("Filtered Data")
    st.write(data.head())

    # Aggregation based on Satuan
    if selected_satuan == "Call":
        aggregated_data = data.groupby('Date')['Value'].sum().reset_index()
        aggregation_title = "Total Calls"
    elif selected_satuan == "GT":
        aggregated_data = data.groupby('Date')['Value'].sum().reset_index()
        aggregation_title = "Total Gross Tonnage (GT)"
    else:
        st.warning("Satuan tidak dikenal. Data akan ditampilkan tanpa agregasi.")
        aggregated_data = data
        aggregation_title = "Raw Data"

    # Visualization
    if 'Value' in aggregated_data.columns and 'Date' in aggregated_data.columns:
        st.subheader(f"Trend Visualization SPJM ({aggregation_title})")
        fig = px.line(aggregated_data, x='Date', y='Value', title=f"Trend Data SPJM ({aggregation_title})", markers=True)
        st.plotly_chart(fig)

        # AI Analysis in SPJM
        if st.button("Generate AI Analysis - SPJM"):
            context = f"Analisis SPJM berdasarkan {aggregation_title}"
            ai_analysis = generate_ai_analysis(aggregated_data, context)
            st.subheader("Hasil Analisis AI SPJM:")
            st.write(ai_analysis)




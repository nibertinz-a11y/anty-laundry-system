"""
ANTY LAUNDRY - SISTEM SEGMENTASI PELANGGAN
Menggunakan K-Means Clustering & RFM Analysis
VERSI SKRIPSI - Data 1 Bulan Terakhir

Author: Anty Laundry Team
Version: 2.1 (FIXED: Adi & Filke masuk, TOP 10 di atas, WA editable, penjelasan segmen)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import urllib.parse

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Anty Laundry - K-Means Clustering",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark background gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }
    
    /* Main container - dark theme */
    .main .block-container {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(30px);
        border-radius: 0px;
        padding: 3rem 4rem;
        box-shadow: none;
        border: none;
        max-width: 100%;
    }
    
    /* Hero Header - Modern Dark */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 3rem 3rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 2.5rem;
        position: relative;
        z-index: 1;
        flex-wrap: nowrap;
    }
    
    .logo-img {
        height: 90px;
        width: 90px;
        border-radius: 20px;
        background: white;
        padding: 12px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        object-fit: contain;
        flex-shrink: 0;
    }
    
    .header-text {
        flex: 1;
        min-width: 0;
    }
    
    .header-text h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }
    
    .header-text h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
        letter-spacing: 0.3px;
    }
    
    .header-text p {
        margin: 0;
        font-size: 0.95rem;
        opacity: 0.85;
        font-weight: 400;
    }
    
    /* Sidebar - Dark Purple */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f3f4f6 !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #e5e7eb !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f3f4f6 !important;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 2rem 1rem;
        background: rgba(139, 92, 246, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    .sidebar-logo img {
        width: 100%;
        max-width: 180px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    /* Large Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        border: none;
        border-radius: 16px;
        padding: 1.2rem 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
        text-transform: none;
        letter-spacing: 0.5px;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    }
    
    /* Metrics Cards - Glass Effect */
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        backdrop-filter: blur(20px);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        background: rgba(99, 102, 241, 0.12);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        font-weight: 900;
        color: #a78bfa;
        letter-spacing: -1px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Alert boxes - Modern */
    .stAlert {
        border-radius: 16px;
        border: none;
        backdrop-filter: blur(10px);
        font-weight: 500;
        padding: 1.2rem 1.5rem;
    }
    
    div[data-baseweb="notification"][kind="info"] {
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        color: #93c5fd;
    }
    
    div[data-baseweb="notification"][kind="success"] {
        background: rgba(34, 197, 94, 0.15);
        border-left: 4px solid #22c55e;
        color: #86efac;
    }
    
    div[data-baseweb="notification"][kind="warning"] {
        background: rgba(251, 146, 60, 0.15);
        border-left: 4px solid #fb923c;
        color: #fdba74;
    }
    
    div[data-baseweb="notification"][kind="error"] {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #ef4444;
        color: #fca5a5;
    }
    
    /* Expanders - Dark Style */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #c4b5fd;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
        color: #ddd6fe;
    }
    
    /* DataFrames - Dark */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.2);
        background: rgba(30, 27, 75, 0.5);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        padding: 1rem;
        border: none;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
        color: #e5e7eb;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(99, 102, 241, 0.08);
    }
    
    .dataframe tbody tr td {
        padding: 0.9rem 1rem;
        border: none;
    }
    
    /* File Uploader - Hero Style */
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(99, 102, 241, 0.08);
        border-radius: 24px;
        padding: 3rem;
        border: 2px dashed rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(139, 92, 246, 0.7);
        background: rgba(99, 102, 241, 0.12);
    }
    
    section[data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 600;
    }
    
    /* Download Buttons - Success Color */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 14px;
        padding: 1rem 1.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.4);
    }
    
    /* WhatsApp Button */
    .stLinkButton>a {
        background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 14px;
        padding: 1rem 1.8rem;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(37, 211, 102, 0.3);
    }
    
    .stLinkButton>a:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(37, 211, 102, 0.4);
    }
    
    /* Divider */
    hr {
        margin: 3rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    }
    
    /* Plotly Charts - Floating Effect */
    .js-plotly-plot {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(99, 102, 241, 0.2);
        background: rgba(30, 27, 75, 0.3);
    }
    
    /* Headings - Light on Dark */
    h1, h2, h3, h4, h5, h6 {
        color: #f3f4f6;
    }
    
    h2 {
        font-weight: 800;
        font-size: 2.2rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    
    h3 {
        color: #c4b5fd;
        font-weight: 700;
        font-size: 1.4rem;
    }
    
    /* Paragraphs */
    p {
        color: #d1d5db;
        line-height: 1.7;
    }
    
    /* Text area - Dark */
    textarea {
        background: rgba(30, 27, 75, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 14px !important;
        color: #e5e7eb !important;
        font-family: 'Inter', monospace;
    }
    
    textarea:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 27, 75, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.7);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI UTAMA: PROSES DATA & CLUSTERING
# ============================================================

class AntyLaundryKMeans:
    """Engine untuk K-Means Clustering dengan RFM Analysis"""
    
    def __init__(self):
        self.n_clusters = 5
        self.model = None
        self.scaler = StandardScaler()
        
    def find_column(self, df, keywords):
        """Mencari kolom berdasarkan keyword - prioritas exact match"""
        for keyword in keywords:
            for col in df.columns:
                if str(col).lower().strip() == keyword.lower():
                    return col
        
        for keyword in keywords:
            keyword_clean = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                col_clean = str(col).lower().strip().replace(' ', '').replace('_', '')
                if keyword_clean == col_clean:
                    return col
        
        for keyword in keywords:
            keyword_clean = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                col_clean = str(col).lower().strip().replace(' ', '').replace('_', '')
                if keyword_clean in col_clean:
                    return col
        
        return None
    
    def load_and_clean_data(self, df, months_back=1):
        """Membersihkan dan memvalidasi data - DENGAN FILTER PERIODE"""
        df = df.copy()
        
        st.info("🔍 Mencari kolom yang dibutuhkan...")
        
        col_mapping = {}
        
        tanggal_col = self.find_column(df, ['tanggal ar', 'tanggal ambil', 'tgl ambil', 'tanggalambil'])
        if tanggal_col:
            col_mapping[tanggal_col] = 'Tanggal'
            st.success(f"✅ Tanggal: **{tanggal_col}** → Tanggal")
        else:
            st.error("❌ Kolom 'Tanggal Ambil' tidak ditemukan!")
            return None
        
        konsumen_col = self.find_column(df, ['konsumer', 'konsumen', 'customer', 'pelanggan'])
        if konsumen_col:
            col_mapping[konsumen_col] = 'Konsumen'
            st.success(f"✅ Konsumen: **{konsumen_col}** → Konsumen")
        else:
            st.error("❌ Kolom 'Konsumen' tidak ditemukan!")
            return None
        
        harga_col = self.find_column(df, ['total harg', 'total harga', 'totalharga'])
        if harga_col:
            col_mapping[harga_col] = 'Total_Harga'
            st.success(f"✅ Total Harga: **{harga_col}** → Total_Harga")
        else:
            st.error("❌ Kolom 'Total Harga' tidak ditemukan!")
            return None
        
        invoice_col = self.find_column(df, ['nota', 'invoice', 'no nota', 'nonota', 'no.nota'])
        if invoice_col:
            col_mapping[invoice_col] = 'No_Invoice'
            st.success(f"✅ No Invoice: **{invoice_col}** → No_Invoice")
        
        status_col = self.find_column(df, ['status order', 'statusorder', 'status'])
        if status_col:
            col_mapping[status_col] = 'Status_Order'
            st.success(f"✅ Status Order: **{status_col}** → Status_Order")
        
        tanggal_order_col = self.find_column(df, ['tanggal order', 'tanggalorder', 'tgl order'])
        if tanggal_order_col:
            col_mapping[tanggal_order_col] = 'Tanggal_Order'
            st.success(f"✅ Tanggal Order: **{tanggal_order_col}** → Tanggal_Order")
        
        df = df.rename(columns=col_mapping)
        
        # 🔴 PERBAIKAN 1: Hanya filter BATAL, sisanya tetap dihitung (termasuk piutang)
        if 'Status_Order' in df.columns:
            before = len(df)
            df = df[~df['Status_Order'].astype(str).str.lower().str.contains('batal', na=False)]
            after = len(df)
            if before > after:
                st.warning(f"⚠️ {before - after} transaksi BATAL dihapus")
        
        # Parse tanggal
        try:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            
            # Jika ada Tanggal Order, gunakan sebagai backup
            if 'Tanggal_Order' in df.columns:
                df['Tanggal_Order'] = pd.to_datetime(df['Tanggal_Order'], errors='coerce')
                df['Tanggal'] = df['Tanggal'].fillna(df['Tanggal_Order'])
            
            df = df.dropna(subset=['Tanggal'])
        except Exception as e:
            st.error(f"❌ Error parsing tanggal: {str(e)}")
            return None
        
        # Filter periode
        max_date = df['Tanggal'].max()
        cutoff_date = max_date - timedelta(days=30 * months_back)
        
        st.info(f"📅 Tanggal maksimal: {max_date.strftime('%d/%m/%Y')}")
        st.info(f"📅 Filter dari: {cutoff_date.strftime('%d/%m/%Y')}")
        
        df_before_filter = len(df)
        df = df[df['Tanggal'] > cutoff_date]
        df_after_filter = len(df)
        
        st.success(f"✅ Data difilter: {df_after_filter} transaksi dari {df_before_filter} (periode {months_back} bulan)")
        st.success(f"✅ Periode: {df['Tanggal'].min().strftime('%d/%m/%Y')} - {df['Tanggal'].max().strftime('%d/%m/%Y')}")
        
        # Filter Total Harga > 0
        if 'Total_Harga' in df.columns:
            df['Total_Harga'] = pd.to_numeric(df['Total_Harga'], errors='coerce')
            df = df[df['Total_Harga'] > 0]
        else:
            st.error(f"❌ Kolom 'Total_Harga' tidak ada setelah rename!")
            return None
        
        # Bersihkan nama konsumen
        df['Konsumen'] = df['Konsumen'].astype(str).str.strip()
        df = df.dropna(subset=['Total_Harga', 'Konsumen'])
        
        st.success(f"✅ Total {len(df)} transaksi valid dari {len(df['Konsumen'].unique())} pelanggan unik")
        
        return df
    
    def calculate_rfm(self, df, reference_date=None):
        """Menghitung nilai RFM"""
        
        if reference_date is None:
            reference_date = df['Tanggal'].max()
        
        st.info(f"📊 Tanggal referensi RFM: {reference_date.strftime('%d/%m/%Y')}")
        
        if 'No_Invoice' in df.columns:
            rfm = df.groupby('Konsumen').agg({
                'Tanggal': lambda x: (reference_date - x.max()).days,
                'No_Invoice': 'nunique',
                'Total_Harga': 'sum'
            }).reset_index()
        else:
            rfm = df.groupby('Konsumen').agg({
                'Tanggal': [lambda x: (reference_date - x.max()).days, 'count'],
                'Total_Harga': 'sum'
            }).reset_index()
            rfm.columns = ['Konsumen', 'Recency', 'Frequency', 'Monetary']
            return rfm
        
        rfm.columns = ['Konsumen', 'Recency', 'Frequency', 'Monetary']
        
        rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce')
        rfm = rfm.dropna()
        
        st.success(f"✅ RFM dihitung untuk {len(rfm)} pelanggan")
        
        st.info(f"📊 Recency: {rfm['Recency'].min():.0f} - {rfm['Recency'].max():.0f} hari")
        st.info(f"📊 Frequency: {rfm['Frequency'].min():.0f} - {rfm['Frequency'].max():.0f} transaksi")
        st.info(f"📊 Monetary: Rp {rfm['Monetary'].min():,.0f} - Rp {rfm['Monetary'].max():,.0f}")
        
        return rfm
    
    def normalize_data(self, rfm_df):
        """Normalisasi data RFM"""
        features = ['Recency', 'Frequency', 'Monetary']
        rfm_scaled = self.scaler.fit_transform(rfm_df[features])
        
        rfm_df['Recency_scaled'] = rfm_scaled[:, 0]
        rfm_df['Frequency_scaled'] = rfm_scaled[:, 1]
        rfm_df['Monetary_scaled'] = rfm_scaled[:, 2]
        
        return rfm_df
    
    def run_kmeans(self, rfm_df):
        """Jalankan K-Means Clustering"""
        X = rfm_df[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']].values
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        rfm_df['Cluster'] = self.model.fit_predict(X)
        
        st.success(f"✅ K-Means clustering selesai dengan {self.n_clusters} cluster")
        st.info(f"📊 Inertia (WCSS): {self.model.inertia_:.2f}")
        
        return rfm_df
    
def label_clusters(self, rfm_df):
        """Label setiap cluster berdasarkan karakteristik RFM dengan RFM Score"""
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Konsumen': 'count'
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        
        # Hitung RFM Score untuk ranking (semakin tinggi semakin baik)
        max_recency = rfm_df['Recency'].max()
        cluster_summary['Recency_Score'] = (max_recency - cluster_summary['Avg_Recency']) / max_recency
        cluster_summary['Frequency_Score'] = cluster_summary['Avg_Frequency'] / rfm_df['Frequency'].max()
        cluster_summary['Monetary_Score'] = cluster_summary['Avg_Monetary'] / rfm_df['Monetary'].max()
        
        # Total RFM Score (0-3, semakin tinggi semakin baik)
        cluster_summary['RFM_Score'] = (
            cluster_summary['Recency_Score'] + 
            cluster_summary['Frequency_Score'] + 
            cluster_summary['Monetary_Score']
        )
        
        # Sort by RFM Score untuk ranking
        cluster_summary = cluster_summary.sort_values('RFM_Score', ascending=False).reset_index(drop=True)
        
        st.info("📊 Karakteristik Cluster (diurutkan dari terbaik):")
        for idx, row in cluster_summary.iterrows():
            st.write(f"**Cluster {row['Cluster']}**: Score={row['RFM_Score']:.2f}, Recency={row['Avg_Recency']:.0f} hari, Freq={row['Avg_Frequency']:.1f}x, Money=Rp{row['Avg_Monetary']:,.0f}, Count={row['Count']}")
        
        # Labeling berdasarkan RANKING, bukan kondisi overlapping
        labels = {}
        
        for idx, row in cluster_summary.iterrows():
            cluster_id = row['Cluster']
            rank = idx + 1  # Ranking dari terbaik (1) ke terburuk (5)
            
            if rank == 1:
                # Cluster terbaik (RFM Score tertinggi)
                labels[cluster_id] = {
                    'name': 'VIP Champions',
                    'icon': '🏆',
                    'discount': 15,
                    'priority': 1,
                    'description': f'Pelanggan TERBAIK dengan RFM Score tertinggi ({row["RFM_Score"]:.2f}/3.0). Rata-rata belanja Rp{row["Avg_Monetary"]:,.0f}, transaksi {row["Avg_Frequency"]:.1f}x, terakhir {row["Avg_Recency"]:.0f} hari lalu. Mereka adalah aset paling berharga - pertahankan dengan pelayanan VIP, reward eksklusif, dan komunikasi personal.',
                    'criteria': f'✓ Ranking #1 dari 5 cluster\n✓ RFM Score: {row["RFM_Score"]:.2f} (tertinggi)\n✓ Recency: {row["Avg_Recency"]:.0f} hari (sangat baru)\n✓ Frequency: {row["Avg_Frequency"]:.1f}x transaksi\n✓ Monetary: Rp{row["Avg_Monetary"]:,.0f}'
                }
            
            elif rank == 2:
                # Cluster kedua terbaik
                labels[cluster_id] = {
                    'name': 'High Value Loyal',
                    'icon': '💎',
                    'discount': 10,
                    'priority': 2,
                    'description': f'Pelanggan SETIA dengan RFM Score tinggi ({row["RFM_Score"]:.2f}/3.0). Rata-rata belanja Rp{row["Avg_Monetary"]:,.0f}, transaksi {row["Avg_Frequency"]:.1f}x, terakhir {row["Avg_Recency"]:.0f} hari lalu. Mereka loyal dan berkontribusi besar - tingkatkan engagement untuk naik ke VIP Champions.',
                    'criteria': f'✓ Ranking #2 dari 5 cluster\n✓ RFM Score: {row["RFM_Score"]:.2f}\n✓ Recency: {row["Avg_Recency"]:.0f} hari\n✓ Frequency: {row["Avg_Frequency"]:.1f}x transaksi\n✓ Monetary: Rp{row["Avg_Monetary"]:,.0f}'
                }
            
            elif rank == 3:
                # Cluster menengah
                labels[cluster_id] = {
                    'name': 'Regular Loyal',
                    'icon': '💚',
                    'discount': 5,
                    'priority': 3,
                    'description': f'Pelanggan REGULER dengan RFM Score menengah ({row["RFM_Score"]:.2f}/3.0). Rata-rata belanja Rp{row["Avg_Monetary"]:,.0f}, transaksi {row["Avg_Frequency"]:.1f}x, terakhir {row["Avg_Recency"]:.0f} hari lalu. Mereka adalah basis pelanggan stabil - tingkatkan dengan promo rutin dan program loyalitas.',
                    'criteria': f'✓ Ranking #3 dari 5 cluster\n✓ RFM Score: {row["RFM_Score"]:.2f} (menengah)\n✓ Recency: {row["Avg_Recency"]:.0f} hari\n✓ Frequency: {row["Avg_Frequency"]:.1f}x transaksi\n✓ Monetary: Rp{row["Avg_Monetary"]:,.0f}'
                }
            
            elif rank == 4:
                # Cluster keempat - at risk
                labels[cluster_id] = {
                    'name': 'At Risk',
                    'icon': '⚠️',
                    'discount': 7,
                    'priority': 4,
                    'description': f'Pelanggan BERISIKO dengan RFM Score rendah ({row["RFM_Score"]:.2f}/3.0). Rata-rata belanja Rp{row["Avg_Monetary"]:,.0f}, transaksi {row["Avg_Frequency"]:.1f}x, terakhir {row["Avg_Recency"]:.0f} hari lalu. Mereka menunjukkan tanda akan berhenti - butuh perhatian khusus dan promo reaktivasi segera.',
                    'criteria': f'✓ Ranking #4 dari 5 cluster\n✓ RFM Score: {row["RFM_Score"]:.2f} (rendah)\n✓ Recency: {row["Avg_Recency"]:.0f} hari (mulai lama)\n✓ Frequency: {row["Avg_Frequency"]:.1f}x transaksi\n✓ Monetary: Rp{row["Avg_Monetary"]:,.0f}'
                }
            
            else:  # rank == 5
                # Cluster terburuk - sleeping
                labels[cluster_id] = {
                    'name': 'Sleeping Customers',
                    'icon': '😴',
                    'discount': 10,
                    'priority': 5,
                    'description': f'Pelanggan TIDUR dengan RFM Score terendah ({row["RFM_Score"]:.2f}/3.0). Rata-rata belanja Rp{row["Avg_Monetary"]:,.0f}, transaksi {row["Avg_Frequency"]:.1f}x, terakhir {row["Avg_Recency"]:.0f} hari lalu. Mereka sudah lama tidak aktif - perlu kampanye win-back agresif dengan diskon besar untuk bangunkan mereka.',
                    'criteria': f'✓ Ranking #5 dari 5 cluster (terburuk)\n✓ RFM Score: {row["RFM_Score"]:.2f} (terendah)\n✓ Recency: {row["Avg_Recency"]:.0f} hari (sangat lama)\n✓ Frequency: {row["Avg_Frequency"]:.1f}x transaksi\n✓ Monetary: Rp{row["Avg_Monetary"]:,.0f}'
                }
        
        rfm_df['Segment'] = rfm_df['Cluster'].map(lambda x: labels[x]['name'])
        rfm_df['Icon'] = rfm_df['Cluster'].map(lambda x: labels[x]['icon'])
        rfm_df['Discount'] = rfm_df['Cluster'].map(lambda x: labels[x]['discount'])
        rfm_df['Priority'] = rfm_df['Cluster'].map(lambda x: labels[x]['priority'])
        rfm_df['Description'] = rfm_df['Cluster'].map(lambda x: labels[x]['description'])
        rfm_df['Criteria'] = rfm_df['Cluster'].map(lambda x: labels[x]['criteria'])
        
        return rfm_df, labels
    
    def get_top_10_customers(self, rfm_df):
        """Pilih TOP 10 pelanggan untuk diskon"""
        top_segments = rfm_df[rfm_df['Segment'].isin(['VIP Champions', 'High Value Loyal'])]
        top_10 = top_segments.nlargest(10, 'Monetary')
        
        if len(top_10) < 10:
            remaining = 10 - len(top_10)
            regular = rfm_df[rfm_df['Segment'] == 'Regular Loyal'].nlargest(remaining, 'Monetary')
            top_10 = pd.concat([top_10, regular])
        
        if len(top_10) < 10:
            remaining = 10 - len(top_10)
            others = rfm_df[~rfm_df['Konsumen'].isin(top_10['Konsumen'])].nlargest(remaining, 'Monetary')
            top_10 = pd.concat([top_10, others])
        
        return top_10.head(10)


# ============================================================
# FUNGSI VISUALISASI
# ============================================================

def create_cluster_distribution_chart(rfm_df):
    """Pie chart distribusi cluster"""
    cluster_counts = rfm_df['Segment'].value_counts()
    
    fig = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title='Distribusi Pelanggan per Segmen',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_rfm_3d_scatter(rfm_df):
    """3D scatter plot RFM"""
    fig = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        hover_data=['Konsumen'],
        title='Visualisasi 3D - RFM Analysis',
        labels={
            'Recency': 'Recency (hari)',
            'Frequency': 'Frequency (transaksi)',
            'Monetary': 'Monetary (Rp)'
        }
    )
    return fig


# ============================================================
# FUNGSI EXPORT & WHATSAPP
# ============================================================

def generate_default_whatsapp_message(top_10):
    """Generate pesan WhatsApp default untuk TOP 10 pelanggan"""
    message = """🎉 *SELAMAT PELANGGAN SETIA ANTY LAUNDRY!* 🎉

Anda terpilih sebagai TOP 10 pelanggan terbaik bulan ini! 🏆

Sebagai apresiasi, Anda berhak mendapat DISKON SPESIAL:

"""
    
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        message += f"{idx}. *{row['Konsumen']}*\n"
        message += f"   💎 Segmen: {row['Segment']}\n"
        message += f"   🎁 Diskon: *{row['Discount']}%*\n"
        message += f"   💰 Total Belanja: Rp {row['Monetary']:,.0f}\n\n"
    
    message += """📅 *Berlaku:* Bulan depan untuk semua layanan
💳 *Cara pakai:* Tunjukkan pesan ini saat transaksi

Terima kasih telah mempercayai ANTY LAUNDRY! 💙

━━━━━━━━━━━━━━━━━━
🧺 ANTY LAUNDRY
📍 Alamat: Tomohon, Sulawesi Utara
📞 Telp: [Isi nomor telepon]
"""
    
    return message

def create_whatsapp_link(message):
    """Generate link WhatsApp dengan pre-filled message"""
    encoded_message = urllib.parse.quote(message)
    return f"https://wa.me/?text={encoded_message}"

def export_to_excel(rfm_df, top_10, cluster_summary):
    """Export hasil ke Excel"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        top_10_export = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        top_10_export.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency (x)', 'Total Belanja (Rp)', 'Diskon (%)']
        top_10_export.to_excel(writer, sheet_name='Top 10 Pelanggan', index=False)
        
        all_customers = rfm_df[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        all_customers.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency (x)', 'Total Belanja (Rp)', 'Diskon (%)']
        all_customers.to_excel(writer, sheet_name='Semua Pelanggan', index=False)
        
        cluster_summary.to_excel(writer, sheet_name='Ringkasan Cluster', index=False)
    
    output.seek(0)
    return output


# ============================================================
# MAIN APP
# ============================================================

def main():
    
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <img src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDZqeHY5cWl3bTF1d2p4dDF6NGw1dHUwcG1yb3M2aTl6Nmd3dXZwOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/jowM6pSgsD9TwLqwje/giphy.gif' 
                 class='logo-img'
                 alt='Anty Laundry Logo'>
            <div class="header-text">
                <h1>🧺 ANTY LAUNDRY</h1>
                <h3>Sistem Segmentasi Pelanggan - K-Means Clustering</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <img src='https://i.imgur.com/BP3MK3t.jpeg' 
                 alt='Anty Laundry Logo'>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Pengaturan Analisis")
        months_back = st.selectbox(
            "📅 Periode Data:",
            options=[1, 2, 3, 6, 12],
            index=0,
            help="Pilih berapa bulan terakhir yang akan dianalisis"
        )
        
        st.info(f"📊 Data akan difilter: **{months_back} bulan terakhir**")
        
        st.markdown("---")
        
        st.markdown("### 📋 Panduan Penggunaan")
        st.markdown("""
        1. 📤 Upload file Excel dari kasir
        2. 🔄 Klik "Jalankan Analisis"
        3. 🏆 Lihat TOP 10 pelanggan
        4. ✏️ Edit pesan WhatsApp
        5. 📥 Download laporan
        """)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Informasi Sistem")
        st.info("""
        **Metode:** K-Means (k=5)
        
        **RFM Analysis:**
        - 📅 Recency: Hari sejak transaksi terakhir
        - 🔄 Frequency: Jumlah transaksi
        - 💰 Monetary: Total belanja
        
        **Catatan:** Semua transaksi dihitung kecuali yang batal
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Segmen Pelanggan")
        st.markdown("""
        🏆 **VIP Champions** → 15%  
        💎 **High Value Loyal** → 10%  
        💚 **Regular Loyal** → 5%  
        ⚠️ **At Risk** → 7%  
        😴 **Sleeping** → 10%
        """)
        
        st.markdown("---")
        st.caption("© 2025 Anty Laundry v2.1")
    
    st.markdown("## 📤 Upload Data Transaksi")
    
    uploaded_file = st.file_uploader(
        "Upload file Excel/CSV dari aplikasi kasir laundry1010dry",
        type=['xlsx', 'xls', 'csv'],
        help="File harus memiliki kolom: Tanggal Ambil, Konsumen, Total Harga"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File berhasil dimuat: **{uploaded_file.name}**")
            st.info(f"📊 Total baris data: {len(df_raw)}")
            
            with st.expander("👀 Preview Data (10 baris pertama)"):
                st.dataframe(df_raw.head(10))
            
            if st.button("🚀 Jalankan Analisis K-Means", type="primary", use_container_width=True):
                
                with st.spinner("⏳ Sedang memproses data..."):
                    
                    engine = AntyLaundryKMeans()
                    
                    st.markdown("### 📊 Step 1: Membersihkan & Filter Data")
                    df_clean = engine.load_and_clean_data(df_raw, months_back=months_back)
                    
                    if df_clean is None:
                        st.error("❌ Gagal memproses data!")
                        st.stop()
                    
                    st.markdown("### 🔢 Step 2: Menghitung RFM")
                    rfm = engine.calculate_rfm(df_clean)
                    
                    st.markdown("### 🔢 Step 3: Normalisasi Data")
                    rfm = engine.normalize_data(rfm)
                    st.success("✅ Data berhasil dinormalisasi")
                    
                    st.markdown("### 🤖 Step 4: K-Means Clustering")
                    rfm = engine.run_kmeans(rfm)
                    
                    st.markdown("### 🏷️ Step 5: Labeling Cluster")
                    rfm, cluster_labels = engine.label_clusters(rfm)
                    st.success("✅ Cluster berhasil dilabeli")
                    
                    st.markdown("### 🏆 Step 6: Memilih TOP 10")
                    top_10 = engine.get_top_10_customers(rfm)
                    st.success(f"✅ {len(top_10)} pelanggan terpilih")
                    
                    st.session_state['rfm_result'] = rfm
                    st.session_state['top_10'] = top_10
                    st.session_state['cluster_labels'] = cluster_labels
                    st.session_state['df_clean'] = df_clean
                
                st.success("✅ Analisis selesai!")
                st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.stop()
    
    # 🔴 PERBAIKAN 2: Tampilkan TOP 10 DI ATAS (setelah analisis selesai)
    if 'rfm_result' in st.session_state:
        
        rfm = st.session_state['rfm_result']
        top_10 = st.session_state['top_10']
        cluster_labels = st.session_state['cluster_labels']
        df_clean = st.session_state['df_clean']
        
        st.markdown("---")
        
        st.markdown("## 🏆 TOP 10 Pelanggan Dapat Diskon Bulan Depan")
        
        st.info("💡 Dipilih otomatis dari segmen **VIP Champions** dan **High Value Loyal** berdasarkan total belanja tertinggi")
        
        top_10_display = top_10.copy()
        top_10_display['Rank'] = range(1, len(top_10_display) + 1)
        top_10_display['Monetary_Formatted'] = top_10_display['Monetary'].apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(
            top_10_display[['Rank', 'Konsumen', 'Segment', 'Frequency', 'Monetary_Formatted', 'Discount']]
            .rename(columns={
                'Rank': '#',
                'Konsumen': 'Nama Pelanggan',
                'Segment': 'Segmen',
                'Frequency': 'Transaksi',
                'Monetary_Formatted': 'Total Belanja',
                'Discount': 'Diskon (%)'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # 🔴 PERBAIKAN 3: Pesan WhatsApp yang bisa diedit
        st.markdown("## 💬 Kirim Pesan ke Pelanggan")
        
        # Inisialisasi session state untuk pesan
        if 'wa_message' not in st.session_state:
            st.session_state['wa_message'] = generate_default_whatsapp_message(top_10)
        
        st.markdown("**✏️ Edit pesan di bawah ini, lalu klik 'Update Pesan' sebelum mengirim:**")
        
        # Text area untuk edit pesan
        edited_message = st.text_area(
            "Pesan WhatsApp",
            value=st.session_state['wa_message'],
            height=400,
            help="Edit pesan sesuai kebutuhan",
            key="wa_message_editor"
        )
        
        # Tombol untuk update pesan
        if st.button("🔄 Update Pesan", use_container_width=True, help="Klik ini setelah selesai edit untuk menyimpan perubahan"):
            st.session_state['wa_message'] = edited_message
            st.success("✅ Pesan berhasil diupdate! Sekarang bisa kirim ke WhatsApp.")
            st.rerun()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Generate link WhatsApp dengan pesan dari session state
            wa_link = create_whatsapp_link(st.session_state['wa_message'])
            
            st.link_button(
                "💬 Kirim ke WhatsApp",
                wa_link,
                use_container_width=True,
                type="primary"
            )
            
        
        with col2:
            cluster_summary = rfm.groupby('Segment').agg({
                'Konsumen': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Discount': 'first'
            }).reset_index()
            
            cluster_summary.columns = ['Segmen', 'Jumlah Pelanggan', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Diskon (%)']
            
            excel_data = export_to_excel(rfm, top_10, cluster_summary)
            
            st.download_button(
                label="📊 Download Laporan Excel",
                data=excel_data,
                file_name=f"Laporan_Segmentasi_KMeans_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            csv = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_csv(index=False)
            
            st.download_button(
                label="📄 Download TOP 10 (CSV)",
                data=csv,
                file_name=f"TOP_10_KMeans_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        st.markdown("## 📊 Hasil Analisis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pelanggan", len(rfm))
        
        with col2:
            st.metric("Total Transaksi", len(df_clean))
        
        with col3:
            period = f"{df_clean['Tanggal'].min().strftime('%d/%m/%Y')} - {df_clean['Tanggal'].max().strftime('%d/%m/%Y')}"
            st.metric("Periode Data", period)
        
        with col4:
            st.metric("Jumlah Cluster", 5)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_cluster_distribution_chart(rfm), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_rfm_3d_scatter(rfm), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("## 🎯 Detail Segmentasi")
        
        st.info("📌 **Penjelasan Segmen:** Setiap segmen memiliki karakteristik RFM yang berbeda. Klik untuk melihat detail pelanggan di setiap segmen beserta strategi yang disarankan.")
        
        for cluster_id in sorted(rfm['Cluster'].unique()):
            cluster_data = rfm[rfm['Cluster'] == cluster_id]
            label_info = cluster_labels[cluster_id]
            
            with st.expander(f"{label_info['icon']} **{label_info['name']}** ({len(cluster_data)} pelanggan) - Diskon {label_info['discount']}%"):
                
                st.markdown("**📝 Deskripsi Segmen:**")
                st.info(label_info['description'])
                
                st.markdown("**🎯 Kriteria & Karakteristik:**")
                st.code(label_info['criteria'], language=None)
                
                st.markdown("---")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.0f} hari")
                
                with col2:
                    st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}x")
                
                with col3:
                    st.metric("Avg Monetary", f"Rp {cluster_data['Monetary'].mean():,.0f}")
                
                with col4:
                    st.metric("Diskon", f"{label_info['discount']}%")
                
                st.markdown("**👥 Daftar Pelanggan:**")
                st.dataframe(
                    cluster_data[['Konsumen', 'Recency', 'Frequency', 'Monetary']].sort_values('Monetary', ascending=False),
                    use_container_width=True
                )
        
        st.markdown("---")
        
        st.success("""
        ✅ **Simpan laporan** dan berikan ke karyawan  
        ✅ **Input diskon** untuk 10 pelanggan di kasir laundry1010dry  
        ✅ **Berlaku mulai** bulan depan  
        ✅ **Upload data baru** setiap bulan untuk update otomatis
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p style='margin: 0.5rem 0; font-size: 1rem;'><strong>© 2025 Anty Laundry</strong></p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Sistem Segmentasi Pelanggan dengan K-Means Clustering</p>
        <p style='margin: 0.5rem 0; font-size: 0.85rem; color: #999;'>RFM Analysis • Data 1 Bulan Terakhir</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()














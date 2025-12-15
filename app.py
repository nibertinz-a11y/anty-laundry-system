"""
ANTY LAUNDRY - SISTEM SEGMENTASI PELANGGAN
Menggunakan K-Means Clustering & RFM Analysis
VERSI SKRIPSI - Data 1 Bulan Terakhir

Author: Anty Laundry Team
Version: 2.1 (IMPROVED - Feedback Owner)
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
# Ganti CSS di bagian st.markdown() dengan kode ini:

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
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Text input and text area */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        color: white;
        padding: 1rem;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        font-weight: 600;
        color: #e5e7eb;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.12);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Custom segment info box */
    .segment-info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #8b5cf6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    
    .segment-info-box h4 {
        color: #a78bfa;
        margin-top: 0;
        font-weight: 700;
    }
    
    .segment-info-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .segment-info-box li {
        margin: 0.3rem 0;
        color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KELAS UTAMA: ANTY LAUNDRY K-MEANS
# ============================================================

class AntyLaundryKMeans:
    """
    Class untuk melakukan K-Means Clustering pada data pelanggan laundry
    """
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = None
    
    def load_and_clean_data(self, df, months_back=1):
        """Load dan bersihkan data dari Excel"""
        
        st.info("📂 Memproses data...")
        
        # Cek kolom yang ada
        st.write("**Kolom yang terdeteksi:**", list(df.columns))
        
        # Mapping kolom (cari otomatis)
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            if 'tanggal' in col_lower or 'date' in col_lower or 'tgl' in col_lower:
                column_mapping[col] = 'Tanggal'
            elif 'konsumen' in col_lower or 'customer' in col_lower or 'nama' in col_lower or 'pelanggan' in col_lower:
                column_mapping[col] = 'Konsumen'
            elif 'total' in col_lower and ('harga' in col_lower or 'price' in col_lower or 'bayar' in col_lower):
                column_mapping[col] = 'Total_Harga'
            elif 'invoice' in col_lower or 'nota' in col_lower or 'no' in col_lower:
                column_mapping[col] = 'No_Invoice'
        
        st.write("**Mapping Kolom:**", column_mapping)
        
        # Rename kolom
        df = df.rename(columns=column_mapping)
        
        # Validasi kolom wajib
        required_cols = ['Tanggal', 'Konsumen', 'Total_Harga']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom berikut tidak ditemukan: {missing_cols}")
            st.info("💡 Pastikan Excel memiliki kolom: Tanggal, Konsumen/Nama/Customer, Total Harga")
            return None
        
        # Parse tanggal
        try:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            df = df.dropna(subset=['Tanggal'])
        except Exception as e:
            st.error(f"❌ Error parsing tanggal: {str(e)}")
            return None
        
        # 🔴 INI YANG PENTING: FILTER PERIODE (1 BULAN TERAKHIR)
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
        
        # Hitung RFM
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
        
        # Convert to numeric
        rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce')
        rfm = rfm.dropna()
        
        st.success(f"✅ RFM dihitung untuk {len(rfm)} pelanggan")
        
        # Debug info
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
        """Label setiap cluster berdasarkan karakteristik RFM"""
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Konsumen': 'count'
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        
        # Tampilkan karakteristik tiap cluster
        st.info("📊 Karakteristik Cluster:")
        for idx, row in cluster_summary.iterrows():
            st.write(f"**Cluster {row['Cluster']}**: Recency={row['Avg_Recency']:.0f} hari, Freq={row['Avg_Frequency']:.1f}x, Money=Rp{row['Avg_Monetary']:,.0f}, Count={row['Count']}")
        
        labels = {}
        for idx, row in cluster_summary.iterrows():
            cluster_id = row['Cluster']
            
            # Logika labeling berdasarkan karakteristik RFM
            if row['Avg_Recency'] < 15 and row['Avg_Frequency'] >= 3 and row['Avg_Monetary'] > rfm_df['Monetary'].quantile(0.75):
                labels[cluster_id] = {
                    'name': 'VIP Champions',
                    'icon': '🏆',
                    'discount': 15,
                    'priority': 1,
                    'description': 'Pelanggan paling berharga dengan transaksi sangat sering, belanja tinggi, dan baru bertransaksi',
                    'characteristics': [
                        'Recency < 15 hari (baru transaksi)',
                        'Frequency ≥ 3 transaksi',
                        'Monetary > 75% pelanggan lain (top 25%)',
                        'Pelanggan paling loyal dan menguntungkan'
                    ]
                }
            elif row['Avg_Recency'] < 20 and row['Avg_Frequency'] >= 2 and row['Avg_Monetary'] > rfm_df['Monetary'].quantile(0.5):
                labels[cluster_id] = {
                    'name': 'High Value Loyal',
                    'icon': '💎',
                    'discount': 10,
                    'priority': 2,
                    'description': 'Pelanggan loyal dengan nilai transaksi tinggi dan cukup aktif',
                    'characteristics': [
                        'Recency < 20 hari',
                        'Frequency ≥ 2 transaksi',
                        'Monetary > 50% pelanggan lain (median)',
                        'Berpotensi menjadi VIP Champions'
                    ]
                }
            elif row['Avg_Frequency'] >= 2 or row['Avg_Monetary'] > rfm_df['Monetary'].quantile(0.3):
                labels[cluster_id] = {
                    'name': 'Regular Loyal',
                    'icon': '💚',
                    'discount': 5,
                    'priority': 3,
                    'description': 'Pelanggan reguler yang cukup sering transaksi atau belanja cukup tinggi',
                    'characteristics': [
                        'Frequency ≥ 2 transaksi ATAU',
                        'Monetary > 30% pelanggan lain',
                        'Pelanggan stabil yang perlu dijaga',
                        'Berpotensi naik ke High Value'
                    ]
                }
            elif row['Avg_Recency'] > 25:
                labels[cluster_id] = {
                    'name': 'Sleeping Customers',
                    'icon': '😴',
                    'discount': 10,
                    'priority': 5,
                    'description': 'Pelanggan yang sudah lama tidak transaksi dan perlu reaktivasi',
                    'characteristics': [
                        'Recency > 25 hari (lama tidak transaksi)',
                        'Perlu kampanye reaktivasi',
                        'Diskon tinggi untuk menarik kembali',
                        'Risiko churn tinggi'
                    ]
                }
            else:
                labels[cluster_id] = {
                    'name': 'At Risk',
                    'icon': '⚠️',
                    'discount': 7,
                    'priority': 4,
                    'description': 'Pelanggan dengan aktivitas menurun yang berisiko hilang',
                    'characteristics': [
                        'Frequency atau Monetary menurun',
                        'Tidak masuk kategori VIP/High Value',
                        'Perlu perhatian khusus',
                        'Diskon untuk mendorong transaksi'
                    ]
                }
        
        rfm_df['Segment'] = rfm_df['Cluster'].map(lambda x: labels[x]['name'])
        rfm_df['Icon'] = rfm_df['Cluster'].map(lambda x: labels[x]['icon'])
        rfm_df['Discount'] = rfm_df['Cluster'].map(lambda x: labels[x]['discount'])
        rfm_df['Priority'] = rfm_df['Cluster'].map(lambda x: labels[x]['priority'])
        
        return rfm_df, labels
    
    def get_top_10_customers(self, rfm_df):
        """Pilih TOP 10 pelanggan untuk diskon"""
        # Prioritas 1: VIP Champions dan High Value Loyal
        top_segments = rfm_df[rfm_df['Segment'].isin(['VIP Champions', 'High Value Loyal'])]
        top_10 = top_segments.nlargest(10, 'Monetary')
        
        # Jika kurang dari 10, ambil dari Regular Loyal
        if len(top_10) < 10:
            remaining = 10 - len(top_10)
            regular = rfm_df[rfm_df['Segment'] == 'Regular Loyal'].nlargest(remaining, 'Monetary')
            top_10 = pd.concat([top_10, regular])
        
        # Jika masih kurang, ambil dari semua cluster berdasarkan Monetary
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
# FUNGSI UTILITY
# ============================================================

def export_to_excel(rfm_df, top_10, cluster_summary):
    """Export hasil ke Excel dengan multiple sheets"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Full RFM
        rfm_export = rfm_df[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount', 'Cluster']].copy()
        rfm_export.to_excel(writer, sheet_name='Full RFM', index=False)
        
        # Sheet 2: TOP 10
        top_10_export = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        top_10_export.to_excel(writer, sheet_name='TOP 10', index=False)
        
        # Sheet 3: Cluster Summary
        cluster_summary.to_excel(writer, sheet_name='Ringkasan Segmen', index=False)
    
    output.seek(0)
    return output.getvalue()

def generate_whatsapp_message(top_10):
    """Generate pesan WhatsApp untuk TOP 10"""
    current_month = datetime.now().strftime('%B %Y')
    next_month = (datetime.now() + timedelta(days=30)).strftime('%B %Y')
    
    message = f"""🧺 *ANTY LAUNDRY - DISKON SPESIAL*

📅 Data Bulan: {current_month}
🎁 Berlaku Bulan: {next_month}

Selamat kepada 10 pelanggan terpilih yang mendapat diskon bulan depan! 🎉

"""
    
    for idx, row in top_10.iterrows():
        rank = top_10.index.get_loc(idx) + 1
        message += f"{rank}. *{row['Konsumen']}*\n"
        message += f"   • Segmen: {row['Segment']}\n"
        message += f"   • Total Belanja: Rp {row['Monetary']:,.0f}\n"
        message += f"   • Diskon: {row['Discount']}%\n\n"
    
    message += """
✨ *Terima kasih atas kesetiaan Anda!*
Diskon otomatis berlaku untuk transaksi bulan depan.

---
_Anty Laundry - Laundry Terpercaya Anda_
"""
    
    return message

def create_whatsapp_link(message):
    """Buat link WhatsApp"""
    encoded_message = urllib.parse.quote(message)
    return f"https://wa.me/?text={encoded_message}"


# ============================================================
# HALAMAN UTAMA
# ============================================================

def main():
    """Fungsi utama aplikasi"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <img src="https://i.ibb.co.com/6gQKVYD/LOGO.png" class="logo-img" alt="Anty Laundry">
            <div class="header-text">
                <h1>🧺 ANTY LAUNDRY</h1>
                <h3>Sistem Segmentasi Pelanggan</h3>
                <p>K-Means Clustering • RFM Analysis • Data 1 Bulan Terakhir</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <img src="https://i.ibb.co.com/6gQKVYD/LOGO.png" alt="Logo">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Pengaturan Analisis")
        
        months_back = st.selectbox(
            "Periode Data (Bulan Terakhir)",
            options=[1, 2, 3],
            index=0,
            help="Pilih berapa bulan ke belakang untuk analisis"
        )
        
        st.markdown("---")
        
        st.markdown("### 📖 Panduan Singkat")
        st.markdown("""
        1. Upload file Excel data transaksi
        2. Pastikan ada kolom: **Tanggal**, **Konsumen**, **Total Harga**
        3. Klik tombol **Jalankan Analisis**
        4. Lihat hasil TOP 10 pelanggan dapat diskon
        5. Download atau bagikan ke WhatsApp
        """)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Informasi")
        st.info("""
        **TOP 10 Dipilih Dari:**
        - Segmen VIP Champions
        - Segmen High Value Loyal
        - Berdasarkan total belanja tertinggi
        
        **Jika pelanggan tidak muncul:**
        - Tidak ada transaksi dalam periode dipilih
        - Tidak masuk TOP 10 berdasarkan total belanja
        - Tidak termasuk segmen VIP/High Value
        """)
    
    # Main content
    st.markdown("## 📤 Upload Data Transaksi")
    
    uploaded_file = st.file_uploader(
        "Pilih file Excel (.xlsx, .xls)",
        type=['xlsx', 'xls'],
        help="Upload file Excel dengan kolom: Tanggal, Konsumen/Nama, Total Harga"
    )
    
    if uploaded_file:
        try:
            # Load data
            with st.spinner("📂 Memuat file..."):
                df_raw = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File berhasil dimuat: **{uploaded_file.name}**")
            st.info(f"📊 Total baris data: {len(df_raw)}")
            
            # Preview data
            with st.expander("👀 Preview Data (10 baris pertama)"):
                st.dataframe(df_raw.head(10))
            
            # Tombol proses
            if st.button("🚀 Jalankan Analisis K-Means", type="primary", use_container_width=True):
                
                with st.spinner("⏳ Sedang memproses data..."):
                    
                    engine = AntyLaundryKMeans()
                    
                    # Step 1: Clean & Filter
                    st.markdown("### 📊 Step 1: Membersihkan & Filter Data")
                    df_clean = engine.load_and_clean_data(df_raw, months_back=months_back)
                    
                    if df_clean is None:
                        st.error("❌ Gagal memproses data!")
                        st.stop()
                    
                    # Step 2: RFM
                    st.markdown("### 🔢 Step 2: Menghitung RFM")
                    rfm = engine.calculate_rfm(df_clean)
                    
                    # Step 3: Normalize
                    st.markdown("### 🔢 Step 3: Normalisasi Data")
                    rfm = engine.normalize_data(rfm)
                    st.success("✅ Data berhasil dinormalisasi")
                    
                    # Step 4: Clustering
                    st.markdown("### 🤖 Step 4: K-Means Clustering")
                    rfm = engine.run_kmeans(rfm)
                    
                    # Step 5: Label
                    st.markdown("### 🏷️ Step 5: Labeling Cluster")
                    rfm, cluster_labels = engine.label_clusters(rfm)
                    st.success("✅ Cluster berhasil dilabeli")
                    
                    # Step 6: Top 10
                    st.markdown("### 🏆 Step 6: Memilih TOP 10")
                    top_10 = engine.get_top_10_customers(rfm)
                    st.success(f"✅ {len(top_10)} pelanggan terpilih")
                    
                    # Save to session
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
    
    # Display Results
    if 'rfm_result' in st.session_state:
        
        rfm = st.session_state['rfm_result']
        top_10 = st.session_state['top_10']
        cluster_labels = st.session_state['cluster_labels']
        df_clean = st.session_state['df_clean']
        
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis")
        
        # Metrics
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
        
        # ========================================
        # BAGIAN INI DIPINDAH KE ATAS (FEEDBACK #2)
        # ========================================
        
        # TOP 10 - DIPINDAH KE ATAS
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
        
        # Export & WhatsApp - DIPINDAH KE ATAS (FEEDBACK #2)
        st.markdown("### 📥 Download & Bagikan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
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
        
        with col2:
            csv = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_csv(index=False)
            
            st.download_button(
                label="📄 Download TOP 10 (CSV)",
                data=csv,
                file_name=f"TOP_10_KMeans_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Initialize WhatsApp message in session state
            if 'wa_message' not in st.session_state:
                st.session_state['wa_message'] = generate_whatsapp_message(top_10)
            
            wa_link = create_whatsapp_link(st.session_state['wa_message'])
            
            st.link_button(
                label="💬 Bagikan ke WhatsApp",
                url=wa_link,
                use_container_width=True
            )
        
        # FEEDBACK #3: Preview & Edit pesan WhatsApp
        with st.expander("👁️ Preview & Edit Pesan WhatsApp", expanded=False):
            st.markdown("**Edit pesan di bawah ini, lalu klik tombol WhatsApp di atas untuk membagikan:**")
            
            # Editable text area
            edited_message = st.text_area(
                "Pesan WhatsApp:",
                value=st.session_state['wa_message'],
                height=400,
                help="Edit pesan ini sesuai kebutuhan, lalu klik tombol 'Bagikan ke WhatsApp' di atas",
                key='wa_message_editor'
            )
            
            # Update session state when message is edited
            if edited_message != st.session_state['wa_message']:
                st.session_state['wa_message'] = edited_message
                st.info("✏️ Pesan telah diubah. Klik tombol 'Bagikan ke WhatsApp' di atas untuk mengirim pesan baru.")
            
            # Reset button
            col_reset1, col_reset2 = st.columns([1, 3])
            with col_reset1:
                if st.button("🔄 Reset ke Template", use_container_width=True):
                    st.session_state['wa_message'] = generate_whatsapp_message(top_10)
                    st.rerun()
        
        st.markdown("---")
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_cluster_distribution_chart(rfm), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_rfm_3d_scatter(rfm), use_container_width=True)
        
        st.markdown("---")
        
        # FEEDBACK #4: Detail per Cluster dengan PENJELASAN LENGKAP
        st.markdown("## 🎯 Detail Segmentasi Pelanggan")
        
        st.info("""
        💡 **Penjelasan Sistem Segmentasi:**
        
        Sistem ini membagi pelanggan ke dalam 5 segmen berdasarkan 3 metrik utama (RFM):
        - **R (Recency)**: Berapa hari sejak transaksi terakhir (semakin kecil = semakin baik)
        - **F (Frequency)**: Jumlah transaksi dalam periode analisis (semakin banyak = semakin baik)
        - **M (Monetary)**: Total belanja dalam periode analisis (semakin tinggi = semakin baik)
        
        Beberapa cluster bisa memiliki nama segmen yang sama (misal: VIP Champions #1 dan VIP Champions #2) 
        karena algoritma K-Means mengelompokkan berdasarkan pola RFM yang mirip, namun cluster yang berbeda 
        tetap memiliki karakteristik RFM yang sedikit berbeda.
        """)
        
        for cluster_id in sorted(rfm['Cluster'].unique()):
            cluster_data = rfm[rfm['Cluster'] == cluster_id]
            label_info = cluster_labels[cluster_id]
            
            with st.expander(f"{label_info['icon']} **{label_info['name']}** (Cluster #{cluster_id}) - {len(cluster_data)} pelanggan"):
                
                # FEEDBACK #4: Tambahkan deskripsi detail
                st.markdown(f"""
                <div class="segment-info-box">
                    <h4>📝 Deskripsi Segmen</h4>
                    <p>{label_info['description']}</p>
                    <h4>🔍 Karakteristik RFM:</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                for char in label_info['characteristics']:
                    st.markdown(f"<li>{char}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.0f} hari")
                
                with col2:
                    st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}x")
                
                with col3:
                    st.metric("Avg Monetary", f"Rp {cluster_data['Monetary'].mean():,.0f}")
                
                with col4:
                    st.metric("Diskon", f"{label_info['discount']}%")
                
                st.markdown("**Daftar Pelanggan di Segmen Ini:**")
                st.dataframe(
                    cluster_data[['Konsumen', 'Recency', 'Frequency', 'Monetary']].sort_values('Monetary', ascending=False),
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Next Steps
        st.markdown("## 📋 Langkah Selanjutnya")
        
        st.success("""
        ✅ **Simpan laporan** dan berikan ke karyawan  
        ✅ **Input diskon** untuk 10 pelanggan di kasir laundry1010dry  
        ✅ **Berlaku mulai** bulan depan  
        ✅ **Upload data baru** setiap bulan untuk update otomatis
        """)
        
        # FEEDBACK #1: Penjelasan kenapa pelanggan tertentu tidak muncul
        st.markdown("---")
        st.markdown("### ❓ FAQ - Kenapa Pelanggan Tertentu Tidak Muncul?")
        
        st.warning("""
        **Jika pelanggan seperti Adi atau Filke tidak muncul di TOP 10, kemungkinan penyebabnya:**
        
        1. **Tidak ada transaksi dalam periode yang dipilih**
           - Cek apakah mereka ada transaksi dalam 1 bulan terakhir (atau periode yang Anda pilih)
           - Sistem hanya menganalisis data sesuai periode yang dipilih di sidebar
        
        2. **Tidak masuk TOP 10 berdasarkan total belanja**
           - TOP 10 dipilih dari segmen VIP Champions & High Value Loyal
           - Urutkan berdasarkan total belanja (Monetary) tertinggi
           - Mungkin ada pelanggan lain dengan total belanja lebih tinggi
        
        3. **Tidak termasuk segmen VIP Champions atau High Value Loyal**
           - Sistem prioritas memilih dari 2 segmen teratas saja
           - Cek di bagian "Detail Segmentasi" untuk melihat semua pelanggan per segmen
        
        **Solusi:**
        - Gunakan periode analisis lebih panjang (2-3 bulan) di sidebar
        - Cek data Excel apakah nama pelanggan sudah benar
        - Lihat semua segmentasi di bawah untuk menemukan pelanggan tersebut
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p style='margin: 0.5rem 0; font-size: 1rem;'><strong>© 2025 Anty Laundry</strong></p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Sistem Segmentasi Pelanggan dengan K-Means Clustering</p>
        <p style='margin: 0.5rem 0; font-size: 0.85rem; color: #999;'>RFM Analysis • Data 1 Bulan Terakhir • v2.1 Improved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

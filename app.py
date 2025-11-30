"""
ANTY LAUNDRY - SISTEM SEGMENTASI PELANGGAN
Menggunakan K-Means Clustering & RFM Analysis
untuk Otomatisasi Pemberian Diskon

Author: Anty Laundry Team
Version: 1.3 (WITH GIF)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import urllib.parse

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Anty Laundry - Segmentasi Pelanggan",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
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
        # Coba exact match dulu (dengan spasi)
        for keyword in keywords:
            for col in df.columns:
                if str(col).lower().strip() == keyword.lower():
                    return col
        
        # Kalau tidak ada, coba partial match (tanpa spasi)
        for keyword in keywords:
            keyword_clean = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                col_clean = str(col).lower().strip().replace(' ', '').replace('_', '')
                if keyword_clean == col_clean:
                    return col
        
        # Terakhir, coba contains
        for keyword in keywords:
            keyword_clean = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                col_clean = str(col).lower().strip().replace(' ', '').replace('_', '')
                if keyword_clean in col_clean:
                    return col
        
        return None
    
    def load_and_clean_data(self, df):
        """Membersihkan dan memvalidasi data"""
        df = df.copy()
        
        st.info("🔍 Mencari kolom yang dibutuhkan...")
        
        # Auto-detect kolom
        col_mapping = {}
        
        # Cari kolom Tanggal Ambil (termasuk yang terpotong)
        tanggal_col = self.find_column(df, ['tanggal ar', 'tanggal ambil', 'tgl ambil', 'tanggalambil'])
        if tanggal_col:
            col_mapping[tanggal_col] = 'Tanggal'
            st.success(f"✅ Tanggal: **{tanggal_col}** → Tanggal")
        else:
            st.error("❌ Kolom 'Tanggal Ambil' tidak ditemukan!")
        
        # Cari kolom Konsumen
        konsumen_col = self.find_column(df, ['konsumer', 'konsumen', 'customer', 'pelanggan'])
        if konsumen_col:
            col_mapping[konsumen_col] = 'Konsumen'
            st.success(f"✅ Konsumen: **{konsumen_col}** → Konsumen")
        
        # Cari kolom Total Harga (termasuk yang terpotong)
        harga_col = self.find_column(df, ['total harg', 'total harga', 'totalharga'])
        if harga_col:
            col_mapping[harga_col] = 'Total_Harga'
            st.success(f"✅ Total Harga: **{harga_col}** → Total_Harga")
        
        # Cari kolom Invoice/Nota
        invoice_col = self.find_column(df, ['nota', 'invoice', 'no nota', 'nonota', 'no.nota'])
        if invoice_col:
            col_mapping[invoice_col] = 'No_Invoice'
            st.success(f"✅ No Invoice: **{invoice_col}** → No_Invoice")
        
        # Cari kolom Status Order
        status_col = self.find_column(df, ['status order', 'statusorder', 'status'])
        if status_col:
            col_mapping[status_col] = 'Status_Order'
            st.success(f"✅ Status Order: **{status_col}** → Status_Order")
        
        # Tampilkan mapping yang ditemukan
        st.info(f"🔄 Mapping kolom (Asli → Baru): {col_mapping}")
        
        # Validasi sebelum rename
        if tanggal_col not in col_mapping:
            st.error("❌ Kolom Tanggal tidak ditemukan!")
            return None
        if konsumen_col not in col_mapping:
            st.error("❌ Kolom Konsumen tidak ditemukan!")
            return None
        if harga_col not in col_mapping:
            st.error("❌ Kolom Total Harga tidak ditemukan!")
            return None
        
        # Rename kolom
        df = df.rename(columns=col_mapping)
        
        # Debug: cek kolom setelah rename
        st.info(f"📋 Kolom setelah rename: {df.columns.tolist()}")
        
        st.success(f"✅ Semua kolom wajib berhasil ditemukan dan di-rename!")
        
        # Filter data
        original_len = len(df)
        
        # Filter Status Batal
        if 'Status_Order' in df.columns:
            before = len(df)
            df = df[df['Status_Order'].astype(str).str.lower() != 'batal']
            after = len(df)
            if before > after:
                st.warning(f"⚠️ {before - after} transaksi batal dihapus")
        
        # Filter Total Harga > 0
        if 'Total_Harga' in df.columns:
            df['Total_Harga'] = pd.to_numeric(df['Total_Harga'], errors='coerce')
            df = df[df['Total_Harga'] > 0]
            st.success(f"✅ {len(df)} transaksi valid dari {original_len} total transaksi")
        else:
            st.error(f"❌ Kolom 'Total_Harga' tidak ada setelah rename!")
            st.error(f"Kolom yang ada: {df.columns.tolist()}")
            return None
        
        # Convert tanggal
        try:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            df = df.dropna(subset=['Tanggal'])
            st.success(f"✅ Tanggal berhasil diparse: {df['Tanggal'].min().strftime('%d/%m/%Y')} - {df['Tanggal'].max().strftime('%d/%m/%Y')}")
        except Exception as e:
            st.error(f"❌ Error parsing tanggal: {str(e)}")
            return None
        
        # Bersihkan nama konsumen
        df['Konsumen'] = df['Konsumen'].astype(str).str.strip()
        df = df.dropna(subset=['Total_Harga', 'Konsumen'])
        
        return df
    
    def calculate_rfm(self, df, reference_date=None):
        """Menghitung nilai RFM"""
        
        if reference_date is None:
            reference_date = df['Tanggal'].max()
        
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
        return rfm_df
    
    def label_clusters(self, rfm_df):
        """Label setiap cluster"""
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Konsumen': 'count'
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Count']
        
        labels = {}
        for idx, row in cluster_summary.iterrows():
            cluster_id = row['Cluster']
            
            if row['Avg_Recency'] < 30 and row['Avg_Frequency'] > 50 and row['Avg_Monetary'] > 2000000:
                labels[cluster_id] = {
                    'name': 'VIP Champions',
                    'icon': '🏆',
                    'discount': 15,
                    'priority': 1
                }
            elif row['Avg_Recency'] < 50 and row['Avg_Frequency'] > 30 and row['Avg_Monetary'] > 1000000:
                labels[cluster_id] = {
                    'name': 'High Value Loyal',
                    'icon': '💎',
                    'discount': 10,
                    'priority': 2
                }
            elif row['Avg_Frequency'] > 5 and row['Avg_Monetary'] > 200000:
                labels[cluster_id] = {
                    'name': 'Regular Loyal',
                    'icon': '💚',
                    'discount': 5,
                    'priority': 3
                }
            elif row['Avg_Recency'] > 200:
                labels[cluster_id] = {
                    'name': 'Sleeping Customers',
                    'icon': '😴',
                    'discount': 10,
                    'priority': 5
                }
            else:
                labels[cluster_id] = {
                    'name': 'At Risk',
                    'icon': '⚠️',
                    'discount': 7,
                    'priority': 4
                }
        
        rfm_df['Segment'] = rfm_df['Cluster'].map(lambda x: labels[x]['name'])
        rfm_df['Icon'] = rfm_df['Cluster'].map(lambda x: labels[x]['icon'])
        rfm_df['Discount'] = rfm_df['Cluster'].map(lambda x: labels[x]['discount'])
        rfm_df['Priority'] = rfm_df['Cluster'].map(lambda x: labels[x]['priority'])
        
        return rfm_df, labels
    
    def get_top_10_customers(self, rfm_df):
        """Pilih TOP 10 pelanggan untuk diskon"""
        top_segments = rfm_df[rfm_df['Segment'].isin(['VIP Champions', 'High Value Loyal'])]
        top_10 = top_segments.nlargest(10, 'Monetary')
        
        if len(top_10) < 10:
            remaining = 10 - len(top_10)
            regular = rfm_df[rfm_df['Segment'] == 'Regular Loyal'].nlargest(remaining, 'Monetary')
            top_10 = pd.concat([top_10, regular])
        
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

def generate_whatsapp_message(top_10):
    """Generate pesan WhatsApp untuk TOP 10 pelanggan"""
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

━━━━━━━━━━━━━━━━
🧺 ANTY LAUNDRY
📍 Alamat: [Isi alamat Anda]
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
        # Sheet 1: Top 10
        top_10_export = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        top_10_export.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency (x)', 'Total Belanja (Rp)', 'Diskon (%)']
        top_10_export.to_excel(writer, sheet_name='Top 10 Pelanggan', index=False)
        
        # Sheet 2: Semua Pelanggan
        all_customers = rfm_df[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        all_customers.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency (x)', 'Total Belanja (Rp)', 'Diskon (%)']
        all_customers.to_excel(writer, sheet_name='Semua Pelanggan', index=False)
        
        # Sheet 3: Ringkasan
        cluster_summary.to_excel(writer, sheet_name='Ringkasan Cluster', index=False)
    
    output.seek(0)
    return output


# ============================================================
# MAIN APP
# ============================================================

def main():
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧺 ANTY LAUNDRY</h1>
        <h3>Sistem Segmentasi Pelanggan Berbasis K-Means & RFM Analysis</h3>
        <p>Otomatisasi Pemberian Diskon untuk Pelanggan Loyal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar dengan GIF Lucu
    with st.sidebar:
        # GIF LUCU LAUNDRY - INI YANG BARU!
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <img src='https://media.giphy.com/media/xT9IgDEI1iZyb2wqo8/giphy.gif' 
                 style='width: 100%; max-width: 250px; border-radius: 10px;'
                 alt='Laundry Animation'>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; color: #667eea;'>🧺 ANTY LAUNDRY</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 📋 Panduan Penggunaan")
        st.markdown("""
        **Step-by-step:**
        1. 📤 Upload file Excel dari kasir
        2. 🔄 Klik tombol "Jalankan Analisis"
        3. 📊 Lihat hasil segmentasi pelanggan
        4. 🏆 Cek TOP 10 pelanggan
        5. 📥 Download laporan Excel
        """)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Informasi Sistem")
        st.info("""
        **Metode:** K-Means Clustering (k=5)
        
        **RFM Analysis:**
        - 📅 Recency: Terakhir transaksi
        - 🔄 Frequency: Jumlah transaksi
        - 💰 Monetary: Total belanja
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Segmen Pelanggan")
        st.markdown("""
        🏆 **VIP Champions** → 15% diskon  
        💎 **High Value Loyal** → 10% diskon  
        💚 **Regular Loyal** → 5% diskon  
        ⚠️ **At Risk** → 7% diskon  
        😴 **Sleeping** → 10% diskon
        """)
        
        st.markdown("---")
        st.caption("© 2025 Anty Laundry v1.3")
    
    # Main Content
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
            
            # Preview data
            with st.expander("👀 Preview Data (10 baris pertama)"):
                st.dataframe(df_raw.head(10))
            
            # Tombol proses
            if st.button("🚀 Jalankan Analisis K-Means", type="primary", use_container_width=True):
                
                with st.spinner("⏳ Sedang memproses data..."):
                    
                    engine = AntyLaundryKMeans()
                    
                    # Step 1: Clean
                    st.markdown("### 📊 Step 1: Membersihkan Data")
                    df_clean = engine.load_and_clean_data(df_raw)
                    
                    if df_clean is None:
                        st.stop()
                    
                    # Step 2: RFM
                    st.markdown("### 🔢 Step 2: Menghitung RFM")
                    rfm = engine.calculate_rfm(df_clean)
                    
                    # Step 3: Normalize
                    st.markdown("### 📏 Step 3: Normalisasi Data")
                    rfm = engine.normalize_data(rfm)
                    st.success("✅ Data berhasil dinormalisasi")
                    
                    # Step 4: Clustering
                    st.markdown("### 🤖 Step 4: K-Means Clustering")
                    rfm = engine.run_kmeans(rfm)
                    st.success("✅ Clustering selesai (k=5)")
                    
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
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_cluster_distribution_chart(rfm), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_rfm_3d_scatter(rfm), use_container_width=True)
        
        st.markdown("---")
        
        # Detail per Cluster
        st.markdown("## 🎯 Detail Segmentasi")
        
        for cluster_id in sorted(rfm['Cluster'].unique()):
            cluster_data = rfm[rfm['Cluster'] == cluster_id]
            label_info = cluster_labels[cluster_id]
            
            with st.expander(f"{label_info['icon']} **{label_info['name']}** ({len(cluster_data)} pelanggan)"):
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.0f} hari")
                
                with col2:
                    st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}x")
                
                with col3:
                    st.metric("Avg Monetary", f"Rp {cluster_data['Monetary'].mean():,.0f}")
                
                with col4:
                    st.metric("Diskon", f"{label_info['discount']}%")
                
                st.dataframe(
                    cluster_data[['Konsumen', 'Recency', 'Frequency', 'Monetary']].sort_values('Monetary', ascending=False),
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # TOP 10
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
        
        # Export & WhatsApp
        st.markdown("## 📥 Download & Bagikan")
        
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
                file_name=f"Laporan_Segmentasi_Pelanggan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            csv = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_csv(index=False)
            
            st.download_button(
                label="📄 Download TOP 10 (CSV)",
                data=csv,
                file_name=f"TOP_10_Pelanggan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Generate WhatsApp message
            wa_message = generate_whatsapp_message(top_10)
            wa_link = create_whatsapp_link(wa_message)
            
            st.link_button(
                label="💬 Bagikan ke WhatsApp",
                url=wa_link,
                use_container_width=True
            )
        
        # Preview pesan WhatsApp
        with st.expander("👁️ Preview Pesan WhatsApp"):
            st.text_area(
                "Pesan yang akan dikirim:",
                wa_message,
                height=400,
                help="Copy pesan ini atau klik tombol 'Bagikan ke WhatsApp' di atas"
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2025 Anty Laundry - Sistem Segmentasi Pelanggan</p>
        <p>Menggunakan K-Means Clustering (k=5) dengan RFM Analysis</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
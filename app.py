import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO
import urllib.parse

st.set_page_config(
    page_title="Anty Laundry - Segmentasi Pelanggan",
    page_icon="🧺",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 100%;
    }

    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 14px;
        padding: 1rem 2rem;
        width: 100%;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.5);
    }

    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }

    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 900;
        color: #a78bfa;
    }

    h1, h2, h3 { color: #f3f4f6; }

    h2 {
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    p { color: #d1d5db; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ENGINE
# ============================================================

class AntyLaundryKMeans:

    def __init__(self):
        self.n_clusters = 6
        self.model = None
        self.scaler = StandardScaler()

    def find_column(self, df, keywords):
        for keyword in keywords:
            for col in df.columns:
                if str(col).lower().strip() == keyword.lower():
                    return col
        for keyword in keywords:
            kw = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                cc = str(col).lower().strip().replace(' ', '').replace('_', '')
                if kw in cc:
                    return col
        return None

    def load_and_clean_data(self, df):
        df = df.copy()

        tanggal_col  = self.find_column(df, ['tanggal ambil', 'tanggalambil', 'tgl ambil', 'tanggal order'])
        konsumen_col = self.find_column(df, ['konsumen', 'customer', 'pelanggan'])
        harga_col    = self.find_column(df, ['total harga', 'totalharga'])
        invoice_col  = self.find_column(df, ['no. nota', 'no nota', 'nota', 'invoice'])
        status_col   = self.find_column(df, ['status order', 'statusorder'])

        if not tanggal_col or not konsumen_col or not harga_col:
            st.error("❌ Kolom yang dibutuhkan tidak ditemukan (Tanggal, Konsumen, Total Harga)")
            return None

        col_map = {tanggal_col: 'Tanggal', konsumen_col: 'Konsumen', harga_col: 'Total_Harga'}
        if invoice_col: col_map[invoice_col] = 'No_Invoice'
        if status_col:  col_map[status_col]  = 'Status_Order'
        df = df.rename(columns=col_map)

        if 'Status_Order' in df.columns:
            df = df[~df['Status_Order'].astype(str).str.lower().str.contains('batal', na=False)]

        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df = df.dropna(subset=['Tanggal'])

        df['Total_Harga'] = pd.to_numeric(df['Total_Harga'], errors='coerce')
        df = df[df['Total_Harga'] > 0]
        df['Konsumen'] = df['Konsumen'].astype(str).str.strip()
        df = df.dropna(subset=['Total_Harga', 'Konsumen'])

        st.success(f"✅ {len(df)} transaksi valid dari {df['Konsumen'].nunique()} pelanggan unik")
        return df

    def calculate_rfm(self, df):
        ref = df['Tanggal'].max() + timedelta(days=1)
        st.info(f"📅 Tanggal referensi: {ref.strftime('%d/%m/%Y')}")

        if 'No_Invoice' in df.columns:
            rfm = df.groupby('Konsumen').agg(
                Recency=('Tanggal', lambda x: (ref - x.max()).days),
                Frequency=('No_Invoice', 'nunique'),
                Monetary=('Total_Harga', 'sum')
            ).reset_index()
        else:
            rfm = df.groupby('Konsumen').agg(
                Recency=('Tanggal', lambda x: (ref - x.max()).days),
                Frequency=('Tanggal', 'count'),
                Monetary=('Total_Harga', 'sum')
            ).reset_index()

        rfm = rfm.dropna()
        st.success(f"✅ RFM dihitung untuk {len(rfm)} pelanggan")
        return rfm

    def normalize_data(self, rfm):
        scaled = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']].values)
        rfm['Recency_scaled']   = scaled[:, 0]
        rfm['Frequency_scaled'] = scaled[:, 1]
        rfm['Monetary_scaled']  = scaled[:, 2]
        return rfm

    def run_kmeans(self, rfm):
        X = rfm[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']].values
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
        rfm['Cluster'] = self.model.fit_predict(X)
        st.success(f"✅ K-Means clustering selesai — k={self.n_clusters}, Inertia: {self.model.inertia_:.2f}")
        return rfm

    def label_clusters(self, rfm):
        summary = rfm.groupby('Cluster').agg(
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Count=('Konsumen', 'count')
        ).reset_index()

        max_r = rfm['Recency'].max()
        summary['Score'] = (
            (max_r - summary['Avg_Recency']) / max_r +
            summary['Avg_Frequency'] / rfm['Frequency'].max() +
            summary['Avg_Monetary'] / rfm['Monetary'].max()
        )
        summary = summary.sort_values('Score', ascending=False).reset_index(drop=True)

        segment_defs = [
            ('VIP Customer',         '🏆', 40, 'Pelanggan paling aktif dan loyal — frekuensi tertinggi, recency terendah, monetary sangat tinggi.'),
            ('Top Spender',          '💎', 35, 'Pelanggan dengan nilai belanja tertinggi secara keseluruhan.'),
            ('High Value Customer',  '💚', 20, 'Pelanggan aktif dengan nilai transaksi tinggi, berpotensi menjadi VIP.'),
            ('Loyal Customer',       '⭐', 15, 'Pelanggan rutin dengan nilai transaksi sedang-tinggi dan recency rendah.'),
            ('Regular Customer',     '👥', 10, 'Pelanggan aktif namun frekuensi dan nilai transaksi masih rendah.'),
            ('Inactive Customer',    '😴',  5, 'Pelanggan yang sudah lama tidak bertransaksi — perlu strategi re-engagement.'),
        ]

        labels = {}
        for idx, row in summary.iterrows():
            name, icon, disc, desc = segment_defs[idx]
            labels[int(row['Cluster'])] = {
                'name': name, 'icon': icon, 'discount': disc, 'description': desc
            }

        rfm['Segment']  = rfm['Cluster'].map(lambda x: labels[x]['name'])
        rfm['Icon']     = rfm['Cluster'].map(lambda x: labels[x]['icon'])
        rfm['Discount'] = rfm['Cluster'].map(lambda x: labels[x]['discount'])
        return rfm, labels

    def get_top_10(self, rfm):
        priority = ['VIP Customer', 'Top Spender', 'High Value Customer',
                    'Loyal Customer', 'Regular Customer', 'Inactive Customer']
        rfm['_rank'] = rfm['Segment'].map({s: i for i, s in enumerate(priority)})
        top = rfm.sort_values(['_rank', 'Monetary'], ascending=[True, False]).head(10)
        rfm.drop(columns=['_rank'], inplace=True)
        return top


# ============================================================
# HELPERS
# ============================================================

def generate_wa_message(top_10):
    msg = "🎉 *SELAMAT PELANGGAN SETIA ANTY LAUNDRY!* 🎉\n\n"
    msg += "Anda terpilih sebagai TOP 10 pelanggan terbaik bulan ini! 🏆\n\n"
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        msg += f"{i}. *{row['Konsumen']}*\n"
        msg += f"   {row['Icon']} Segmen: {row['Segment']}\n"
        msg += f"   🎁 Diskon: *{row['Discount']}%*\n"
        msg += f"   💰 Total Belanja: Rp {row['Monetary']:,.0f}\n\n"
    msg += "📅 Berlaku bulan depan untuk semua layanan\n"
    msg += "💳 Tunjukkan pesan ini saat transaksi\n\n"
    msg += "Terima kasih telah mempercayai ANTY LAUNDRY! 💙\n"
    msg += "━━━━━━━━━━━━━━━━━━\n🧺 ANTY LAUNDRY — Tomohon, Sulawesi Utara"
    return msg

def export_excel(rfm, top_10):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        top_10[['Konsumen','Segment','Recency','Frequency','Monetary','Discount']].rename(
            columns={'Konsumen':'Nama','Segment':'Segmen','Recency':'Recency (hari)',
                     'Frequency':'Frequency (x)','Monetary':'Total Belanja (Rp)','Discount':'Diskon (%)'}
        ).to_excel(writer, sheet_name='Top 10 Pelanggan', index=False)

        rfm[['Konsumen','Segment','Recency','Frequency','Monetary','Discount']].rename(
            columns={'Konsumen':'Nama','Segment':'Segmen','Recency':'Recency (hari)',
                     'Frequency':'Frequency (x)','Monetary':'Total Belanja (Rp)','Discount':'Diskon (%)'}
        ).to_excel(writer, sheet_name='Semua Pelanggan', index=False)
    output.seek(0)
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem;">🧺 ANTY LAUNDRY</h1>
        <p style="margin:0.3rem 0 0 0; opacity:0.9; font-size:1rem;">
            Sistem Segmentasi Pelanggan — K-Means Clustering (k=6)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📤 Upload Data Transaksi")
    uploaded = st.file_uploader(
        "Upload file Excel/CSV dari aplikasi kasir laundry1010dry",
        type=['xlsx', 'xls', 'csv'],
        help="File harus memiliki kolom: Tanggal Ambil, Konsumen, Total Harga"
    )

    if not uploaded:
        st.info("👆 Upload file Excel untuk memulai analisis segmentasi pelanggan.")
        return

    try:
        df_raw = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        st.success(f"✅ File dimuat: **{uploaded.name}** — {len(df_raw)} baris")

        with st.expander("👀 Preview data (10 baris pertama)"):
            st.dataframe(df_raw.head(10))

    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        return

    if not st.button("🚀 Jalankan Analisis K-Means", type="primary", use_container_width=True):
        return

    with st.spinner("⏳ Memproses data..."):
        engine = AntyLaundryKMeans()

        st.markdown("### Step 1 — Membersihkan Data")
        df_clean = engine.load_and_clean_data(df_raw)
        if df_clean is None:
            return

        st.markdown("### Step 2 — Menghitung RFM")
        rfm = engine.calculate_rfm(df_clean)

        st.markdown("### Step 3 — Normalisasi Data")
        rfm = engine.normalize_data(rfm)
        st.success("✅ Normalisasi selesai")

        st.markdown("### Step 4 — K-Means Clustering")
        rfm = engine.run_kmeans(rfm)

        st.markdown("### Step 5 — Labeling Segmen")
        rfm, labels = engine.label_clusters(rfm)
        st.success("✅ Segmen berhasil dilabeli")

        st.markdown("### Step 6 — Memilih TOP 10")
        top_10 = engine.get_top_10(rfm)
        st.success("✅ TOP 10 pelanggan terpilih")

        st.session_state.update({'rfm': rfm, 'top_10': top_10,
                                  'labels': labels, 'df_clean': df_clean})

    st.success("✅ Analisis selesai!")
    st.balloons()
    st.rerun()

    if 'rfm' not in st.session_state:
        return

    rfm    = st.session_state['rfm']
    top_10 = st.session_state['top_10']
    labels = st.session_state['labels']
    df_clean = st.session_state['df_clean']

    st.markdown("---")
    st.markdown("## 🏆 TOP 10 Pelanggan Loyal — Dapat Diskon!")

    top_disp = top_10.copy()
    top_disp.insert(0, 'No', ['🥇','🥈','🥉'] + [f'#{i}' for i in range(4,11)])
    top_disp['Monetary'] = top_disp['Monetary'].apply(lambda x: f"Rp {x:,.0f}")
    top_disp['Discount'] = top_disp['Discount'].apply(lambda x: f"🎁 {x}%")

    st.dataframe(
        top_disp[['No','Konsumen','Segment','Frequency','Monetary','Discount']]
        .rename(columns={'No':'#','Konsumen':'Nama','Segment':'Segmen',
                         'Frequency':'Trx','Monetary':'Total','Discount':'Diskon'}),
        use_container_width=True, hide_index=True, height=420
    )

    st.markdown("### 📤 Bagikan Sekarang")

    if 'wa_msg' not in st.session_state:
        st.session_state['wa_msg'] = generate_wa_message(top_10)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("💬 Kirim WhatsApp",
                       f"https://wa.me/?text={urllib.parse.quote(st.session_state['wa_msg'])}",
                       use_container_width=True)
    with c2:
        st.download_button("📊 Download Excel", export_excel(rfm, top_10),
                           f"Laporan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3:
        csv = top_10[['Konsumen','Segment','Recency','Frequency','Monetary','Discount']].to_csv(index=False)
        st.download_button("📄 Download CSV", csv,
                           f"TOP10_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv", use_container_width=True)

    with st.expander("✏️ Edit Pesan WhatsApp", expanded=False):
        edited = st.text_area("Pesan", value=st.session_state['wa_msg'], height=280)
        if st.button("🔄 Update Pesan"):
            st.session_state['wa_msg'] = edited
            st.success("✅ Pesan diupdate!")
            st.rerun()

    st.markdown("---")

    with st.expander("📊 Ringkasan & Statistik Lengkap", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Pelanggan", len(rfm))
        c2.metric("Total Transaksi", len(df_clean))
        c3.metric("Jumlah Segmen", 6)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(rfm['Segment'].value_counts().reset_index(),
                             values='count', names='Segment',
                             title='Distribusi Pelanggan per Segmen', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                                   color='Segment', hover_data=['Konsumen'],
                                   title='Visualisasi 3D RFM')
            st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Detail 6 Segmen Pelanggan")

        for cid in sorted(rfm['Cluster'].unique()):
            info  = labels[cid]
            data  = rfm[rfm['Cluster'] == cid]
            with st.expander(f"{info['icon']} **{info['name']}** ({len(data)} pelanggan) — Diskon {info['discount']}%"):
                st.info(info['description'])
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Recency",   f"{data['Recency'].mean():.0f} hari")
                c2.metric("Avg Frequency", f"{data['Frequency'].mean():.1f}x")
                c3.metric("Avg Monetary",  f"Rp {data['Monetary'].mean():,.0f}")
                st.dataframe(
                    data[['Konsumen','Recency','Frequency','Monetary']]
                    .sort_values('Monetary', ascending=False),
                    use_container_width=True
                )

    st.markdown("---")
    st.markdown("<p style='text-align:center;'>© 2025 Anty Laundry — Sistem Segmentasi Pelanggan K-Means k=6</p>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()

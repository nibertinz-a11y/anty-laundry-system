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


st.set_page_config(
    page_title="Anty Laundry - Segmentasi Pelanggan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .stApp {
        background-color: #f5f6fa;
    }

    .main .block-container {
        max-width: 1100px;
        padding: 2rem 2rem 4rem 2rem;
    }

    .page-header {
        background: #1e1e2e;
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }

    .page-header h1 {
        margin: 0 0 0.25rem 0;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: white;
    }

    .page-header p {
        margin: 0;
        font-size: 0.9rem;
        color: #a0a0b0;
    }

    h2 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e1e2e;
        margin-top: 2rem;
        margin-bottom: 0.75rem;
    }

    h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #2d2d44;
    }

    p {
        color: #444;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e5ef;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e1e2e;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .stButton > button {
        background: #1e1e2e;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background 0.2s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: #2d2d44;
    }

    .stDownloadButton > button {
        background: #1e1e2e;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background 0.2s ease;
        width: 100%;
    }

    .stDownloadButton > button:hover {
        background: #2d2d44;
    }

    .stLinkButton > a {
        background: #25d366;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none;
        display: inline-block;
        transition: background 0.2s ease;
        width: 100%;
        text-align: center;
        box-sizing: border-box;
    }

    .stLinkButton > a:hover {
        background: #1db954;
    }

    section[data-testid="stFileUploadDropzone"] {
        background: white;
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 2rem;
        transition: border-color 0.2s;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #1e1e2e;
    }

    .dataframe {
        border-radius: 10px;
        border: 1px solid #e5e5ef;
        font-size: 0.85rem;
    }

    .dataframe thead tr th {
        background: #1e1e2e;
        color: white;
        font-weight: 700;
        padding: 0.75rem 1rem;
    }

    .dataframe tbody tr td {
        padding: 0.6rem 1rem;
        color: #333;
    }

    .dataframe tbody tr:hover {
        background: #f0f0f8;
    }

    .streamlit-expanderHeader {
        background: white;
        border: 1px solid #e5e5ef;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #1e1e2e;
        padding: 1rem 1.25rem;
    }

    .stAlert {
        border-radius: 10px;
        font-size: 0.85rem;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e5ef;
        margin: 1.5rem 0;
    }

    textarea {
        font-size: 0.875rem !important;
        border-radius: 10px !important;
        border: 1px solid #ddd !important;
    }

    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #f0f0f5;
    }

    ::-webkit-scrollbar-thumb {
        background: #bbb;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


class LaundryKMeans:

    def __init__(self):
        self.n_clusters = 5
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
                c = str(col).lower().strip().replace(' ', '').replace('_', '')
                if kw == c:
                    return col

        for keyword in keywords:
            kw = keyword.lower().replace(' ', '').replace('_', '')
            for col in df.columns:
                c = str(col).lower().strip().replace(' ', '').replace('_', '')
                if kw in c:
                    return col

        return None

    def load_and_clean_data(self, df, months_back=1):
        df = df.copy()

        st.info("Mendeteksi kolom yang diperlukan...")

        col_mapping = {}

        tanggal_col = self.find_column(df, ['tanggal ar', 'tanggal ambil', 'tgl ambil', 'tanggalambil'])
        if tanggal_col:
            col_mapping[tanggal_col] = 'Tanggal'
            st.success(f"Kolom tanggal ditemukan: '{tanggal_col}'")
        else:
            st.error("Kolom tanggal ambil tidak ditemukan.")
            return None

        konsumen_col = self.find_column(df, ['konsumer', 'konsumen', 'customer', 'pelanggan'])
        if konsumen_col:
            col_mapping[konsumen_col] = 'Konsumen'
            st.success(f"Kolom konsumen ditemukan: '{konsumen_col}'")
        else:
            st.error("Kolom konsumen tidak ditemukan.")
            return None

        harga_col = self.find_column(df, ['total harg', 'total harga', 'totalharga'])
        if harga_col:
            col_mapping[harga_col] = 'Total_Harga'
            st.success(f"Kolom total harga ditemukan: '{harga_col}'")
        else:
            st.error("Kolom total harga tidak ditemukan.")
            return None

        invoice_col = self.find_column(df, ['nota', 'invoice', 'no nota', 'nonota', 'no.nota'])
        if invoice_col:
            col_mapping[invoice_col] = 'No_Invoice'

        status_col = self.find_column(df, ['status order', 'statusorder', 'status'])
        if status_col:
            col_mapping[status_col] = 'Status_Order'

        tanggal_order_col = self.find_column(df, ['tanggal order', 'tanggalorder', 'tgl order'])
        if tanggal_order_col:
            col_mapping[tanggal_order_col] = 'Tanggal_Order'

        df = df.rename(columns=col_mapping)

        if 'Status_Order' in df.columns:
            before = len(df)
            df = df[~df['Status_Order'].astype(str).str.lower().str.contains('batal', na=False)]
            removed = before - len(df)
            if removed > 0:
                st.warning(f"{removed} transaksi berstatus 'Batal' telah dihapus dari analisis.")

        try:
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            if 'Tanggal_Order' in df.columns:
                df['Tanggal_Order'] = pd.to_datetime(df['Tanggal_Order'], errors='coerce')
                df['Tanggal'] = df['Tanggal'].fillna(df['Tanggal_Order'])
            df = df.dropna(subset=['Tanggal'])
        except Exception as e:
            st.error(f"Gagal memparse kolom tanggal: {e}")
            return None

        max_date = df['Tanggal'].max()
        cutoff_date = max_date - timedelta(days=30 * months_back)

        st.info(f"Rentang analisis: {cutoff_date.strftime('%d/%m/%Y')} sampai {max_date.strftime('%d/%m/%Y')}")

        total_before = len(df)
        df = df[df['Tanggal'] > cutoff_date]
        st.success(f"Data yang digunakan: {len(df)} dari {total_before} transaksi (periode {months_back} bulan terakhir)")

        if 'Total_Harga' in df.columns:
            df['Total_Harga'] = pd.to_numeric(df['Total_Harga'], errors='coerce')
            df = df[df['Total_Harga'] > 0]
        else:
            st.error("Kolom Total_Harga tidak tersedia setelah proses rename.")
            return None

        df['Konsumen'] = df['Konsumen'].astype(str).str.strip()
        df = df.dropna(subset=['Total_Harga', 'Konsumen'])

        st.success(f"Total {len(df)} transaksi valid dari {df['Konsumen'].nunique()} pelanggan unik.")
        return df

    def calculate_rfm(self, df, reference_date=None):
        if reference_date is None:
            reference_date = df['Tanggal'].max()

        st.info(f"Tanggal referensi RFM: {reference_date.strftime('%d/%m/%Y')}")

        if 'No_Invoice' in df.columns:
            rfm = df.groupby('Konsumen').agg(
                Recency=('Tanggal', lambda x: (reference_date - x.max()).days),
                Frequency=('No_Invoice', 'nunique'),
                Monetary=('Total_Harga', 'sum')
            ).reset_index()
        else:
            rfm = df.groupby('Konsumen').agg(
                Recency=('Tanggal', lambda x: (reference_date - x.max()).days),
                Frequency=('Tanggal', 'count'),
                Monetary=('Total_Harga', 'sum')
            ).reset_index()

        rfm[['Recency', 'Frequency', 'Monetary']] = rfm[['Recency', 'Frequency', 'Monetary']].apply(
            pd.to_numeric, errors='coerce'
        )
        rfm = rfm.dropna()

        st.success(f"RFM berhasil dihitung untuk {len(rfm)} pelanggan.")
        st.info(
            f"Recency: {rfm['Recency'].min():.0f}-{rfm['Recency'].max():.0f} hari | "
            f"Frequency: {rfm['Frequency'].min():.0f}-{rfm['Frequency'].max():.0f} | "
            f"Monetary: Rp{rfm['Monetary'].min():,.0f} - Rp{rfm['Monetary'].max():,.0f}"
        )
        return rfm

    def normalize_data(self, rfm_df):
        features = ['Recency', 'Frequency', 'Monetary']
        scaled = self.scaler.fit_transform(rfm_df[features])
        rfm_df['Recency_scaled'] = scaled[:, 0]
        rfm_df['Frequency_scaled'] = scaled[:, 1]
        rfm_df['Monetary_scaled'] = scaled[:, 2]
        return rfm_df

    def run_kmeans(self, rfm_df):
        X = rfm_df[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']].values
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
        rfm_df['Cluster'] = self.model.fit_predict(X)
        st.success(f"K-Means selesai. WCSS (Inertia): {self.model.inertia_:.2f}")
        return rfm_df

    def label_clusters(self, rfm_df):
        summary = rfm_df.groupby('Cluster').agg(
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Count=('Konsumen', 'count')
        ).reset_index()

        max_recency = rfm_df['Recency'].max()
        summary['score'] = (
            (max_recency - summary['Avg_Recency']) / max_recency +
            summary['Avg_Frequency'] / rfm_df['Frequency'].max() +
            summary['Avg_Monetary'] / rfm_df['Monetary'].max()
        )

        summary = summary.sort_values('score', ascending=False).reset_index(drop=True)

        segment_definitions = [
            {
                'name': 'VIP Customer',
                'discount': 10,
                'priority': 1,
                'description': 'Pelanggan dengan frekuensi dan nilai transaksi tertinggi, serta yang paling aktif.'
            },
            {
                'name': 'Top Spender',
                'discount': 10,
                'priority': 2,
                'description': 'Pelanggan dengan total belanja besar meskipun frekuensi tidak setinggi VIP.'
            },
            {
                'name': 'High Value Customer',
                'discount': 10,
                'priority': 3,
                'description': 'Pelanggan dengan nilai cukup tinggi dan berpotensi naik ke tier atas.'
            },
            {
                'name': 'Pelanggan Reguler',
                'discount': 10,
                'priority': 4,
                'description': 'Pelanggan yang cukup rutin namun dengan nilai transaksi yang lebih kecil.'
            },
            {
                'name': 'Pelanggan Tidak Aktif',
                'discount': 10,
                'priority': 5,
                'description': 'Pelanggan yang sudah lama tidak bertransaksi. Perlu strategi re-engagement.'
            }
        ]

        labels = {}
        for rank, row in summary.iterrows():
            cid = row['Cluster']
            defn = segment_definitions[rank]
            labels[cid] = {
                **defn,
                'rfm_score': row['score'],
                'avg_recency': row['Avg_Recency'],
                'avg_frequency': row['Avg_Frequency'],
                'avg_monetary': row['Avg_Monetary'],
                'count': row['Count']
            }

        rfm_df['Segment'] = rfm_df['Cluster'].map(lambda x: labels[x]['name'])
        rfm_df['Discount'] = rfm_df['Cluster'].map(lambda x: labels[x]['discount'])
        rfm_df['Priority'] = rfm_df['Cluster'].map(lambda x: labels[x]['priority'])
        rfm_df['Description'] = rfm_df['Cluster'].map(lambda x: labels[x]['description'])

        return rfm_df, labels

    def get_top_10_customers(self, rfm_df):
        priority_segments = ['VIP Customer', 'Top Spender', 'High Value Customer']

        top = rfm_df[rfm_df['Segment'].isin(priority_segments)].nlargest(10, 'Monetary')

        if len(top) < 10:
            remaining = rfm_df[~rfm_df['Konsumen'].isin(top['Konsumen'])].nlargest(10 - len(top), 'Monetary')
            top = pd.concat([top, remaining])

        return top.head(10)


def chart_segment_distribution(rfm_df):
    counts = rfm_df['Segment'].value_counts().reset_index()
    counts.columns = ['Segment', 'Jumlah']
    fig = px.pie(
        counts,
        values='Jumlah',
        names='Segment',
        title='Distribusi Pelanggan per Segmen',
        hole=0.4,
        color_discrete_sequence=['#1e1e2e', '#3b3b5c', '#5c5c8a', '#8888c0', '#bbbbdd']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Plus Jakarta Sans'),
        legend=dict(orientation='h', y=-0.2)
    )
    return fig


def chart_rfm_scatter(rfm_df):
    fig = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        hover_data=['Konsumen'],
        title='Visualisasi 3D RFM',
        labels={
            'Recency': 'Recency (hari)',
            'Frequency': 'Frequency',
            'Monetary': 'Monetary (Rp)'
        },
        color_discrete_sequence=['#1e1e2e', '#3b3b5c', '#5c5c8a', '#8888c0', '#bbbbdd']
    )
    fig.update_layout(
        paper_bgcolor='white',
        font=dict(family='Plus Jakarta Sans')
    )
    return fig


def build_whatsapp_message(top_10):
    lines = [
        "*SELAMAT PELANGGAN SETIA ANTY LAUNDRY*\n",
        "Anda terpilih sebagai TOP 10 pelanggan terbaik bulan ini.\n",
        "Sebagai bentuk apresiasi, Anda mendapatkan diskon spesial:\n"
    ]

    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        lines.append(f"{i}. *{row['Konsumen']}*")
        lines.append(f"   Segmen: {row['Segment']}")
        lines.append(f"   Diskon: {row['Discount']}%")
        lines.append(f"   Total Belanja: Rp {row['Monetary']:,.0f}\n")

    lines += [
        "Berlaku bulan depan untuk semua layanan.",
        "Tunjukkan pesan ini saat melakukan transaksi.\n",
        "Terima kasih telah mempercayai Anty Laundry.\n",
        "---",
        "ANTY LAUNDRY",
        "Tomohon, Sulawesi Utara"
    ]

    return "\n".join(lines)


def make_whatsapp_link(message):
    return f"https://wa.me/?text={urllib.parse.quote(message)}"


def export_excel(rfm_df, top_10, cluster_summary):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        top_export = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        top_export.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency', 'Total Belanja (Rp)', 'Diskon (%)']
        top_export.to_excel(writer, sheet_name='Top 10 Pelanggan', index=False)

        all_export = rfm_df[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].copy()
        all_export.columns = ['Nama Pelanggan', 'Segmen', 'Recency (hari)', 'Frequency', 'Total Belanja (Rp)', 'Diskon (%)']
        all_export.to_excel(writer, sheet_name='Semua Pelanggan', index=False)

        cluster_summary.to_excel(writer, sheet_name='Ringkasan Cluster', index=False)

    output.seek(0)
    return output


def main():
    st.markdown("""
    <div class="page-header">
        <h1>Anty Laundry</h1>
        <p>Sistem Segmentasi Pelanggan &mdash; K-Means Clustering &amp; RFM Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Upload Data Transaksi")

    col_setting, col_upload = st.columns([1, 2])

    with col_setting:
        months_back = st.selectbox(
            "Periode analisis (bulan terakhir)",
            options=[1, 2, 3, 6, 12],
            index=0
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "Pilih file Excel atau CSV dari aplikasi kasir",
            type=['xlsx', 'xls', 'csv'],
            help="File harus memiliki kolom: Tanggal Ambil, Konsumen, Total Harga"
        )

    if uploaded_file is None:
        st.info("Upload file untuk memulai analisis.")
        return

    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    st.success(f"File dimuat: {uploaded_file.name} ({len(df_raw)} baris)")

    with st.expander("Preview data (10 baris pertama)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    if not st.button("Jalankan Analisis K-Means", type="primary", use_container_width=False):
        return

    with st.spinner("Memproses data..."):
        engine = LaundryKMeans()

        st.markdown("### Langkah 1: Pembersihan dan Filter Data")
        df_clean = engine.load_and_clean_data(df_raw, months_back=months_back)
        if df_clean is None:
            st.error("Proses berhenti karena data tidak valid.")
            return

        st.markdown("### Langkah 2: Perhitungan RFM")
        rfm = engine.calculate_rfm(df_clean)

        st.markdown("### Langkah 3: Normalisasi Data")
        rfm = engine.normalize_data(rfm)
        st.success("Normalisasi selesai.")

        st.markdown("### Langkah 4: K-Means Clustering")
        rfm = engine.run_kmeans(rfm)

        st.markdown("### Langkah 5: Pelabelan Cluster")
        rfm, cluster_labels = engine.label_clusters(rfm)
        st.success("Cluster berhasil dilabeli.")

        st.markdown("### Langkah 6: Pemilihan TOP 10")
        top_10 = engine.get_top_10_customers(rfm)
        st.success(f"{len(top_10)} pelanggan terpilih.")

        st.session_state['rfm'] = rfm
        st.session_state['top_10'] = top_10
        st.session_state['cluster_labels'] = cluster_labels
        st.session_state['df_clean'] = df_clean
        st.session_state['wa_message'] = build_whatsapp_message(top_10)

    st.success("Analisis selesai.")
    st.rerun()

    if 'rfm' not in st.session_state:
        return

    rfm = st.session_state['rfm']
    top_10 = st.session_state['top_10']
    cluster_labels = st.session_state['cluster_labels']
    df_clean = st.session_state['df_clean']

    st.markdown("---")
    st.markdown("## TOP 10 Pelanggan Loyal")
    st.caption("Pelanggan berikut dipilih berdasarkan nilai RFM tertinggi dan berhak mendapat diskon bulan depan.")

    display = top_10.copy()
    display.insert(0, 'No', range(1, len(display) + 1))
    display['Total Belanja'] = display['Monetary'].apply(lambda x: f"Rp {x:,.0f}")
    display['Diskon'] = display['Discount'].apply(lambda x: f"{x}%")

    st.dataframe(
        display[['No', 'Konsumen', 'Segment', 'Frequency', 'Total Belanja', 'Diskon']].rename(columns={
            'Konsumen': 'Nama Pelanggan',
            'Segment': 'Segmen',
            'Frequency': 'Frekuensi Transaksi'
        }),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.markdown("### Bagikan dan Unduh Laporan")

    cluster_summary = rfm.groupby('Segment').agg(
        Jumlah_Pelanggan=('Konsumen', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Diskon=('Discount', 'first')
    ).reset_index()

    col1, col2, col3 = st.columns(3)

    with col1:
        wa_link = make_whatsapp_link(st.session_state.get('wa_message', ''))
        st.link_button("Kirim via WhatsApp", wa_link, use_container_width=True)

    with col2:
        excel_data = export_excel(rfm, top_10, cluster_summary)
        st.download_button(
            label="Unduh Excel",
            data=excel_data,
            file_name=f"Laporan_Segmentasi_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col3:
        csv = top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_csv(index=False)
        st.download_button(
            label="Unduh CSV",
            data=csv,
            file_name=f"TOP10_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with st.expander("Edit pesan WhatsApp"):
        edited_msg = st.text_area(
            "Pesan",
            value=st.session_state.get('wa_message', ''),
            height=300,
            label_visibility="collapsed"
        )
        if st.button("Simpan perubahan"):
            st.session_state['wa_message'] = edited_msg
            st.success("Pesan diperbarui.")
            st.rerun()

    st.markdown("---")
    st.markdown("## Ringkasan Analisis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pelanggan", len(rfm))
    with col2:
        st.metric("Total Transaksi", len(df_clean))
    with col3:
        st.metric("VIP Customer", len(rfm[rfm['Segment'] == 'VIP Customer']))
    with col4:
        period = f"{df_clean['Tanggal'].min().strftime('%d/%m')} - {df_clean['Tanggal'].max().strftime('%d/%m/%Y')}"
        st.metric("Periode Data", period)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(chart_segment_distribution(rfm), use_container_width=True)
    with col_chart2:
        st.plotly_chart(chart_rfm_scatter(rfm), use_container_width=True)

    st.markdown("## Detail Segmentasi Pelanggan")
    st.caption("Klik masing-masing segmen untuk melihat daftar pelanggan dan statistik detail.")

    sorted_clusters = sorted(cluster_labels.items(), key=lambda x: x[1]['priority'])

    for cluster_id, info in sorted_clusters:
        cluster_data = rfm[rfm['Cluster'] == cluster_id]
        label = f"{info['name']} ({len(cluster_data)} pelanggan) — Diskon {info['discount']}%"

        with st.expander(label):
            st.caption(info['description'])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Avg Recency", f"{cluster_data['Recency'].mean():.0f} hari")
            with c2:
                st.metric("Avg Frequency", f"{cluster_data['Frequency'].mean():.1f}x")
            with c3:
                st.metric("Avg Monetary", f"Rp {cluster_data['Monetary'].mean():,.0f}")
            with c4:
                st.metric("Diskon", f"{info['discount']}%")

            st.markdown("**Daftar Pelanggan**")
            st.dataframe(
                cluster_data[['Konsumen', 'Recency', 'Frequency', 'Monetary']].sort_values('Monetary', ascending=False),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")
    st.caption("Anty Laundry &mdash; Sistem Segmentasi Pelanggan v2.1 &mdash; 2025")


if __name__ == "__main__":
    main()

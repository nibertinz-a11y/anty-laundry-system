"""
ANTY LAUNDRY - SISTEM SEGMENTASI PELANGGAN
v2.1 - Fixed: Hitung semua transaksi (termasuk piutang), TOP 10 di atas, WA editable
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO
import urllib.parse

st.set_page_config(page_title="Anty Laundry", page_icon="🧺", layout="wide")

# CSS Styling (dipendekkan)
st.markdown("""
<style>
    * { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%); }
    .main-header { background: linear-gradient(135deg, #6366f1 0%, #d946ef 100%); padding: 2rem; border-radius: 20px; color: white; margin-bottom: 2rem; }
    h1, h2, h3 { color: #f3f4f6; }
    .stButton>button { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; font-weight: 700; border-radius: 12px; padding: 1rem 2rem; width: 100%; }
    textarea { background: rgba(30, 27, 75, 0.5) !important; border: 1px solid rgba(99, 102, 241, 0.3) !important; color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)


class AntyLaundryKMeans:
    def __init__(self):
        self.n_clusters = 5
        self.scaler = StandardScaler()
    
    def find_column(self, df, keywords):
        for keyword in keywords:
            for col in df.columns:
                if keyword.lower().replace(' ', '') in str(col).lower().replace(' ', ''):
                    return col
        return None
    
    def load_and_clean_data(self, df, months_back=1):
        df = df.copy()
        
        # Auto detect columns
        tanggal_col = self.find_column(df, ['tanggal ambil', 'tanggalar'])
        konsumen_col = self.find_column(df, ['konsumen', 'konsumer'])
        harga_col = self.find_column(df, ['total harga', 'totalharga'])
        invoice_col = self.find_column(df, ['nota', 'invoice'])
        status_col = self.find_column(df, ['status order', 'status'])
        
        if not tanggal_col or not konsumen_col or not harga_col:
            st.error("❌ Kolom penting tidak ditemukan!")
            return None
        
        # Rename
        col_map = {
            tanggal_col: 'Tanggal',
            konsumen_col: 'Konsumen',
            harga_col: 'Total_Harga'
        }
        if invoice_col: col_map[invoice_col] = 'No_Invoice'
        if status_col: col_map[status_col] = 'Status_Order'
        
        df = df.rename(columns=col_map)
        
        # 🔴 PENTING: Hanya filter transaksi BATAL, sisanya tetap dihitung (termasuk piutang/belum diambil)
        if 'Status_Order' in df.columns:
            before = len(df)
            df = df[~df['Status_Order'].astype(str).str.lower().str.contains('batal', na=False)]
            after = len(df)
            if before > after:
                st.warning(f"⚠️ {before - after} transaksi BATAL dihapus")
        
        # Parse tanggal - gunakan Tanggal Order jika Tanggal Ambil kosong
        try:
            # Coba cari Tanggal Order sebagai alternatif
            tanggal_order_col = self.find_column(df, ['tanggal order', 'tanggalorder'])
            if tanggal_order_col:
                df['Tanggal_Order'] = pd.to_datetime(df[tanggal_order_col], errors='coerce')
            
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            
            # Jika Tanggal Ambil kosong, gunakan Tanggal Order
            if 'Tanggal_Order' in df.columns:
                df['Tanggal'] = df['Tanggal'].fillna(df['Tanggal_Order'])
            
            df = df.dropna(subset=['Tanggal'])
        except Exception as e:
            st.error(f"❌ Error parsing tanggal: {str(e)}")
            return None
        
        # Filter periode
        max_date = df['Tanggal'].max()
        cutoff_date = max_date - timedelta(days=30 * months_back)
        
        st.info(f"📅 Filter dari: {cutoff_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")
        
        df = df[df['Tanggal'] > cutoff_date]
        
        # Clean harga
        df['Total_Harga'] = pd.to_numeric(df['Total_Harga'], errors='coerce')
        df = df[df['Total_Harga'] > 0]
        
        df['Konsumen'] = df['Konsumen'].astype(str).str.strip()
        df = df.dropna(subset=['Total_Harga', 'Konsumen'])
        
        st.success(f"✅ Total {len(df)} transaksi dari {len(df['Konsumen'].unique())} pelanggan unik")
        
        return df
    
    def calculate_rfm(self, df):
        ref_date = df['Tanggal'].max()
        
        if 'No_Invoice' in df.columns:
            rfm = df.groupby('Konsumen').agg({
                'Tanggal': lambda x: (ref_date - x.max()).days,
                'No_Invoice': 'nunique',
                'Total_Harga': 'sum'
            }).reset_index()
        else:
            rfm = df.groupby('Konsumen').agg({
                'Tanggal': lambda x: (ref_date - x.max()).days,
                'Total_Harga': ['count', 'sum']
            }).reset_index()
            rfm.columns = ['Konsumen', 'Recency', 'Frequency', 'Monetary']
            return rfm
        
        rfm.columns = ['Konsumen', 'Recency', 'Frequency', 'Monetary']
        return rfm.dropna()
    
    def run_kmeans(self, rfm):
        X = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        model = KMeans(n_clusters=5, random_state=42, n_init=10)
        rfm['Cluster'] = model.fit_predict(X)
        return rfm
    
    def label_clusters(self, rfm):
        stats = rfm.groupby('Cluster').agg({
            'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
        })
        
        labels = {}
        for cid, row in stats.iterrows():
            if row['Recency'] < 15 and row['Frequency'] >= 3 and row['Monetary'] > rfm['Monetary'].quantile(0.75):
                labels[cid] = {
                    'name': 'VIP Champions', 'icon': '🏆', 'discount': 15,
                    'desc': 'Pelanggan terbaik: transaksi sering, baru, nilai tinggi. Aset utama yang harus dipertahankan.'
                }
            elif row['Recency'] < 20 and row['Frequency'] >= 2 and row['Monetary'] > rfm['Monetary'].quantile(0.5):
                labels[cid] = {
                    'name': 'High Value Loyal', 'icon': '💎', 'discount': 10,
                    'desc': 'Pelanggan setia bernilai tinggi. Sering kembali, berpotensi naik ke VIP.'
                }
            elif row['Frequency'] >= 2 or row['Monetary'] > rfm['Monetary'].quantile(0.3):
                labels[cid] = {
                    'name': 'Regular Loyal', 'icon': '💚', 'discount': 5,
                    'desc': 'Pelanggan konsisten. Basis stabil yang dapat ditingkatkan loyalitasnya.'
                }
            elif row['Recency'] > 25:
                labels[cid] = {
                    'name': 'Sleeping Customers', 'icon': '😴', 'discount': 10,
                    'desc': 'Tidak aktif lama. Perlu reaktivasi dengan promo menarik.'
                }
            else:
                labels[cid] = {
                    'name': 'At Risk', 'icon': '⚠️', 'discount': 7,
                    'desc': 'Berpotensi berhenti. Butuh perhatian khusus untuk mencegah churn.'
                }
        
        rfm['Segment'] = rfm['Cluster'].map(lambda x: labels[x]['name'])
        rfm['Discount'] = rfm['Cluster'].map(lambda x: labels[x]['discount'])
        rfm['Description'] = rfm['Cluster'].map(lambda x: labels[x]['desc'])
        
        return rfm, labels
    
    def get_top_10(self, rfm):
        top = rfm[rfm['Segment'].isin(['VIP Champions', 'High Value Loyal'])].nlargest(10, 'Monetary')
        if len(top) < 10:
            extra = rfm[~rfm['Konsumen'].isin(top['Konsumen'])].nlargest(10 - len(top), 'Monetary')
            top = pd.concat([top, extra])
        return top.head(10)


def generate_wa_message(top_10):
    msg = "🎉 *SELAMAT PELANGGAN SETIA ANTY LAUNDRY!* 🎉\n\n"
    msg += "Anda terpilih sebagai TOP 10 pelanggan terbaik! 🏆\n\n"
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        msg += f"{i}. *{row['Konsumen']}*\n"
        msg += f"   💎 {row['Segment']}\n"
        msg += f"   🎁 Diskon: *{row['Discount']}%*\n"
        msg += f"   💰 Rp {row['Monetary']:,.0f}\n\n"
    
    msg += "📅 Berlaku bulan depan\n💳 Tunjukkan pesan ini saat transaksi\n\n"
    msg += "━━━━━━━━━━━━━━━━\n🧺 ANTY LAUNDRY\n📍 Tomohon\n📞 [Nomor telp]"
    return msg

def export_excel(rfm, top_10):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        top_10[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_excel(w, sheet_name='Top 10', index=False)
        rfm[['Konsumen', 'Segment', 'Recency', 'Frequency', 'Monetary', 'Discount']].to_excel(w, sheet_name='Semua', index=False)
    out.seek(0)
    return out


def main():
    st.markdown('<div class="main-header"><h1>🧺 ANTY LAUNDRY</h1><h3>Sistem Segmentasi Pelanggan</h3></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        months = st.selectbox("📅 Periode:", [1,2,3,6,12], index=0)
        st.markdown("---")
        st.markdown("### 💡 Info")
        st.info("Sistem menghitung SEMUA transaksi termasuk piutang & belum diambil")
        st.markdown("---")
        st.caption("v2.1 © 2025")
    
    st.markdown("## 📤 Upload Data")
    file = st.file_uploader("Upload Excel/CSV dari kasir", type=['xlsx','xls','csv'])
    
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        st.success(f"✅ {file.name}")
        
        with st.expander("👀 Preview"):
            st.dataframe(df.head(10))
        
        if st.button("🚀 Jalankan Analisis", type="primary"):
            with st.spinner("⏳ Memproses..."):
                engine = AntyLaundryKMeans()
                
                df_clean = engine.load_and_clean_data(df, months)
                if df_clean is None: st.stop()
                
                rfm = engine.calculate_rfm(df_clean)
                rfm = engine.run_kmeans(rfm)
                rfm, labels = engine.label_clusters(rfm)
                top_10 = engine.get_top_10(rfm)
                
                st.session_state.update({'rfm': rfm, 'top_10': top_10, 'labels': labels, 'df': df_clean})
            
            st.success("✅ Selesai!")
            st.balloons()
    
    if 'top_10' in st.session_state:
        rfm = st.session_state['rfm']
        top_10 = st.session_state['top_10']
        
        st.markdown("---")
        st.markdown("## 🏆 TOP 10 Pelanggan Diskon")
        
        st.info("💡 Dipilih dari VIP Champions & High Value Loyal berdasarkan total belanja")
        
        top_10['#'] = range(1, len(top_10)+1)
        st.dataframe(
            top_10[['#', 'Konsumen', 'Segment', 'Frequency', 'Monetary', 'Discount']]
            .assign(Monetary=lambda x: x['Monetary'].apply(lambda v: f"Rp {v:,.0f}")),
            use_container_width=True, hide_index=True
        )
        
        st.markdown("### 💬 Kirim Pesan")
        st.markdown("**✏️ Edit pesan sebelum kirim:**")
        msg = st.text_area("Pesan WhatsApp", generate_wa_message(top_10), height=400)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("💬 Kirim WA", f"https://wa.me/?text={urllib.parse.quote(msg)}", use_container_width=True)
        with col2:
            st.download_button("📊 Excel", export_excel(rfm, top_10), f"Laporan_{datetime.now():%Y%m%d}.xlsx", use_container_width=True)
        with col3:
            st.download_button("📄 CSV", top_10.to_csv(index=False), f"Top10_{datetime.now():%Y%m%d}.csv", use_container_width=True)
        
        st.markdown("---")
        st.markdown("## 📊 Ringkasan")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pelanggan", len(rfm))
        col2.metric("Total Transaksi", len(st.session_state['df']))
        col3.metric("Cluster", 5)
        
        st.markdown("---")
        st.markdown("## 🎯 Detail Segmen")
        
        st.info("📌 **Penjelasan Segmen:** Setiap segmen memiliki karakteristik RFM berbeda yang menentukan strategi marketing")
        
        for cid in sorted(rfm['Cluster'].unique()):
            data = rfm[rfm['Cluster'] == cid]
            info = st.session_state['labels'][cid]
            
            with st.expander(f"{info['icon']} {info['name']} ({len(data)} pelanggan) - Diskon {info['discount']}%"):
                st.info(f"📝 {info['desc']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Recency", f"{data['Recency'].mean():.0f} hari")
                col2.metric("Avg Frequency", f"{data['Frequency'].mean():.1f}x")
                col3.metric("Avg Monetary", f"Rp {data['Monetary'].mean():,.0f}")
                
                st.dataframe(
                    data[['Konsumen','Recency','Frequency','Monetary']]
                    .sort_values('Monetary', ascending=False)
                    .assign(Monetary=lambda x: x['Monetary'].apply(lambda v: f"Rp {v:,.0f}")),
                    use_container_width=True, hide_index=True
                )

if __name__ == "__main__":
    main()

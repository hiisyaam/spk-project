from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.cluster import KMeans
import io

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/proses-otomatis/")
async def hitung_spk_otomatis(
    file: UploadFile = File(...),
    w_modul: float = Form(...),
    w_utp: float = Form(...),
    w_uap: float = Form(...),
    w_keaktifan: float = Form(...)
):
    # 1. Baca file CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # ==========================================================
    # 1.5 DATA CLEANING: Standarisasi Nama Kolom
    # ==========================================================
    # A. Hapus spasi di awal/akhir nama kolom (misal " Nama " jadi "Nama")
    df.columns = df.columns.str.strip()
    
    # B. Buat kamus pemetaan agar kebal huruf besar/kecil
    rename_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'nama': rename_mapping[col] = 'Nama'
        elif col_lower == 'nim': rename_mapping[col] = 'NIM'
        elif col_lower == 'utp': rename_mapping[col] = 'UTP'
        elif col_lower == 'uap': rename_mapping[col] = 'UAP'
        elif col_lower == 'keaktifan': rename_mapping[col] = 'Keaktifan'
        
    # Terapkan perubahan nama kolom
    df.rename(columns=rename_mapping, inplace=True)
    # ==========================================================
    
    # 2. Pre-processing: Deteksi otomatis kolom modul 1-13
    kolom_modul = [col for col in df.columns if 'modul' in col.lower()]
    kolom_nilai = kolom_modul + ['UTP', 'UAP', 'Keaktifan']
    
    # Looping untuk membersihkan semua kolom nilai (menangani teks/koma)
    for col in kolom_nilai:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Hitung Rata-rata Modul
    df['Rata_Modul'] = df[kolom_modul].mean(axis=1)
    
    # 3. SPK (Metode SAW) - Menggunakan bobot dinamis dari UI
    # ... (kode ke bawahnya tetap sama persis seperti sebelumnya) ...
    bobot = {
        'Rata_Modul': w_modul,
        'UTP': w_utp,
        'UAP': w_uap,
        'Keaktifan': w_keaktifan
    }
    
    # Normalisasi (Benefit)
    df['Norm_Modul'] = df['Rata_Modul'] / df['Rata_Modul'].max()
    df['Norm_UTP'] = df['UTP'] / df['UTP'].max()
    df['Norm_UAP'] = df['UAP'] / df['UAP'].max()
    df['Norm_Keaktifan'] = df['Keaktifan'] / df['Keaktifan'].max()
    
    # Hitung Skor Akhir SPK
    df['Skor_Akhir'] = (
        (df['Norm_Modul'] * bobot['Rata_Modul']) + 
        (df['Norm_UTP'] * bobot['UTP']) + 
        (df['Norm_UAP'] * bobot['UAP']) + 
        (df['Norm_Keaktifan'] * bobot['Keaktifan'])
    )
    
    # Urutkan Ranking
    df = df.sort_values(by='Skor_Akhir', ascending=False).reset_index(drop=True)
    
    # 4. Data Mining (K-Means Clustering)
    fitur_clustering = df[['Rata_Modul', 'UTP', 'UAP', 'Keaktifan']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(fitur_clustering)
    
    hasil_ranking = df[['NIM', 'Nama', 'Rata_Modul', 'UTP', 'UAP', 'Keaktifan', 'Skor_Akhir', 'Cluster']].to_dict(orient="records")
    
    return {
        "status": "success",
        "insight": f"Data berhasil diproses. Dari {len(df)} praktikan, model mengelompokkan mereka ke dalam 3 profil berdasarkan bobot yang Anda tentukan.",
        "data": hasil_ranking
    }
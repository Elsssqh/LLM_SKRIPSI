import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
import numpy as np

def calculate_metrics_for_column(df_pakar, df_ai, column_name):
    """
    Fungsi helper untuk menghitung QWK dan Interpretasi.
    MAE DIHAPUS sesuai permintaan.
    """
    
    # 1. Ekstrak dan bersihkan data
    y_true_raw = pd.to_numeric(df_pakar[column_name], errors='coerce')
    y_pred_raw = pd.to_numeric(df_ai[column_name], errors='coerce')

    # 2. Gabungkan dan bersihkan data yang tidak valid (NaN)
    comparison_df = pd.DataFrame({
        'human_score': y_true_raw,
        'ai_score': y_pred_raw
    })
    clean_df = comparison_df.dropna()

    if len(clean_df) == 0:
        return None # Kembalikan None jika tidak ada data valid

    # --- PENTING: Pembulatan untuk QWK ---
    y_true_rounded = clean_df['human_score'].round().astype(int)
    y_pred_rounded = clean_df['ai_score'].round().astype(int)
    
    # 3. Hitung QWK (menggunakan skor yang dibulatkan)
    qwk = cohen_kappa_score(y_true_rounded, y_pred_rounded, weights='quadratic')

    # 4. MAE Dihapus
    
    # 5. Interpretasi QWK berdasarkan skala umum (Tetap ada)
    if qwk >= 0.80:
        interpretation = "Kesepakatan hampir sempurna" # (0.80-1.00)
    elif qwk >= 0.60:
        interpretation = "Kesepakatan tinggi" # (0.60-0.79)
    elif qwk >= 0.40:
        interpretation = "Kesepakatan moderat" # (0.40-0.59)
    elif qwk >= 0.20:
        interpretation = "Kesepakatan rendah" # (0.20-0.39)
    else:
        interpretation = "Kesepakatan sangat rendah" # (< 0.20)
        
    return {
        "qwk": qwk,
        # "mae": mae, # MAE Dihapus
        "interpretation": interpretation,
        "valid_rows": len(clean_df),
        "total_rows": len(comparison_df)
    }

def evaluate_from_excel(excel_file_path: str, pakar_sheet_name: str, ai_sheet_name: str):
    """
    Menghitung QWK dan Interpretasi untuk SEMUA kolom metrik dari dua sheet
    dalam satu file Excel.
    """
    try:
        # 1. Muat kedua dataset dari sheet yang berbeda di file Excel yang sama
        df_pakar = pd.read_excel(excel_file_path, sheet_name=pakar_sheet_name)
        df_ai = pd.read_excel(excel_file_path, sheet_name=ai_sheet_name)

        # --- Membersihkan nama kolom (menghapus spasi di awal/akhir) ---
        df_pakar.columns = df_pakar.columns.str.strip()
        df_ai.columns = df_ai.columns.str.strip()

        # --- DAFTAR METRIK DARI PROPOSAL/DATA ---
        score_columns = [
            'Overall Scoring', 
            'Grammar', 
            'Vocabulary', 
            'Coherence', 
            'Cultural Adaptation'
        ]

        # 2. Validasi Kolom Metrik
        valid_columns = []
        for col in score_columns:
            if col not in df_pakar.columns:
                print(f"Peringatan: Kolom '{col}' tidak ditemukan di sheet '{pakar_sheet_name}'")
            elif col not in df_ai.columns:
                print(f"Peringatan: Kolom '{col}' tidak ditemukan di sheet '{ai_sheet_name}'")
            else:
                valid_columns.append(col)
        
        if not valid_columns:
            print("Error: Tidak ada kolom metrik yang valid untuk dibandingkan.")
            return

        # 3. Tampilkan Hasil (Header) - DIPERBARUI
        print("--- Hasil Evaluasi QWK & Interpretasi ---")
        print(f"File Excel: {excel_file_path}")
        print(f"Sheet Pakar: '{pakar_sheet_name}'")
        print(f"Sheet AI   : '{ai_sheet_name}'")
        print("-" * 65) # Lebar disesuaikan
        # Menyesuaikan lebar kolom untuk output yang rapi
        print(f"{'Metrik':<22} | {'QWK':<8} | {'Interpretasi':<30}")
        print("-" * 65)

        # Loop melalui kolom yang valid dan hitung
        for col in valid_columns:
            results = calculate_metrics_for_column(df_pakar, df_ai, col)
            
            if results:
                # Mencetak setiap baris hasil dengan format yang sama - DIPERBARUI
                print(f"{col:<22} | {results['qwk']:<8.4f} | {results['interpretation']}")
            else:
                print(f"{col:<22} | {'-':<8} | {'-':<30}")
        
        print("-" * 65)

    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di path '{excel_file_path}'")
    except ImportError:
        print("Error: Library 'openpyxl' tidak ditemukan.")
        print("Silakan instal dengan menjalankan: pip install openpyxl")
    except ValueError as e:
        # Ini akan menangkap error jika nama sheet tidak ditemukan
        print(f"Error: Nama sheet tidak ditemukan. Pastikan nama sudah benar. Detail: {e}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# --- PENTING: Tentukan path ke file Anda ---
# MEMPERBAIKI SyntaxError: (unicode error)
# Gunakan forward slashes '/' alih-alih backslashes '\'
EXCEL_FILE_PATH = "BackEnd/QWK.xlsx" 
PAKAR_SHEET_NAME = "Pakar" # Ganti jika nama sheet-nya berbeda
AI_SHEET_NAME = "AI"     # Ganti jika nama sheet-nya berbeda

if __name__ == "__main__":
    evaluate_from_excel(EXCEL_FILE_PATH, PAKAR_SHEET_NAME, AI_SHEET_NAME)
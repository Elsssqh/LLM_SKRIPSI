# # backend/main.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import json
# import logging
# import os # Impor os untuk memeriksa file

# # --- SETUP LOGGING ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- IMPORT MODEL ---
# # Ini akan mengimpor class QwenScorer BARU dari model.py (hasil Langkah 2)
# from model import QwenScorer 

# # --- FASTAPI APP SETUP ---
# app = FastAPI(title="AutoGrade-X: Mandarin Essay Scorer (PROMPT TUNED)")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- INITIALIZE MODEL ---
# # Pastikan file .pt ada sebelum memuat
# prompt_files = ["scoring_soft_prompt.pt", "error_soft_prompt.pt", "feedback_soft_prompt.pt"]
# for f in prompt_files:
#     if not os.path.exists(f):
#         logger.critical(f"FATAL ERROR: File prompt '{f}' tidak ditemukan!")
#         logger.critical("Pastikan Anda sudah menjalankan 'train_prompts.py' dan meletakkan file .pt di folder ini.")
#         # Hentikan aplikasi jika file tidak ada
#         raise FileNotFoundError(f"File prompt '{f}' tidak ditemukan.")

# logger.info("Semua file .pt ditemukan. Memuat model...")
# scorer = QwenScorer()
# logger.info("Model QwenScorer (Prompt Tuned) berhasil dimuat.")


# # --- REQUEST MODEL ---
# class EssayRequest(BaseModel):
#     text: str
#     hsk_level: int

# # --- API ENDPOINT ---
# @app.post("/score")
# async def score_essay(req: EssayRequest):
#     """
#     Endpoint utama untuk menilai esai HSK menggunakan soft prompts.
#     """
#     essay = req.text
#     hsk_level = req.hsk_level

#     logger.info(f"Received essay for HSK {hsk_level} (length: {len(essay)} chars).")

#     if hsk_level not in [1, 2, 3]: # Anda bisa sesuaikan ini
#         raise HTTPException(status_code=400, detail="HSK level must be 1, 2, or 3")

#     try:
#         # 1. Panggil fungsi generate_json dari model.py BARU
#         # Ini akan mengembalikan string JSON yang SUDAH BERSIH
#         raw_json_string_from_model = scorer.generate_json(essay, hsk_level)

#         logger.info(f"Raw JSON string from model: {raw_json_string_from_model[:200]}...")

#         # 2. Karena outputnya sudah string JSON bersih, kita tinggal parse
#         # Kita tidak perlu helper 'generate_json_from_llm_output' lagi
#         parsed_data = json.loads(raw_json_string_from_model)

#         # 3. Kembalikan datanya
#         # (Logika standarisasi Anda dari main.py lama bisa dibuang
#         # karena model.py baru sudah menghasilkan format yang benar)
#         return parsed_data

#     except json.JSONDecodeError as e:
#         logger.error(f"JSON Decode Error: {e}")
#         logger.error(f"Raw output was: {raw_json_string_from_model}")
#         raise HTTPException(status_code=500, detail=f"Backend error: Invalid JSON from Model. {str(e)}")

#     except Exception as e:
#         logger.exception(f"General Error in /score: {e}")
#         raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

# -*- coding: utf-8 -*-
# FILE: main.py
# VERSI FINAL: Menggabungkan logika "Prompt Tuning" dengan API Server

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # <-- PENTING untuk menghubungkan React
import json

# Impor semua kode Anda dari model.py
# Pastikan file model.py berada di direktori yang sama
from model import QwenScorer, logger 

# ---------------- API Setup ----------------
app = FastAPI(
    title="Shu - Chinese Essay Scoring API",
    description="API for scoring Chinese essays using a prompt-tuned Qwen model.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Ini memungkinkan aplikasi React Anda (yang berjalan di port lain)
# untuk berkomunikasi dengan API Python ini.
origins = [
    "http://localhost",
    "http://localhost:3000", # Port default untuk Create React App
    "http://localhost:5173", # Port default untuk Vite/React
    # Tambahkan URL frontend production Anda di sini jika perlu
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Data Models ----------------
# Ini mendefinisikan struktur data yang diharapkan untuk permintaan (request)
class EssayRequest(BaseModel):
    essay: str
    hsk_level: int = 3


# ---------------- API Initialization ----------------
# Muat model hanya sekali saat aplikasi dimulai untuk efisiensi
scorer = None

@app.on_event("startup")
def load_model():
    global scorer
    logger.info("Server startup: Memuat model QwenScorer...")
    try:
        scorer = QwenScorer()
        logger.info("Model QwenScorer berhasil dimuat.")
    except Exception as e:
        logger.critical(f"Gagal memuat model saat startup: {e}", exc_info=True)
        # Jika model gagal dimuat, server tidak akan berguna.
        # Anda bisa memilih untuk menghentikan server di sini jika perlu.


# ---------------- API Endpoint ----------------
@app.post("/score-essay/")
def score_essay(request: EssayRequest):
    """
    Endpoint utama untuk menerima esai dan mengembalikannya dengan skor.
    """
    if not scorer:
        # Jika model gagal dimuat saat startup, kembalikan error
        return {"error": "Model not loaded. Please check server logs."}
    
    logger.info(f"Menerima permintaan untuk HSK Level {request.hsk_level}")
    
    # Panggil fungsi generate_json yang sudah Anda buat
    result_json_string = scorer.generate_json(request.essay, request.hsk_level)
    
    # Konversi string JSON menjadi dictionary Python agar FastAPI dapat menanganinya dengan benar
    result_dict = json.loads(result_json_string)
    
    return result_dict

# ---------------- How to Run ----------------
# Simpan file ini sebagai main.py
# Jalankan dari terminal dengan perintah:
# uvicorn main:app --reload
# Server akan berjalan di http://127.0.0.1:8000
# Anda bisa melihat dokumentasi API interaktif di http://127.0.0.1:8000/docs
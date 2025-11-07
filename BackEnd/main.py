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


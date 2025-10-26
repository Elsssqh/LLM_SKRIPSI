# # -*- coding: utf-8 -*-
# # FILE: model.py
# # VERSI FINAL: Menggabungkan logika "Prompt Tuning" (Baru) 
# # dengan parsing (Lama)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import json
# import re
# import logging
# import torch
# import torch.nn as nn
# import time
# import math  # <-- DIPINDAHKAN DARI FILE LAMA
# from typing import List, Tuple, Dict, Optional, Any # <-- DIPINDAHKAN DARI FILE LAMA
# import jieba # <-- DIPINDAHKAN DARI FILE LAMA
# import jieba.posseg as pseg # <-- DIPINDAHKAN DARI FILE LAMA

# # ---------------- Logger ----------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# # ---------------- Helpers (dari file lama) ----------------
# def cosine_similarity(v1: List[float], v2: List[float]) -> float:
#     if not v1 or not v2 or len(v1) != len(v2): return 0.0
#     dot = sum(a * b for a, b in zip(v1, v2))
#     n1 = math.sqrt(sum(a * a for a in v1))
#     n2 = math.sqrt(sum(b * b for b in v2))
#     if n1 == 0 or n2 == 0: return 0.0
#     return dot / (n1 * n2)

# # ---------------- QwenScorer (Versi "Prompt Tuned") ----------------

# class QwenScorer:
#     """
#     Implementasi inference menggunakan "Soft Prompts" yang telah dilatih
#     """

#     def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
#         logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # 1. Muat Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#         # 2. Muat Base Model (BEKU) untuk INFERENCE
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             model_name, device_map="auto", trust_remote_code=True, torch_dtype="auto"
#         ).eval()
        
#         self.config = self.base_model.config
#         logger.info("Model Qwen-1.8B berhasil dimuat dan diatur ke mode eval.")

#         # 3. MUAT PARAMETER SOFT PROMPT YANG SUDAH DILATIH
#         try:
#             self.error_prompt = self._load_soft_prompt("error_soft_prompt.pt")
#             self.scoring_prompt = self._load_soft_prompt("scoring_soft_prompt.pt")
#             self.feedback_prompt = self._load_soft_prompt("feedback_soft_prompt.pt")
#             logger.info("Berhasil memuat 3 soft prompt yang telah dilatih.")
#         except FileNotFoundError as e:
#             logger.error(f"Gagal memuat file soft prompt: {e}")
#             logger.error("Pastikan Anda sudah menjalankan 'train_prompts.py' untuk setiap tugas.")
#             raise
            
#         # 4. INISIALISASI JIEBA (DARI FILE LAMA)
#         try:
#             jieba.setLogLevel(logging.INFO)
#             jieba.initialize()
#             logger.info("Jieba berhasil diinisialisasi.")
#         except Exception as e:
#             logger.warning(f"Gagal inisialisasi Jieba sepenuhnya: {e}")
#             pass

#         # 5. INISIALISASI RUBRIK (DARI FILE LAMA)
#         self.rubric_weights = {
#             "grammar": 0.30,
#             "vocabulary": 0.30,
#             "coherence": 0.20,
#             "cultural_adaptation": 0.20
#         }
#         logger.info(f"Rubric weights set (untuk output JSON): {self.rubric_weights}")

#     def _load_soft_prompt(self, prompt_path: str) -> nn.Parameter:
#         """Helper untuk memuat file state_dict prompt."""
#         embed_dim = self.config.hidden_size
        
#         state_dict = torch.load(prompt_path, map_location=self.device)
#         prompt_tensor = state_dict[next(iter(state_dict))]
#         prompt_length = prompt_tensor.shape[1] 
        
#         soft_prompt = nn.Parameter(torch.empty(1, prompt_length, embed_dim))
#         soft_prompt.load_state_dict(state_dict)
#         soft_prompt.to(self.device)
#         logger.info(f"Memuat prompt dari {prompt_path} (panjang: {prompt_length})")
#         return soft_prompt

#     def _get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         """Helper untuk mendapatkan embedding dari token ID."""
#         if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
#              return self.base_model.transformer.wte(input_ids)
#         elif hasattr(self.base_model, 'get_input_embeddings'):
#              return self.base_model.get_input_embeddings()(input_ids)
#         else:
#              raise NotImplementedError("Tidak dapat menemukan layer input embedding.")

#     def _generate_with_prompt(self, text_input: str, soft_prompt: nn.Parameter) -> str:
#         """
#         Fungsi helper untuk melakukan 'generate' menggunakan soft prompt.
#         """
#         tokenized_input = self.tokenizer(text_input, return_tensors="pt").to(self.device)
#         input_ids = tokenized_input.input_ids
#         attention_mask = tokenized_input.attention_mask
        
#         inputs_embeds = self._get_input_embeddings(input_ids)
#         batch_size = inputs_embeds.shape[0]

#         prompt_embeds = soft_prompt.expand(batch_size, -1, -1).to(inputs_embeds.device)
#         combined_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
#         prompt_length = soft_prompt.shape[1]
#         prompt_mask = torch.ones(batch_size, prompt_length, dtype=attention_mask.dtype).to(attention_mask.device)
#         combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)

#         outputs = self.base_model.generate(
#             inputs_embeds=combined_embeds,
#             attention_mask=combined_mask,
#             max_new_tokens=256, 
#             pad_token_id=self.tokenizer.pad_token_id
#         )
        
#         input_token_len = combined_mask.shape[1]
#         generated_ids = outputs[0, input_token_len:]
        
#         return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

#     # --- FUNGSI HELPER DARI FILE LAMA (LENGKAP) ---

#     def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
#         """(Fungsi ini diambil LENGKAP dari file lama Anda)"""
#         try:
#             cleaned_essay = re.sub(r'\s+', '', essay).strip()
#             if not cleaned_essay:
#                 logger.warning("Input esai kosong setelah dibersihkan.")
#                 return "", ""
#             words_with_pos = list(pseg.cut(cleaned_essay))
#             segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
#             pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
#             logger.debug(f"Jieba Segmented: {segmented}")
#             return segmented, pos_lines
#         except Exception as e:
#             logger.exception("Preprocessing Jieba gagal.")
#             return essay, "Jieba preprocessing gagal."
            
#     def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
#         """(Fungsi ini diambil LENGKAP dari file lama Anda)"""
#         validated_error_list = []
#         if "TIDAK ADA KESALAHAN" in error_response or error_response.strip() == "":
#             return []
        
#         pattern = re.compile(r"(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)")
        
#         for line in error_response.split('\n'):
#             line = line.strip()
#             match = pattern.search(line)
            
#             if match:
#                 try:
#                     err_type = match.group(1).strip()
#                     incorrect_frag = match.group(2).strip()
#                     correction = match.group(3).strip()
#                     explanation = match.group(4).strip()
                    
#                     start_index = essay_text.find(incorrect_frag)
#                     if start_index != -1:
#                         end_index = start_index + len(incorrect_frag)
#                         pos = [start_index, end_index]
#                     else:
#                         logger.warning(f"Tidak dapat menemukan posisi untuk fragmen: '{incorrect_frag}'. Menggunakan posisi default [0, 0].")
#                         pos = [0, 0]

#                     validated_error_list.append({
#                         "error_type": err_type,
#                         "error_position": pos,
#                         "incorrect_fragment": incorrect_frag,
#                         "suggested_correction": correction,
#                         "explanation": explanation
#                     })
#                 except Exception as e:
#                     logger.warning(f"Gagal mem-parsing baris error: '{line}'. Error: {e}")
                    
#         return validated_error_list

#     def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
#         """(Fungsi ini diambil LENGKAP dari file lama Anda)"""
#         try:
#             extracted_data = {"score": {}}
#             found_any_score = False
#             patterns = {
#                 "grammar": r"(?:语法准确性|grammar)\s*[:：分]?\s*(\d{1,3})",
#                 "vocabulary": r"(?:词汇水平|vocabulary)\s*[:：分]?\s*(\d{1,3})",
#                 "coherence": r"(?:篇章连贯|连贯性|coherence)\s*[:：分]?\s*(\d{1,3})",
#                 "task_fulfillment": r"(?:任务完成度|task_fulfillment|cultural_adaptation)\s*[:：分]?\s*(\d{1,3})",
#                 "overall": r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})"
#             }

#             for key, pattern in patterns.items():
#                 match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
#                 if match:
#                     score_val = int(match.group(1))
#                     score_clamped = max(0, min(100, score_val))
#                     extracted_data["score"][key] = score_clamped
#                     found_any_score = True
#                     logger.debug(f"Parser Skor: Ditemukan skor {key}={score_clamped}")

#             if not found_any_score:
#                 logger.warning("Parser Skor: Tidak ada skor yang dapat diekstrak dari teks.")
#                 return None
            
#             extracted_data["feedback"] = ""
#             extracted_data["errors"] = []
            
#             return extracted_data
        
#         except Exception as e:
#             logger.error(f"Parser Skor: Ekstraksi skor dari teks gagal total: {e}")
#             return None

#     # --- FUNGSI UTAMA: GENERATE_JSON (VERSI MODIFIKASI LENGKAP) ---
    
#     def generate_json(self, essay: str, hsk_level: int = 3) -> str:
#         """
#         Fungsi utama untuk menilai esai menggunakan "soft prompts" yang telah dilatih.
#         """
#         start_time = time.time()
#         logger.info(f"Menerima permintaan (generate_json) 'Prompt Tuned'")

#         if not essay or not essay.strip():
#             logger.warning("Input esai kosong atau hanya berisi spasi.")
#             error_result = {"error": "Input esai kosong.", "essay": essay}
#             duration = time.time() - start_time
#             error_result["processing_time"] = f"{duration:.2f} detik"
#             return json.dumps(error_result, ensure_ascii=False, indent=2)

#         # --- LANGKAH 1: DETEKSI KESALAHAN (MENGGUNAKAN SOFT PROMPT 1) ---
#         logger.info("Memulai Langkah 1: Mendeteksi Kesalahan (dengan error_prompt.pt)...")
#         validated_error_list = []
#         try:
#             error_response = self._generate_with_prompt(essay, self.error_prompt)
#             logger.debug(f"Langkah 1 (Raw Response): {error_response}")
#             validated_error_list = self._parse_errors_from_text(error_response, essay)
#             logger.info(f"Langkah 1 Selesai. Ditemukan {len(validated_error_list)} kesalahan.")
#         except Exception as e:
#             logger.exception("Langkah 1 (Deteksi Kesalahan) Gagal.")
#             validated_error_list = []

#         # --- LANGKAH 2: PENILAIAN (MENGGUNAKAN SOFT PROMPT 2) ---
#         logger.info("Memulai Langkah 2: Memberikan Skor (dengan scoring_prompt.pt)...")
#         parsed_scores = {}
#         grammar_s, vocab_s, coherence_s, cultural_s, overall_s = 0, 0, 0, 0, 0
#         try:
#             scoring_input = f"HSK: {hsk_level} Esai: {essay}"
#             scoring_response = self._generate_with_prompt(scoring_input, self.scoring_prompt)
#             logger.debug(f"Langkah 2 (Raw Response): {scoring_response}")
            
#             parsed_scores_data = self._extract_scores_from_text(scoring_response)
            
#             if not parsed_scores_data or "score" not in parsed_scores_data:
#                 logger.error("Langkah 2 Gagal: Tidak dapat mem-parsing skor dari model.")
#                 raise ValueError("Gagal mem-parsing skor.")
                
#             parsed_scores = parsed_scores_data.get("score", {})
            
#             grammar_s = parsed_scores.get("grammar", 0)
#             vocab_s = parsed_scores.get("vocabulary", 0)
#             coherence_s = parsed_scores.get("coherence", 0)
#             cultural_s = parsed_scores.get("task_fulfillment", 0)
#             overall_s = parsed_scores.get("overall", 0)

#             if overall_s == 0 and (grammar_s > 0 or vocab_s > 0):
#                 logger.info("Skor 'overall' tidak ada/0. Menghitung berdasarkan bobot rubrik...")
#                 calc_score = (grammar_s * self.rubric_weights["grammar"]) + \
#                              (vocab_s * self.rubric_weights["vocabulary"]) + \
#                              (coherence_s * self.rubric_weights["coherence"]) + \
#                              (cultural_s * self.rubric_weights["cultural_adaptation"])
#                 overall_s = max(0, min(100, int(round(calc_score))))

#             logger.info(f"Langkah 2 Selesai. Skor diterima (Overall: {overall_s}).")

#         except Exception as e:
#             logger.exception("Langkah 2 (Penilaian) Gagal Total.")
#             parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "task_fulfillment": 0, "overall": 0}


#         # --- LANGKAH 3: UMPAN BALIK (MENGGUNAKAN SOFT PROMPT 3) ---
#         logger.info("Memulai Langkah 3: Menghasilkan Feedback (dengan feedback_prompt.pt)...")
#         feedback = "Gagal menghasilkan feedback."
#         try:
#             score_summary = f"Skor: {overall_s}"
#             error_summary = f"Kesalahan: {len(validated_error_list)}"
#             feedback_input = f"Esai: {essay}\n{score_summary}\n{error_summary}"

#             feedback_response = self._generate_with_prompt(feedback_input, self.feedback_prompt)
#             feedback = feedback_response.strip()
            
#             # --- LOGIKA LAMA ANDA DIMASUKKAN KE SINI ---
#             if not feedback:
#                 if not validated_error_list and overall_s > 80:
#                     feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik, tidak ditemukan kesalahan signifikan. Teruslah berusaha!)"
#                 elif validated_error_list:
#                     feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan dalam esai, silakan periksa daftar kesalahan untuk detailnya.)"
#                 else:
#                     feedback = "请重新检查你的作文。(Harap periksa kembali esai Anda.)"
#             # --- AKHIR LOGIKA LAMA ---
            
#             logger.info("Langkah 3 Selesai.")
#         except Exception as e:
#             logger.exception("Langkah 3 (Feedback) Gagal.")
#             # --- LOGIKA LAMA ANDA DIMASUKKAN KE SINI ---
#             if validated_error_list:
#                 feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan dalam esai, silakan periksa daftar kesalahan untuk detailnya.)"
#             elif overall_s > 80:
#                 feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik, tidak ditemukan kesalahan signifikan. Teruslah berusaha!)"
#             # --- AKHIR LOGIKA LAMA ---


#         # --- FINAL: PERAKITAN JSON (DARI FILE LAMA) ---
#         final_result = {
#             "text": essay,
#             "overall_score": overall_s,
#             "detailed_scores": {
#                 "grammar": grammar_s,
#                 "vocabulary": vocab_s,
#                 "coherence": coherence_s,
#                 "cultural_adaptation": cultural_s 
#             },
#             "error_list": validated_error_list,
#             "feedback": feedback
#         }

#         end_time = time.time()
#         duration = end_time - start_time
#         final_result["processing_time"] = f"{duration:.2f} detik"
#         logger.info(f"Semua langkah 'Prompt Tuned' selesai. Waktu pemrosesan: {duration:.2f} detik")

#         return json.dumps(final_result, ensure_ascii=False, indent=2)


# # # ---------------- Simulasi (Main execution) ----------------
# if __name__ == "__main__":
#     logger.setLevel(logging.INFO)
#     logger.warning("="*50)
#     logger.warning("INI ADALAH SCRIPT INFERENCE (model.py)")
#     logger.warning("PASTIKAN ANDA SUDAH MELATIH 3 PROMPT:")
#     logger.warning("1. error_soft_prompt.pt")
#     logger.warning("2. scoring_soft_prompt.pt")
#     logger.warning("3. feedback_soft_prompt.pt")
#     logger.warning("menggunakan 'train_prompts.py' sebelum menjalankan ini.")
#     logger.warning("="*50)
    
#     # Anda bisa meng-uncomment ini untuk tes cepat
#     # try:
#     #     scorer = QwenScorer()
#     #     
#     #     essay_errors = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
#     #     result_json = scorer.generate_json(essay_errors, hsk_level=3)
#     #     print("\n--- HASIL SIMULASI 'PROMPT TUNED' ---")
#     #     print(result_json)
#     #     print("---------------------------------\n")
#     #
#     # except Exception as e:
#     #     logger.critical(f"Gagal menjalankan program utama: {e}", exc_info=True)


# -*- coding: utf-8 -*-
# FILE: model.py
# VERSI FINAL HYBRID: Memuat .pt, Pakai model.chat, Parser Error Diperkuat, JSON Dump Dilindungi
#versi hanya muncul overall skor valid
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import json
# import re
# import logging
# import torch
# import torch.nn as nn
# import time
# import math
# from typing import List, Tuple, Dict, Optional, Any
# import jieba
# import jieba.posseg as pseg
# import os # Untuk cek file .pt

# # ---------------- Logger ----------------
# logger = logging.getLogger(__name__)
# # Set ke INFO, bisa ubah ke DEBUG jika perlu detail lebih
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("matplotlib").setLevel(logging.ERROR)
# logging.getLogger("h5py").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# # ---------------- Helpers ----------------
# def cosine_similarity(v1: List[float], v2: List[float]) -> float:
#     # (Tidak berubah)
#     if not v1 or not v2 or len(v1) != len(v2): return 0.0
#     dot = sum(a * b for a, b in zip(v1, v2))
#     n1 = math.sqrt(sum(a * a for a in v1))
#     n2 = math.sqrt(sum(b * b for b in v2))
#     denominator = n1 * n2
#     # Tambahkan penanganan pembagian dengan nol eksplisit
#     return dot / denominator if denominator != 0 else 0.0

# # ---------------- QwenScorer ----------------

# class QwenScorer:
#     """
#     Implementasi HYBRID: Memuat file .pt TAPI menggunakan model.chat()
#     Fokus: Mendapatkan SKOR OVERALL yang valid.
#     """

#     def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
#         logger.info(f"Memulai inisialisasi QwenScorer (HYBRID) dgn model: {model_name}")
#         try:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"Device: {self.device}")

#             self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#             logger.info("Tokenizer loaded.")

#             # --- Perbaikan CPU (Wajib) ---
#             logger.info("Loading base model...")
#             # Kita GANTI nama variabel dari self.base_model ke self.model agar cocok dgn kode .chat()
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True
#             ).to(self.device).eval()
#             logger.info(f"Model Qwen-1.8B berhasil dimuat dan dipindahkan ke {self.device}.")
#             self.config = self.model.config # Ambil config dari self.model
#             # --- Akhir Perbaikan CPU ---

#             # --- MEMUAT SOFT PROMPT (.pt files) ---
#             # Kita tetap muat ini agar struktur kode sesuai proposal
#             # TAPI KITA TIDAK AKAN PAKAI Embedding-nya di generasi teks
#             logger.info("Loading soft prompt files (.pt)...")
#             prompt_files = ["error_soft_prompt.pt", "scoring_soft_prompt.pt", "feedback_soft_prompt.pt"]
#             # Cek file dulu
#             for f in prompt_files:
#                 if not os.path.exists(f):
#                      logger.critical(f"FATAL: Soft prompt file '{f}' not found in current directory!")
#                      # Tidak raise error di sini agar __init__ selesai, tapi model.chat akan dipakai
#                      # raise FileNotFoundError(f"Soft prompt file '{f}' not found.")
#                      logger.warning(f"File soft prompt '{f}' tidak ditemukan. Model akan tetap berjalan menggunakan model.chat() tanpa tuning.")

#             # Muat file (tapi tidak dipakai di generasi) jika ada
#             try:
#                 # Muat seperti biasa, TAPI variabelnya tidak akan dipakai di generate_json
#                 if os.path.exists("error_soft_prompt.pt"):
#                     self._loaded_error_prompt = self._load_soft_prompt("error_soft_prompt.pt")
#                 if os.path.exists("scoring_soft_prompt.pt"):
#                     self._loaded_scoring_prompt = self._load_soft_prompt("scoring_soft_prompt.pt")
#                 if os.path.exists("feedback_soft_prompt.pt"):
#                     self._loaded_feedback_prompt = self._load_soft_prompt("feedback_soft_prompt.pt")
#                 logger.warning("Soft prompt files (.pt) loaded (if found), BUT WILL NOT BE USED for text generation due to technical issues. Using model.chat() instead.")
#             except Exception as e:
#                 logger.critical(f"Failed to load .pt files even though they might exist: {e}", exc_info=True)
#                 # Jangan raise error agar server tetap bisa start
#             # --- AKHIR MEMUAT SOFT PROMPT ---

#         except Exception as e:
#             logger.exception(f"Failed to load model or tokenizer {model_name}.")
#             raise # Raise error jika model/tokenizer dasar gagal
#         try:
#             jieba.setLogLevel(logging.WARNING) # Kurangi log Jieba
#             jieba.initialize()
#             logger.info("Jieba initialized.")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Jieba fully: {e}")
#             pass

#         self.rubric_weights = {
#             "grammar": 0.30,
#             "vocabulary": 0.30,
#             "coherence": 0.20,
#             "cultural_adaptation": 0.20 # Sesuaikan nama jika perlu (di proposal 'task_fulfillment'?)
#         }
#         logger.info(f"Rubric weights set: {self.rubric_weights}")

#     # Fungsi _load_soft_prompt tetap ada untuk memuat file .pt (jika ada)
#     def _load_soft_prompt(self, prompt_path: str) -> Optional[nn.Parameter]:
#         # Fungsi ini tetap sama seperti sebelumnya
#         try:
#             embed_dim = self.config.hidden_size
#             state = torch.load(prompt_path, map_location=self.device)
#             if isinstance(state, dict):
#                 # ... (logika load dari dict)
#                  try:
#                     prompt_tensor = next(iter(state.values()))
#                     if not isinstance(prompt_tensor, torch.Tensor): raise ValueError(f"Dict content not tensor: {type(prompt_tensor)}")
#                  except Exception as e: raise ValueError(f"Invalid prompt dict {prompt_path}: {state.keys()} | Error: {e}")
#             elif isinstance(state, torch.Tensor):
#                 # ... (logika load dari tensor)
#                 prompt_tensor = state
#             else:
#                 raise TypeError(f"Unrecognized prompt file type: {type(state)}")

#             if prompt_tensor.shape[-1] != embed_dim: raise ValueError(f"Embedding dim mismatch for {prompt_path}")

#             prompt_length = prompt_tensor.shape[1]
#             soft_prompt = nn.Parameter(torch.empty(1, prompt_length, embed_dim, dtype=prompt_tensor.dtype))
#             with torch.no_grad(): soft_prompt.copy_(prompt_tensor)
#             soft_prompt.to(self.device)
#             logger.info(f"Loaded (unused) prompt from {prompt_path} (len: {prompt_length}) to {self.device}")
#             return soft_prompt
#         except Exception as e:
#             logger.error(f"Error loading soft prompt file {prompt_path}: {e}", exc_info=True)
#             return None # Kembalikan None jika gagal load

#     # HAPUS fungsi _get_input_embeddings dan _generate_with_prompt yang lama

#     def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
#         # (Fungsi ini tidak berubah)
#         try:
#             cleaned_essay = re.sub(r'\s+', '', essay).strip()
#             if not cleaned_essay: logger.warning("Empty essay after cleaning."); return "", ""
#             words_with_pos = list(pseg.cut(cleaned_essay))
#             segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
#             pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
#             return segmented, pos_lines
#         except Exception as e: logger.exception("Jieba preprocessing failed."); return essay, "Jieba preprocessing failed."

#     # --- PROMPT BUILDERS ---
#     def _build_error_detection_prompt(self, essay: str) -> str:
#         # (Prompt error detection tidak berubah)
#         return f"""
#         您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。
#         # 您的任务【仅仅】是找出下文中的语法、词汇或语序错误。
#         请【严格】遵守以下格式：
#         - 如果发现错误，请使用此格式： 错误类型 | 错误原文 | 修正建议 | 简短解释
#         - 每个错误占一行。
#         - 如果【没有发现任何错误】，请【只】回答 'TIDAK ADA KESALAHAN'。
#         --- 示例 ---
#         示例 1: 输入: 我妹妹是十岁。 输出: 助词误用(是) | 我妹妹是十岁 | 我妹妹十岁 | 表达年龄时通常不需要'是'。
#         示例 2: 输入: 我们住雅加达在。 输出: 语序干扰(SPOK) | 我们住雅加达在 | 我们住在雅加达 | 地点状语(在雅加达)应放在动词(住)之前。
#         示例 3: 输入: 路很忙。 输出: 词语误用(False Friend) | 路很忙 | 路很拥挤 | '忙'(máng)通常用于人，而非道路。
#         示例 4: 输入: 我喜欢学中文。 输出: TIDAK ADA KESALAHAN
#         --- 示例结束 ---
#         --- 主要任务 ---
#         请分析以下作文，找出所有错误。请严格遵守格式。
#         作文：
#         "{essay}"
#         """

#     # --- PARSER ERROR (Diperkuat Lagi) ---
#     def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
#         """ Parser error yang lebih robust """
#         validated_error_list = []
#         if not error_response or "TIDAK ADA KESALAHAN" in error_response:
#             logger.debug("Error Parser: No errors reported by model.")
#             return []

#         # Pola utama
#         pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
#         # Keyword/pola untuk di-skip
#         skip_patterns = [
#             re.compile(r"^\s*示例 \d+:"), # Baris "示例 x:"
#             re.compile(r"^\s*输入:"),      # Baris "输入:"
#             re.compile(r"^\s*输出:"),      # Baris "输出:"
#             re.compile(r"^\s*---"),        # Baris "---"
#             re.compile(r"^\s*$"),         # Baris kosong
#             re.compile(r"错误类型|错误原文|修正建议|简短解释") # Header tabel
#         ]

#         logger.debug(f"Error Parser: Parsing response: {repr(error_response)}")
#         lines_processed = 0
#         lines_skipped = 0

#         for line in error_response.splitlines(): # Gunakan splitlines
#             line = line.strip()
#             lines_processed += 1

#             # --- Cek skip ---
#             should_skip = False
#             for skip_pattern in skip_patterns:
#                 if skip_pattern.search(line):
#                     logger.debug(f"Error Parser: Skipping line matching skip pattern: '{line}'")
#                     should_skip = True
#                     lines_skipped += 1
#                     break
#             if should_skip: continue
#             # --- Akhir cek skip ---

#             match = pattern.match(line) # Gunakan match()
#             if match:
#                 try:
#                     err_type = str(match.group(1).strip())
#                     incorrect_frag = str(match.group(2).strip())
#                     correction = str(match.group(3).strip())
#                     explanation = str(match.group(4).strip())

#                     if not incorrect_frag:
#                         logger.warning(f"Error Parser: Empty incorrect fragment in line: '{line}'. Skipping.")
#                         continue

#                     start_index = essay_text.find(incorrect_frag)
#                     pos = [start_index, start_index + len(incorrect_frag)] if start_index != -1 else [0, 0]
#                     if start_index == -1: logger.warning(f"Error Parser: Position not found for: '{incorrect_frag}'. Using [0,0].")

#                     error_entry = {
#                         "error_type": err_type, "error_position": pos,
#                         "incorrect_fragment": incorrect_frag,
#                         "suggested_correction": correction, "explanation": explanation
#                     }
#                     if all(isinstance(v, (str, list)) for k, v in error_entry.items() if k != 'error_position') and isinstance(error_entry['error_position'], list):
#                         validated_error_list.append(error_entry)
#                         logger.debug(f"Error Parser: Parsed error: {error_entry}")
#                     else:
#                         logger.error(f"Error Parser: Invalid data type detected in parsed line '{line}'. Skipping. Data: {error_entry}")
#                 except Exception as e:
#                     logger.warning(f"Error Parser: Failed processing matched line: '{line}'. Error: {e}", exc_info=True)
#             else:
#                  logger.warning(f"Error Parser: Line did not match pattern '|': '{line}'")

#         logger.info(f"Error Parser Done. Lines processed: {lines_processed}. Skipped: {lines_skipped}. Valid errors found: {len(validated_error_list)}")
#         return validated_error_list

#     # --- PROMPT SKOR (Sederhana) ---
#     def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
#         # (Prompt sederhana tidak berubah)
#         logger.info("Building simplified scoring prompt (overall only).")
#         return f"""
#         请为这篇HSK{hsk_level}作文打一个总体分数（0-100分）。
#         作文: "{essay}"
#         请【只】回答总体分数，格式如下：
#         总体得分: [分数]
#         """

#     # --- PARSER SKOR (Fleksibel v2 - Perbaikan Regex) ---
#     def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
#         """ Parser skor SUPER FLEKSIBEL v2 """
#         try:
#             extracted_data = {"score": {"grammar": 0, "vocabulary": 0, "coherence": 0,"cultural_adaptation": 0, "overall": 0}}
#             overall_score_found = False
#             logger.debug(f"Extracting score (flexible v2) from: {repr(text)}")
#             match_keyword = re.search(r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})", text, re.I | re.S)
#             if match_keyword:
#                 try:
#                     score_val = int(match_keyword.group(1)); score_clamped = max(0, min(100, score_val))
#                     extracted_data["score"]["overall"] = score_clamped; overall_score_found = True
#                     logger.info(f"Score Parser (Flex v2): Overall score via keyword = {score_clamped}")
#                 except ValueError: logger.warning(f"Score Parser (Flex v2): Value '{match_keyword.group(1)}' not int.")
#             if not overall_score_found:
#                 logger.debug("Score Parser (Flex v2): Keyword not found. Seeking 1-3 digits...")
#                 # --- PERBAIKAN REGEX DARI SEBELUMNYA ---
#                 potential_numbers = re.findall(r'(\d{1,3})', text) # Cari 1-3 digit TANPA \b
#                 # --- AKHIR PERBAIKAN ---
#                 if potential_numbers:
#                     logger.debug(f"Score Parser (Flex v2): Candidates: {potential_numbers}")
#                     for num_str in potential_numbers:
#                         try:
#                             score_val = int(num_str)
#                             if 0 <= score_val <= 100:
#                                 score_clamped = score_val; extracted_data["score"]["overall"] = score_clamped
#                                 overall_score_found = True; logger.info(f"Score Parser (Flex v2): Overall score via digits '{num_str}' = {score_clamped}")
#                                 break # Ambil yang pertama valid
#                         except ValueError: logger.warning(f"Score Parser (Flex v2): Candidate '{num_str}' not int.")
#                     if not overall_score_found: logger.warning("Score Parser (Flex v2): Digits found but none valid (0-100).")
#                 else: logger.warning("Score Parser (Flex v2): No 1-3 digits found.")
#             if not overall_score_found: logger.warning(f"Score Parser (Flex v2): NO overall score extracted from: {repr(text)}")
#             return extracted_data
#         except Exception as e:
#             logger.error(f"Score Parser (Flex v2): Failed: {e}", exc_info=True)
#             return {"score": {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0, "overall": 0}}

#     # --- PROMPT FEEDBACK ---
#     def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
#         # (Prompt feedback tidak berubah)
#         score_summary = f"总体得分: {scores.get('overall', 'N/A')}"
#         error_summary = "未发现主要错误。"
#         if errors: error_summary = "发现的主要错误:\n" + "".join([f"- {err.get('explanation', 'N/A')}\n" for err in errors[:2]])
#         return f"""
# 您是一位友好且善于鼓励的中文老师。
# 您的任务是根据学生的作文、得分和错误，写一段简短的评语（2-3句话）。
# 请用中文书写评语，并在括号()中附上简短的印尼语翻译。
# 学生作文: "{essay}"
# 所得分数: {score_summary}
# 错误备注: {error_summary}
# 请现在撰写您的评语：
# """

#     # --- FUNGSI UTAMA: generate_json (Menggunakan model.chat) ---
#     def generate_json(self, essay: str, hsk_level: int = 3) -> str:
#         start_time = time.time()
#         logger.info(f"Menerima request (HYBRID - model.chat) u/ HSK {hsk_level}.")
#         if not essay or not essay.strip():
#              logger.warning("Empty essay input."); error_result = {"error": "Input essay empty.", "essay": essay, "processing_time": "0.00s"}
#              # Gunakan try-except untuk json.dumps error darurat
#              try: return json.dumps(error_result, ensure_ascii=False, indent=2)
#              except Exception: return '{"error": "Input essay empty and failed to create error JSON."}'


#         # --- LANGKAH 1: ERROR DETECTION ---
#         logger.info("Step 1: Detecting Errors (via model.chat)...")
#         validated_error_list = []
#         try:
#             error_prompt = self._build_error_detection_prompt(essay)
#             error_response, _ = self.model.chat(self.tokenizer, error_prompt, history=None, system="You are a helpful assistant.")
#             logger.debug(f"Step 1 Raw Response: {repr(error_response)}")
#             validated_error_list = self._parse_errors_from_text(error_response, essay) # Pakai parser yg diperkuat
#             logger.info(f"Step 1 Done. Found {len(validated_error_list)} errors.")
#         except Exception as e: logger.exception("Step 1 (Error Detection) Failed."); validated_error_list = []

#         # --- LANGKAH 2: SCORING ---
#         logger.info("Step 2: Getting Overall Score (via model.chat)...")
#         parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0, "overall": 0}
#         try:
#             scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
#             scoring_response, _ = self.model.chat(self.tokenizer, scoring_prompt, history=None, system="You are a helpful assistant.")
#             logger.info(f"Step 2 Raw Response: {repr(scoring_response)}") # INFO level
#             parsed_scores_data = self._extract_scores_from_text(scoring_response) # Pakai parser fleksibel v2
#             if parsed_scores_data and "score" in parsed_scores_data:
#                  parsed_scores = parsed_scores_data["score"]
#                  if parsed_scores.get("overall", 0) > 0: logger.info("Step 2 Successfully PARSED overall score.")
#                  else: logger.warning("Step 2 PARSED ok, but overall score is still 0.")
#             else: logger.error("Step 2 PARSING FAILED strangely.")
#         except Exception as e: logger.exception("Step 2 (Scoring) Failed TOTAL.")
#         overall_s = parsed_scores.get("overall", 0)
#         logger.info(f"Step 2 Done. Final Overall Score: {overall_s}.")

#         # --- LANGKAH 3: FEEDBACK ---
#         logger.info("Step 3: Generating Feedback (via model.chat)...")
#         feedback = "Gagal menghasilkan feedback."
#         try:
#             feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
#             feedback_response, _ = self.model.chat(self.tokenizer, feedback_prompt, history=None, system="You are a helpful assistant.")
#             feedback = feedback_response.strip() if feedback_response else feedback
#             if not feedback: # Fallback
#                 logger.warning("Step 3: model.chat returned empty feedback. Using fallback.")
#                 if not validated_error_list and overall_s > 80: feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik...)"
#                 elif validated_error_list: feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan...)"
#                 else: feedback = "请根据得到的总体分数和错误列表（若有）检查你的作文。(Harap periksa esai...)"
#             logger.info("Step 3 Done.")
#             logger.debug(f"Step 3 Feedback: {repr(feedback)}")
#         except Exception as e:
#             logger.exception("Step 3 (Feedback) Failed TOTAL.")
#             # Fallback jika exception
#             if validated_error_list: feedback = "作文中发现一些错误...(Ada error...)"
#             elif overall_s > 80: feedback = "作文写得很好...(Esai baik...)"
#             else: feedback = "请根据得到的总体分数...(Harap periksa esai...)"

#         # --- FINAL ASSEMBLY ---
#         final_result = {
#             "text": essay, "overall_score": overall_s,
#             "detailed_scores": {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0 },
#             "error_list": validated_error_list, "feedback": feedback
#         }
#         duration = time.time() - start_time
#         final_result["processing_time"] = f"{duration:.2f} detik"
#         logger.info(f"All steps (HYBRID) done. Time: {duration:.2f}s.")

#         # --- Robust JSON Dump (PENTING) ---
#         try:
#             json_output_string = json.dumps(final_result, ensure_ascii=False, indent=2)
#             logger.debug(f"JSON string generated successfully:\n{json_output_string[:500]}...")
#             return json_output_string
#         except TypeError as e:
#             logger.error(f"FATAL: Failed to convert final_result to JSON! Error: {e}", exc_info=True)
#             logger.error(f"Data causing error: {final_result}")
#             error_json = {"error": "Internal error: Failed to serialize result.", "details": str(e), "essay": essay, "processing_time": f"{duration:.2f}s"}
#             # Coba lagi dump error JSON, jika ini gagal juga, kembalikan string error mentah
#             try:
#                 return json.dumps(error_json, ensure_ascii=False, indent=2)
#             except Exception as final_e:
#                  logger.critical(f"ULTRA FATAL: Failed even to dump the error JSON: {final_e}")
#                  return '{"error": "Internal error: Failed to serialize result and error message."}'
#         # --- End Robust JSON Dump ---

# # ---------------- Simulasi (Main execution) ----------------
# if __name__ == "__main__":
#     logger.setLevel(logging.INFO)
#     logger.warning("="*50 + "\nRUNNING model.py SCRIPT (HYBRID VERSION v2)\n" + "="*50)
#     try:
#         scorer = QwenScorer()
#         logger.info("\n" + "="*20 + " SIMULATION 1: Good Essay " + "="*20)
#         essay_1 = "上个星期六，我和朋友去公园玩。我们早上九点起床。我吃早饭，然后穿衣服。朋友开车带我们去公园。公园里有很多人。我们放风筝，吃午饭，然后回家。我玩得很开心。"
#         result_1 = scorer.generate_json(essay_1, hsk_level=2)
#         print("\n--- SIMULATION 1 RESULT (JSON) ---"); print(result_1); print("---------------------------------\n")
#         logger.info("\n" + "="*20 + " SIMULATION 2: Essay Errors " + "="*20)
#         essay_2 = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
#         result_2 = scorer.generate_json(essay_2, hsk_level=3)
#         print("\n--- SIMULATION 2 RESULT (JSON) ---"); print(result_2); print("---------------------------------\n")
#         logger.info("Simulations finished.")
#     except Exception as e:
#         logger.critical(f"Failed to run main simulation: {e}", exc_info=True)
# -*- coding: utf-8 -*-
# Pastikan encoding UTF-8 di awal
# FILE: model.py
# VERSI FINAL: Murni "Chain of Prompts" (Stabil) - Meminta Skor Detail

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import logging
import math
from typing import List, Tuple, Dict, Optional, Any
import time
import jieba
import jieba.posseg as pseg
import torch

# ---------------- Logger ----------------
logger = logging.getLogger(__name__)
# Set ke INFO
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ---------------- Helpers ----------------
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    # (Tidak berubah)
    if not v1 or not v2 or len(v1) != len(v2): return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    denominator = n1 * n2
    return dot / denominator if denominator != 0 else 0.0

# ---------------- QwenScorer ----------------

class QwenScorer:
    """ Implementasi Murni Chain of Prompts via model.chat() """

    def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
        logger.info(f"Memulai inisialisasi QwenScorer (Chain of Prompts) dgn model: {model_name}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Tokenizer loaded.")

            # --- Perbaikan CPU (Wajib) ---
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to(self.device).eval() # Paksa ke CPU
            logger.info(f"Model Qwen-1.8B loaded to {self.device}.")
            self.config = self.model.config
            # --- Akhir Perbaikan CPU ---

            logger.info("Pemuatan file soft prompt (.pt) diskip.") # Konfirmasi skip

        except Exception as e:
            logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
            raise
        try:
            jieba.setLogLevel(logging.WARNING)
            jieba.initialize()
            logger.info("Jieba initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize Jieba fully: {e}")
            pass

        self.rubric_weights = {
            "grammar": 0.30, "vocabulary": 0.30,
            "coherence": 0.20, "cultural_adaptation": 0.20
        }
        logger.info(f"Rubric weights set: {self.rubric_weights}")

    # HAPUS _load_soft_prompt, _get_input_embeddings, _generate_with_prompt

    def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
        # (Tidak berubah)
        try:
            cleaned_essay = re.sub(r'\s+', '', essay).strip()
            if not cleaned_essay: logger.warning("Empty essay after cleaning."); return "", ""
            words_with_pos = list(pseg.cut(cleaned_essay))
            segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
            pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
            return segmented, pos_lines
        except Exception as e: logger.exception("Jieba preprocessing failed."); return essay, "Jieba preprocessing failed."

    # --- PROMPT BUILDERS ---
    def _build_error_detection_prompt(self, essay: str) -> str:
        # (Prompt error detection tidak berubah)
        return f"""
        您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。
        # 您的任务【仅仅】是找出下文中的语法、词汇或语序错误。
        请【严格】遵守以下格式：
        - 如果发现错误，请使用此格式： 错误类型 | 错误原文 | 修正建议 | 简短解释
        - 每个错误占一行。
        - 如果【没有发现任何错误】，请【只】回答 'TIDAK ADA KESALAHAN'。
        --- 示例 ---
        示例 1: 输入: 我妹妹是十岁。 输出: 助词误用(是) | 我妹妹是十岁 | 我妹妹十岁 | 表达年龄时通常不需要'是'。
        示例 2: 输入: 我们住雅加达在。 输出: 语序干扰(SPOK) | 我们住雅加达在 | 我们住在雅加达 | 地点状语(在雅加达)应放在动词(住)之前。
        示例 3: 输入: 路很忙。 输出: 词语误用(False Friend) | 路很忙 | 路很拥挤 | '忙'(máng)通常用于人，而非道路。
        示例 4: 输入: 我喜欢学中文。 输出: TIDAK ADA KESALAHAN
        --- 示例结束 ---
        --- 主要任务 ---
        请分析以下作文，找出所有错误。请严格遵守格式。
        作文：
        "{essay}"
        """

    # --- PARSER ERROR (Robust) ---
    def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
        # (Versi robust dari sebelumnya)
        validated_error_list = []
        if not error_response or "TIDAK ADA KESALAHAN" in error_response: return []
        pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
        skip_patterns = [re.compile(r"^\s*示例 \d+:"), re.compile(r"^\s*输入:"), re.compile(r"^\s*输出:"), re.compile(r"^\s*---"), re.compile(r"^\s*$"), re.compile(r"错误类型|错误原文|修正建议|简短解释")]
        logger.debug(f"Error Parser: Parsing response: {repr(error_response)}")
        lines_processed, lines_skipped = 0, 0
        for line in error_response.splitlines():
            line = line.strip(); lines_processed += 1
            should_skip = any(p.search(line) for p in skip_patterns)
            if should_skip: logger.debug(f"Error Parser: Skipping line: '{line}'"); lines_skipped += 1; continue
            match = pattern.match(line)
            if match:
                try:
                    err_type, incorrect_frag, correction, explanation = map(str, (g.strip() for g in match.groups()))
                    if not incorrect_frag: logger.warning(f"Error Parser: Empty incorrect fragment: '{line}'. Skip."); continue
                    start_index = essay_text.find(incorrect_frag); pos = [start_index, start_index + len(incorrect_frag)] if start_index != -1 else [0, 0]
                    if start_index == -1: logger.warning(f"Error Parser: Pos not found: '{incorrect_frag}'. Use [0,0].")
                    error_entry = {"error_type": err_type, "error_position": pos, "incorrect_fragment": incorrect_frag, "suggested_correction": correction, "explanation": explanation}
                    if all(isinstance(v, (str, list)) for k, v in error_entry.items() if k != 'error_position') and isinstance(pos, list):
                        validated_error_list.append(error_entry); logger.debug(f"Error Parser: Parsed: {error_entry}")
                    else: logger.error(f"Error Parser: Invalid data type: '{line}'. Skip. Data: {error_entry}")
                except Exception as e: logger.warning(f"Error Parser: Failed matched line: '{line}'. Error: {e}", exc_info=True)
            else: logger.warning(f"Error Parser: Line unmatched: '{line}'")
        logger.info(f"Error Parser Done. Lines: {lines_processed}. Skipped: {lines_skipped}. Errors: {len(validated_error_list)}")
        return validated_error_list

    # --- PROMPT SKOR (KEMBALI KE DETAIL) ---
    def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
        """
        Membangun prompt yang meminta SEMUA skor detail (seperti kode asli Anda).
        """
        logger.info("Building DETAILED scoring prompt (requesting all 5 scores).")
        return f"""
        您是HSK作文评分员。
        您的任务【仅仅】是提供分数（0-100）。
        【不要】写任何评语或解释。

        HSK等级: {hsk_level}
        作文: "{essay}"

        请【必须】按照以下纯文本格式提供所有5个分数。
        【不要】写任何其他文字。

        语法准确性: [分数]
        词汇水平: [分数]
        篇章连贯: [分数]
        任务完成度: [分数]
        总体得分: [分数]
        """

    # --- PARSER SKOR (KEMBALI KE ASLI - Cari Keyword) ---
    def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Mengekstrak skor dari output teks. KEMBALI ke versi asli yang mencari keyword.
        """
        try:
            extracted_data = {"score": {}} # Mulai kosong
            found_any_score = False
            # Pola regex asli Anda
            patterns = {
                "grammar": r"(?:语法准确性|grammar)\s*[:：分]?\s*(\d{1,3})",
                "vocabulary": r"(?:词汇水平|vocabulary)\s*[:：分]?\s*(\d{1,3})",
                "coherence": r"(?:篇章连贯|连贯性|coherence)\s*[:：分]?\s*(\d{1,3})",
                # Sesuaikan nama kunci 'cultural_adaptation' jika perlu
                "cultural_adaptation": r"(?:任务完成度|task_fulfillment|cultural_adaptation)\s*[:：分]?\s*(\d{1,3})",
                "overall": r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})"
            }

            logger.debug(f"Score Parser (Keyword): Attempting to extract scores from: {repr(text)}")

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score_val = int(match.group(1))
                        score_clamped = max(0, min(100, score_val))
                        extracted_data["score"][key] = score_clamped # Tambahkan skor jika ditemukan
                        found_any_score = True
                        logger.info(f"Score Parser (Keyword): Found score {key}={score_clamped}")
                    except ValueError:
                        logger.warning(f"Score Parser (Keyword): Value '{match.group(1)}' for '{key}' not int.")
                # else:
                #     logger.debug(f"Score Parser (Keyword): Pattern for '{key}' not found.") # Aktifkan jika perlu

            # Jika SAMA SEKALI tidak ada keyword skor yang cocok
            if not found_any_score:
                 logger.warning(f"Score Parser (Keyword): NO scores could be extracted via keywords from: {repr(text)}")
                 return None # Kembalikan None jika parsing GAGAL TOTAL

            # Jika setidaknya SATU skor ditemukan, pastikan semua key ada (default 0)
            # Ini penting agar generate_json tidak crash saat get()
            final_scores = {}
            for key in patterns: # Iterasi semua key yang DIHARAPKAN
                final_scores[key] = extracted_data["score"].get(key, 0) # Ambil nilai jika ada, else 0
                if key not in extracted_data["score"]:
                     logger.warning(f"Score Parser (Keyword): Score for '{key}' not found, setting to 0.")

            # Kembalikan dictionary yang lengkap dengan skor (atau 0)
            return {"score": final_scores}

        except Exception as e:
            logger.error(f"Score Parser (Keyword): Failed: {e}", exc_info=True)
            return None # Kembalikan None jika ada error tak terduga

    # --- PROMPT FEEDBACK ---
    def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
        # (Prompt feedback tidak berubah, tapi sekarang bisa menampilkan skor detail jika ada)
        score_summary = (
            f"总体得分: {scores.get('overall', 'N/A')}, "
            f"语法: {scores.get('grammar', 'N/A')}, " # Tampilkan lagi
            f"词汇: {scores.get('vocabulary', 'N/A')}" # Tampilkan lagi
        )
        error_summary = "未发现主要错误。"
        if errors: error_summary = "发现的主要错误:\n" + "".join([f"- {err.get('explanation', 'N/A')}\n" for err in errors[:2]])
        return f"""
您是一位友好且善于鼓励的中文老师。
您的任务是根据学生的作文、得分和错误，写一段简短的评语（2-3句话）。
请用中文书写评语，并在括号()中附上简短的印尼语翻译。
学生作文: "{essay}"
所得分数: {score_summary}
错误备注: {error_summary}
请现在撰写您的评语：
"""

    # --- FUNGSI UTAMA: generate_json (Menggunakan model.chat) ---
    def generate_json(self, essay: str, hsk_level: int = 3) -> str:
        start_time = time.time()
        logger.info(f"Request received (Chain of Prompts - Detailed Scores) for HSK {hsk_level}.")
        if not essay or not essay.strip():
             logger.warning("Empty essay input."); error_result = {"error": "Input essay empty.", "essay": essay, "processing_time": "0.00s"}
             try: return json.dumps(error_result, ensure_ascii=False, indent=2)
             except Exception: return '{"error": "Input essay empty and failed to create error JSON."}'

        # --- LANGKAH 1: ERROR DETECTION ---
        logger.info("Step 1: Detecting Errors (via model.chat)...")
        validated_error_list = []
        try:
            error_prompt = self._build_error_detection_prompt(essay)
            error_response, _ = self.model.chat(self.tokenizer, error_prompt, history=None, system="You are a helpful assistant.")
            logger.debug(f"Step 1 Raw Response: {repr(error_response)}")
            validated_error_list = self._parse_errors_from_text(error_response, essay)
            logger.info(f"Step 1 Done. Found {len(validated_error_list)} errors.")
        except Exception as e: logger.exception("Step 1 (Error Detection) Failed."); validated_error_list = []

        # --- LANGKAH 2: SCORING (Dengan Parser Keyword Asli) ---
        logger.info("Step 2: Getting Detailed Scores (via model.chat)...")
        # Default skor 0
        parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0, "overall": 0}
        try:
            # Menggunakan prompt skor DETAIL
            scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
            scoring_response, _ = self.model.chat(self.tokenizer, scoring_prompt, history=None, system="You are a helpful assistant.")
            logger.info(f"Step 2 Raw Response: {repr(scoring_response)}") # INFO level

            # Menggunakan parser skor ASLI (keyword)
            parsed_scores_data = self._extract_scores_from_text(scoring_response)

            if parsed_scores_data and "score" in parsed_scores_data:
                 parsed_scores = parsed_scores_data["score"] # Ambil dict skor (sudah di-default 0 jika missing)
                 logger.info(f"Step 2 PARSED scores (might be 0): {parsed_scores}")
            else:
                 # Jika parser kembalikan None (gagal total)
                 logger.error("Step 2 PARSING FAILED: Parser returned None. Using default scores (all 0).")
                 # parsed_scores sudah default 0

        except Exception as e:
            logger.exception("Step 2 (Scoring) Failed TOTAL.")
            # parsed_scores sudah default 0

        # Ekstrak skor ke variabel individual (setelah try-except)
        grammar_s = parsed_scores.get("grammar", 0)
        vocab_s = parsed_scores.get("vocabulary", 0)
        coherence_s = parsed_scores.get("coherence", 0)
        cultural_s = parsed_scores.get("cultural_adaptation", 0)
        overall_s = parsed_scores.get("overall", 0)

        # Hitung ulang overall JIKA 0 tapi komponen lain ada nilainya
        if overall_s == 0 and (grammar_s > 0 or vocab_s > 0 or coherence_s > 0 or cultural_s > 0):
            logger.info("Overall score is 0 but detailed scores exist. Recalculating overall based on weights...")
            calc_score = (grammar_s * self.rubric_weights["grammar"]) + \
                         (vocab_s * self.rubric_weights["vocabulary"]) + \
                         (coherence_s * self.rubric_weights["coherence"]) + \
                         (cultural_s * self.rubric_weights["cultural_adaptation"])
            overall_s = max(0, min(100, int(round(calc_score))))
            logger.info(f"Recalculated overall score: {overall_s}")

        logger.info(f"Step 2 Done. Final Scores -> Overall: {overall_s}, Grammar: {grammar_s}, Vocab: {vocab_s}, Coherence: {coherence_s}, Cultural: {cultural_s}")


        # --- LANGKAH 3: FEEDBACK ---
        logger.info("Step 3: Generating Feedback (via model.chat)...")
        feedback = "Gagal menghasilkan feedback."
        try:
            # Feedback prompt sekarang bisa pakai skor detail
            feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
            feedback_response, _ = self.model.chat(self.tokenizer, feedback_prompt, history=None, system="You are a helpful assistant.")
            feedback = feedback_response.strip() if feedback_response else feedback
            if not feedback: # Fallback
                logger.warning("Step 3: model.chat returned empty feedback. Using fallback.")
                if not validated_error_list and overall_s > 80: feedback = "作文写得很好...(Esai baik...)"
                elif validated_error_list: feedback = "作文中发现一些错误...(Ada error...)"
                else: feedback = "请根据得到的总体分数...(Periksa esai...)"
            logger.info("Step 3 Done.")
            logger.debug(f"Step 3 Feedback: {repr(feedback)}")
        except Exception as e:
            logger.exception("Step 3 (Feedback) Failed TOTAL.")
            if validated_error_list: feedback = "作文中发现一些错误...(Ada error...)"
            elif overall_s > 80: feedback = "作文写得很好...(Esai baik...)"
            else: feedback = "请根据得到的总体分数...(Periksa esai...)"

        # --- FINAL ASSEMBLY ---
        final_result = {
            "text": essay, "overall_score": overall_s,
            "detailed_scores": { # Sekarang bisa berisi nilai detail
                "grammar": grammar_s, "vocabulary": vocab_s,
                "coherence": coherence_s, "cultural_adaptation": cultural_s
            },
            "error_list": validated_error_list, "feedback": feedback
        }
        duration = time.time() - start_time
        final_result["processing_time"] = f"{duration:.2f} detik"
        logger.info(f"All steps (Chain of Prompts - Detailed) done. Time: {duration:.2f}s.")

        # --- Robust JSON Dump ---
        try:
            json_output_string = json.dumps(final_result, ensure_ascii=False, indent=2)
            logger.debug(f"JSON generated successfully:\n{json_output_string[:500]}...")
            return json_output_string
        except TypeError as e:
            logger.error(f"FATAL: Failed converting final_result to JSON! Error: {e}", exc_info=True)
            logger.error(f"Data causing error: {final_result}")
            error_json = {"error": "Internal error: Failed to serialize result.", "details": str(e), "essay": essay, "processing_time": f"{duration:.2f}s"}
            try: return json.dumps(error_json, ensure_ascii=False, indent=2)
            except Exception as final_e: logger.critical(f"ULTRA FATAL: Failed dumping error JSON: {final_e}"); return '{"error": "Internal error: Failed serialization."}'
        # --- End Robust JSON Dump ---

# ---------------- Simulasi (Main execution) ----------------
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.warning("="*50 + "\nRUNNING model.py SCRIPT (Chain of Prompts - DETAILED Scores)\n" + "="*50)
    try:
        scorer = QwenScorer()
        logger.info("\n" + "="*20 + " SIMULATION 1: Good Essay " + "="*20)
        essay_1 = "上个星期六，我和朋友去公园玩。我们早上九点起床。我吃早饭，然后穿衣服。朋友开车带我们去公园。公园里有很多人。我们放风筝，吃午饭，然后回家。我玩得很开心。"
        result_1 = scorer.generate_json(essay_1, hsk_level=2)
        print("\n--- SIMULATION 1 RESULT (JSON) ---"); print(result_1); print("---------------------------------\n")
        logger.info("\n" + "="*20 + " SIMULATION 2: Essay Errors " + "="*20)
        essay_2 = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
        result_2 = scorer.generate_json(essay_2, hsk_level=3)
        print("\n--- SIMULATION 2 RESULT (JSON) ---"); print(result_2); print("---------------------------------\n")
        logger.info("Simulations finished.")
    except Exception as e:
        logger.critical(f"Failed to run main simulation: {e}", exc_info=True)
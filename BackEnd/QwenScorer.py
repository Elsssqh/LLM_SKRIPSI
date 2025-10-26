# -*- coding: utf-8 -*-
# FILE: QwenScorer_tuned.py
# Ini adalah skrip INFERENCE yang menggunakan "Soft Prompts" yang telah dilatih
# (Modifikasi dari kode asli Anda)

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import logging
import torch
import torch.nn as nn
import time
# (Helper Jieba, cosine_similarity, dll tetap sama)
# ... (masukkan helper Anda di sini)

# ---------------- Logger ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# ---------------- Helpers ----------------
# (Helper functions cosine_similarity, etc. tetap sama)
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2): return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0: return 0.0
    return dot / (n1 * n2)

# ---------------- QwenScorer (Versi "Prompt Tuned") ----------------

class QwenScorer:
    """
    Implementasi inference menggunakan "Soft Prompts" yang telah dilatih
    """

    def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
        logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Muat Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Muat Base Model (BEKU) untuk INFERENCE
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
        
        # Dapatkan konfigurasi untuk memuat prompt
        self.config = self.base_model.config
        logger.info("Model Qwen-1.8B berhasil dimuat dan diatur ke mode eval.")

        # 3. MUAT PARAMETER SOFT PROMPT YANG SUDAH DILATIH
        # Asumsi Anda sudah melatih 3 prompt terpisah
        try:
            self.error_prompt = self._load_soft_prompt("error_soft_prompt.pt")
            self.scoring_prompt = self._load_soft_prompt("scoring_soft_prompt.pt")
            self.feedback_prompt = self._load_soft_prompt("feedback_soft_prompt.pt")
            logger.info("Berhasil memuat 3 soft prompt yang telah dilatih.")
        except FileNotFoundError as e:
            logger.error(f"Gagal memuat file soft prompt: {e}")
            logger.error("Pastikan Anda sudah menjalankan 'train_prompts.py' untuk setiap tugas.")
            raise
            
        # (Fungsi Jieba dan rubric_weights Anda tetap sama)
        
    def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
       
        try:
            cleaned_essay = re.sub(r'\s+', '', essay).strip()
            if not cleaned_essay:
                 logger.warning("Input esai kosong setelah dibersihkan.")
                 return "", ""
            words_with_pos = list(pseg.cut(cleaned_essay))
            segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
            pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
            logger.debug(f"Jieba Segmented: {segmented}")
            return segmented, pos_lines
        except Exception as e:
            logger.exception("Preprocessing Jieba gagal.")
            return essay, "Jieba preprocessing gagal."
            raise
        try:
            jieba.setLogLevel(logging.INFO)
            jieba.initialize()
            logger.info("Jieba berhasil diinisialisasi.")
        except Exception as e:
            logger.warning(f"Gagal inisialisasi Jieba sepenuhnya: {e}")
            pass
        
    
        self.rubric_weights = {
            "grammar": 0.30,
            "vocabulary": 0.30,
            "coherence": 0.20,
            "cultural_adaptation": 0.20
        }
        logger.info(f"Rubric weights set (untuk output JSON): {self.rubric_weights}")

       
        
    def _load_soft_prompt(self, prompt_path: str) -> nn.Parameter:
        """Helper untuk memuat file state_dict prompt."""
        embed_dim = self.config.hidden_size
        
        # Tentukan panjang prompt dari file state_dict (lebih aman)
        state_dict = torch.load(prompt_path, map_location=self.device)
        # Kunci default adalah 'weight', sesuaikan jika Anda menyimpan dengan nama lain
        prompt_tensor = state_dict[next(iter(state_dict))]
        prompt_length = prompt_tensor.shape[1] # (1, prompt_length, embed_dim)
        
        # Buat parameter dan muat nilainya
        soft_prompt = nn.Parameter(torch.empty(1, prompt_length, embed_dim))
        soft_prompt.load_state_dict(state_dict)
        soft_prompt.to(self.device)
        logger.info(f"Memuat prompt dari {prompt_path} (panjang: {prompt_length})")
        return soft_prompt

    def _get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Helper untuk mendapatkan embedding dari token ID."""
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
             return self.base_model.transformer.wte(input_ids)
        elif hasattr(self.base_model, 'get_input_embeddings'):
             return self.base_model.get_input_embeddings()(input_ids)
        else:
             raise NotImplementedError("Tidak dapat menemukan layer input embedding.")

    def _generate_with_prompt(self, text_input: str, soft_prompt: nn.Parameter) -> str:
        """
        Fungsi helper untuk melakukan 'generate' menggunakan soft prompt.
        Ini menggantikan 'model.chat()' Anda.
        """
        # Tokenize input
        tokenized_input = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask
        
        # Dapatkan embedding dari input asli
        inputs_embeds = self._get_input_embeddings(input_ids)
        batch_size = inputs_embeds.shape[0]

        # 1. GABUNGKAN SOFT PROMPT DENGAN EMBEDDING INPUT 
        prompt_embeds = soft_prompt.expand(batch_size, -1, -1).to(inputs_embeds.device)
        combined_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # 2. BUAT ATTENTION MASK BARU UNTUK PROMPT
        prompt_length = soft_prompt.shape[1]
        prompt_mask = torch.ones(batch_size, prompt_length, dtype=attention_mask.dtype).to(attention_mask.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 3. JALANKAN GENERATE HANYA DENGAN EMBEDDING
        # Kita menggunakan model yang BEKU [cite: 68]
        outputs = self.base_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=256, # Sesuaikan
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 4. Decode output
        # Kita perlu memotong bagian input & prompt dari output
        # Panjang total input (prompt + text) dalam token
        input_token_len = combined_mask.shape[1]
        generated_ids = outputs[0, input_token_len:]
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # --- FUNGSI PARSING ANDA (TETAP SAMA) ---
    # _build_..._prompt functions DIHAPUS karena kita tidak lagi menggunakan teks prompt
    
    def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
        """
        Mem-parsing output teks dari _build_error_detection_prompt.
        (Fungsi ini tidak berubah, karena 'TIDAK ADA KESALAHAN' dan '|' bersifat universal)
        """
        validated_error_list = []
        # Keyword 'TIDAK ADA KESALAHAN' sengaja tidak diterjemahkan agar unik
        if "TIDAK ADA KESALAHAN" in error_response or error_response.strip() == "":
            return []
        
        # Pola regex untuk menangkap 4 bagian yang dipisahkan oleh '|'
        pattern = re.compile(r"(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+)")
        
        for line in error_response.split('\n'):
            line = line.strip()
            match = pattern.search(line)
            
            if match:
                try:
                    err_type = match.group(1).strip()
                    incorrect_frag = match.group(2).strip()
                    correction = match.group(3).strip()
                    explanation = match.group(4).strip()
                    
                    start_index = essay_text.find(incorrect_frag)
                    if start_index != -1:
                        end_index = start_index + len(incorrect_frag)
                        pos = [start_index, end_index]
                    else:
                        logger.warning(f"Tidak dapat menemukan posisi untuk fragmen: '{incorrect_frag}'. Menggunakan posisi default [0, 0].")
                        pos = [0, 0]

                    validated_error_list.append({
                        "error_type": err_type,
                        "error_position": pos,
                        "incorrect_fragment": incorrect_frag,
                        "suggested_correction": correction,
                        "explanation": explanation
                    })
                except Exception as e:
                    logger.warning(f"Gagal mem-parsing baris error: '{line}'. Error: {e}")
                    
        return validated_error_list

    # --- FUNGSI UTAMA: GENERATE_JSON (VERSI MODIFIKASI) ---
    def generate_json(self, essay: str, hsk_level: int = 3) -> str:
        """
        Fungsi utama untuk menilai esai menggunakan "soft prompts" yang telah dilatih.
        """
        start_time = time.time()
        logger.info(f"Menerima permintaan (generate_json) 'Prompt Tuned'")

        if not essay or not essay.strip():
            # ... (Logika error Anda tetap sama)
            pass

        # --- LANGKAH 1: DETEKSI KESALAHAN (MENGGUNAKAN SOFT PROMPT 1) ---
        logger.info("Memulai Langkah 1: Mendeteksi Kesalahan (dengan error_prompt.pt)...")
        validated_error_list = []
        try:
            # Input untuk prompt eror HANYA esai
            error_response = self._generate_with_prompt(essay, self.error_prompt)
            logger.debug(f"Langkah 1 (Raw Response): {error_response}")
            validated_error_list = self._parse_errors_from_text(error_response, essay)
            logger.info(f"Langkah 1 Selesai. Ditemukan {len(validated_error_list)} kesalahan.")
        except Exception as e:
            logger.exception("Langkah 1 (Deteksi Kesalahan) Gagal.")
            validated_error_list = []

        # --- LANGKAH 2: PENILAIAN (MENGGUNAKAN SOFT PROMPT 2) ---
        logger.info("Memulai Langkah 2: Memberikan Skor (dengan scoring_prompt.pt)...")
        parsed_scores = {}
        try:
            # Input untuk prompt skor (sesuai data latih di Bagian 1)
            scoring_input = f"HSK: {hsk_level} Esai: {essay}"
            scoring_response = self._generate_with_prompt(scoring_input, self.scoring_prompt)
            logger.debug(f"Langkah 2 (Raw Response): {scoring_response}")
            
            parsed_scores_data = self._extract_scores_from_text(scoring_response)
            # ... (Sisa logika parsing skor Anda tetap sama)
            # ...
            logger.info(f"Langkah 2 Selesai. Skor diterima.")
        except Exception as e:
            logger.exception("Langkah 2 (Penilaian) Gagal Total.")
            parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "task_fulfillment": 0, "overall": 0}

        # --- LANGKAH 3: UMPAN BALIK (MENGGUNAKAN SOFT PROMPT 3) ---
        logger.info("Memulai Langkah 3: Menghasilkan Feedback (dengan feedback_prompt.pt)...")
        feedback = "Gagal menghasilkan feedback."
        try:
            # Susun input untuk prompt feedback (sesuai data latih Anda)
            score_summary = f"Skor: {parsed_scores.get('overall', 'N/A')}"
            error_summary = f"Kesalahan: {len(validated_error_list)}"
            feedback_input = f"Esai: {essay}\n{score_summary}\n{error_summary}"

            feedback_response = self._generate_with_prompt(feedback_input, self.feedback_prompt)
            feedback = feedback_response.strip()
            # ... (Logika fallback Anda tetap sama)
            # ...
            logger.info("Langkah 3 Selesai.")
        except Exception as e:
            logger.exception("Langkah 3 (Feedback) Gagal.")
            # ... (Logika fallback Anda tetap sama)
            # ...

        # --- FINAL: PERAKITAN JSON ---
        # ... (Logika perakitan JSON Anda tetap sama)
        final_result = {
             "text": essay,
             "overall_score": parsed_scores.get('overall', 0),
             # ... etc
        }
        # ...
        logger.info(f"Semua langkah 'Prompt Tuned' selesai.")
        return json.dumps(final_result, ensure_ascii=False, indent=2)


# # ---------------- Simulasi (Main execution) ----------------
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.warning("PASTIKAN ANDA SUDAH MELATIH 3 PROMPT:")
    logger.warning("1. error_soft_prompt.pt")
    logger.warning("2. scoring_soft_prompt.pt")
    logger.warning("3. feedback_soft_prompt.pt")
    logger.warning("menggunakan 'train_prompts.py' sebelum menjalankan ini.")
    
    # try:
    #     scorer = QwenScorer()
    #     
    #     essay_errors = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
    #     result_json = scorer.generate_json(essay_errors, hsk_level=3)
    #     print("\n--- HASIL SIMULASI 'PROMPT TUNED' ---")
    #     print(result_json)
    #     print("---------------------------------\n")
    #
    # except Exception as e:
    #     logger.critical(f"Gagal menjalankan program utama: {e}", exc_info=True)
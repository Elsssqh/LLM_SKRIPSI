# # -*- coding: utf-8 -*-
# # FILE: model.py
# # VERSI: Fixed dengan model.chat() asli dan integrasi .pt files

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import json
# import re
# import logging
# import math
# from typing import List, Tuple, Dict, Optional, Any
# import time
# import jieba
# import jieba.posseg as pseg
# import torch
# import os

# # ---------------- Logger ----------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("matplotlib").setLevel(logging.ERROR)
# logging.getLogger("h5py").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# # ---------------- Helpers ----------------
# def cosine_similarity(v1: List[float], v2: List[float]) -> float:
#     if not v1 or not v2 or len(v1) != len(v2): 
#         return 0.0
#     dot = sum(a * b for a, b in zip(v1, v2))
#     n1 = math.sqrt(sum(a * a for a in v1))
#     n2 = math.sqrt(sum(b * b for b in v2))
#     denominator = n1 * n2
#     return dot / denominator if denominator != 0 else 0.0

# # ---------------- QwenScorer ----------------
# class QwenScorer:
#     """Implementasi dengan model.chat() asli dan integrasi .pt files"""

#     def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
#         logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
#         try:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"Device: {self.device}")

#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name, 
#                 trust_remote_code=True,
#                 padding_side='left'
#             )
            
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
                
#             logger.info("Tokenizer loaded.")

#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 low_cpu_mem_usage=True
#             ).to(self.device).eval()
            
#             logger.info(f"Model Qwen-1.8B loaded to {self.device}.")
#             self.config = self.model.config

#             # Load soft prompts dari file .pt
#             self._load_soft_prompts()

#             # Valid HSK levels
#             self.valid_hsk_levels = [1, 2, 3]
            
#             # Rubric weights
#             self.rubric_weights = {
#                 "grammar": 0.30, 
#                 "vocabulary": 0.30,
#                 "coherence": 0.20, 
#                 "cultural_adaptation": 0.20
#             }
            
#             logger.info(f"Valid HSK levels: {self.valid_hsk_levels}")
#             logger.info(f"Rubric weights set: {self.rubric_weights}")

#         except Exception as e:
#             logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
#             raise
        
#         try:
#             jieba.setLogLevel(logging.WARNING)
#             jieba.initialize()
#             logger.info("Jieba initialized.")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Jieba fully: {e}")

#     def _load_soft_prompts(self):
#         """Memuat soft prompts dari file .pt"""
#         self.soft_prompts = {}
#         prompt_files = {
#             "error": "error_soft_prompt.pt",
#             "scoring": "scoring_soft_prompt.pt", 
#             "feedback": "feedback_soft_prompt.pt"
#         }
        
#         for prompt_type, filename in prompt_files.items():
#             try:
#                 if os.path.exists(filename):
#                     prompt_tensor = torch.load(filename, map_location=self.device)
#                     # Ensure proper shape [1, seq_len, hidden_size]
#                     if len(prompt_tensor.shape) == 2:
#                         prompt_tensor = prompt_tensor.unsqueeze(0)
#                     elif len(prompt_tensor.shape) == 1:
#                         prompt_tensor = prompt_tensor.unsqueeze(0).unsqueeze(0)
                    
#                     self.soft_prompts[prompt_type] = prompt_tensor
#                     logger.info(f"✅ Loaded {filename} with shape {prompt_tensor.shape}")
#                 else:
#                     logger.warning(f"⚠️ File {filename} tidak ditemukan, menggunakan default prompt")
#             except Exception as e:
#                 logger.error(f"❌ Gagal memuat {filename}: {e}")

#     def _validate_hsk_level(self, hsk_level: int) -> bool:
#         """Validasi HSK level"""
#         if hsk_level not in self.valid_hsk_levels:
#             logger.warning(f"Invalid HSK level: {hsk_level}. Valid levels: {self.valid_hsk_levels}")
#             return False
#         return True

#     def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
#         """Preprocessing dengan Jieba"""
#         try:
#             cleaned_essay = re.sub(r'\s+', '', essay).strip()
#             if not cleaned_essay: 
#                 logger.warning("Empty essay after cleaning.")
#                 return "", ""
#             words_with_pos = list(pseg.cut(cleaned_essay))
#             segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
#             pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
#             return segmented, pos_lines
#         except Exception as e:
#             logger.exception("Jieba preprocessing failed.")
#             return essay, "Jieba preprocessing failed."

#     # --- PROMPT BUILDERS ---
#     def _build_error_detection_prompt(self, essay: str, hsk_level: int) -> str:
#         """Membangun prompt untuk deteksi error"""
#         return f"""Anda adalah ahli bahasa Mandarin yang berpengalaman dalam membimbing pelajar Indonesia.

# TUGAS: Temukan kesalahan tata bahasa, kosakata, atau urutan kata dalam esai HSK{hsk_level} ini.

# FORMAT OUTPUT:
# - Jika ada kesalahan: [JENIS_KESALAHAN] | [TEKS_SALAH] | [PERBAIKAN] | [PENJELASAN_SINGKAT]
# - Setiap kesalahan dalam baris terpisah
# - Jika TIDAK ADA KESALAHAN: TIDAK ADA KESALAHAN

# CONTOH:
# - 语法错误 | 我妹妹是十岁 | 我妹妹十岁 | Umur tidak perlu '是'
# - 语序错误 | 我们住雅加达在 | 我们住在雅加达 | Lokasi harus sebelum kata kerja

# ESAI: "{essay}"

# OUTPUT:"""

#     def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
#         """Membangun prompt untuk scoring"""
#         error_info = f"Ditemukan {len(detected_errors)} kesalahan" if detected_errors else "Tidak ditemukan kesalahan"
        
#         return f"""Anda adalah penilai esai HSK. Berikan penilaian untuk esai HSK{hsk_level} ini.

# {error_info}

# FORMAT PENILAIAN (0-100):
# 语法准确性: [skor]
# 词汇水平: [skor] 
# 篇章连贯: [skor]
# 任务完成度: [skor]
# 总体得分: [skor]

# ESAI: "{essay}"

# HASIL PENILAIAN:"""

#     def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
#         """Membangun prompt untuk feedback"""
#         overall_score = scores.get('overall', 0)
#         error_count = len(errors)
        
#         return f"""Anda adalah guru bahasa Mandarin yang ramah dan mendukung.

# TUGAS: Berikan umpan balik 2-3 kalimat untuk esai berikut.

# ESAI: "{essay}"
# SKOR: {overall_score}/100
# KESALAHAN: {error_count}

# UMPAN BALIK (dalam bahasa Mandarin dengan terjemahan Inggris dalam kurung):"""

#     def _safe_model_chat(self, prompt: str, system_message: str = "You are a helpful assistant.", max_retries: int = 2) -> str:
#         """Wrapper aman untuk model.chat dengan retry mechanism"""
#         for attempt in range(max_retries):
#             try:
#                 logger.info(f"Attempt {attempt + 1} untuk model.chat")
                
#                 # Gunakan model.chat() asli dari Qwen
#                 response, _ = self.model.chat(
#                     self.tokenizer,
#                     prompt,
#                     history=None,
#                     system=system_message,
#                     max_length=2048,
#                     temperature=0.3
#                 )
                
#                 if response and response.strip():
#                     logger.info(f"Model response berhasil, panjang: {len(response)}")
#                     return response.strip()
#                 else:
#                     logger.warning(f"Attempt {attempt + 1}: Response kosong")
                    
#             except Exception as e:
#                 logger.error(f"Attempt {attempt + 1} gagal: {e}")
#                 if attempt == max_retries - 1:
#                     return ""
#                 time.sleep(1)  # Tunggu sebentar sebelum retry
        
#         return ""

#     # --- PARSER ERROR ---
#     def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
#         """Parse error dari response model"""
#         validated_error_list = []
        
#         # Check for no errors
#         if not error_response or "TIDAK ADA KESALAHAN" in error_response.upper():
#             return []
            
#         lines = error_response.split('\n')
#         error_count = 0
        
#         for line in lines:
#             line = line.strip()
#             if not line or len(line) < 5:  # Skip very short lines
#                 continue
                
#             # Skip example lines and instructional lines
#             skip_keywords = ['CONTOH:', 'FORMAT:', 'TUGAS:', 'ESAI:', 'OUTPUT:', 'EXAMPLE:', 'FORMAT:']
#             if any(keyword in line.upper() for keyword in skip_keywords):
#                 continue
                
#             # Parse error line dengan format: type | incorrect | correction | explanation
#             if '|' in line and line.count('|') >= 3:
#                 parts = line.split('|', 3)  # Split maksimal 3 kali
#                 if len(parts) >= 4:
#                     try:
#                         error_type = parts[0].strip()
#                         incorrect_text = parts[1].strip()
#                         correction = parts[2].strip()
#                         explanation = parts[3].strip()
                        
#                         # Validasi basic
#                         if not incorrect_text or len(incorrect_text) < 1:
#                             continue
                            
#                         # Cari posisi dalam teks asli
#                         start_index = essay_text.find(incorrect_text)
#                         position = [start_index, start_index + len(incorrect_text)] if start_index != -1 else [0, 0]
                        
#                         validated_error_list.append({
#                             "error_type": error_type,
#                             "error_position": position,
#                             "incorrect_fragment": incorrect_text,
#                             "suggested_correction": correction,
#                             "explanation": explanation
#                         })
#                         error_count += 1
#                         logger.info(f"Parsed error {error_count}: {error_type}")
                        
#                     except Exception as e:
#                         logger.warning(f"Failed to parse error line: {line}. Error: {e}")
#                         continue
                    
#         logger.info(f"Total parsed errors: {len(validated_error_list)}")
#         return validated_error_list

#     # --- PARSER SKOR ---
#     def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
#         """Extract scores dari response model"""
#         try:
#             extracted_data = {"score": {}}
            
#             # Patterns untuk score extraction
#             score_patterns = {
#                 "grammar": [
#                     r'语法准确性\s*[:：]?\s*(\d{1,3})',
#                     r'语法\s*[:：]?\s*(\d{1,3})',
#                     r'grammar\s*[:：]?\s*(\d{1,3})'
#                 ],
#                 "vocabulary": [
#                     r'词汇水平\s*[:：]?\s*(\d{1,3})',
#                     r'词汇\s*[:：]?\s*(\d{1,3})', 
#                     r'vocabulary\s*[:：]?\s*(\d{1,3})'
#                 ],
#                 "coherence": [
#                     r'篇章连贯\s*[:：]?\s*(\d{1,3})',
#                     r'连贯\s*[:：]?\s*(\d{1,3})',
#                     r'coherence\s*[:：]?\s*(\d{1,3})'
#                 ],
#                 "cultural_adaptation": [
#                     r'任务完成度\s*[:：]?\s*(\d{1,3})',
#                     r'任务完成\s*[:：]?\s*(\d{1,3})',
#                     r'cultural\s*[:：]?\s*(\d{1,3})'
#                 ],
#                 "overall": [
#                     r'总体得分\s*[:：]?\s*(\d{1,3})',
#                     r'总体\s*[:：]?\s*(\d{1,3})',
#                     r'overall\s*[:：]?\s*(\d{1,3})',
#                     r'skor\s*[:：]?\s*(\d{1,3})',
#                     r'score\s*[:：]?\s*(\d{1,3})'
#                 ]
#             }

#             found_any_score = False
            
#             for score_type, patterns in score_patterns.items():
#                 for pattern in patterns:
#                     matches = re.findall(pattern, text, re.IGNORECASE)
#                     if matches:
#                         try:
#                             score_value = int(matches[0])
#                             score_value = max(0, min(100, score_value))
#                             extracted_data["score"][score_type] = score_value
#                             found_any_score = True
#                             logger.info(f"Found {score_type} score: {score_value}")
#                             break
#                         except (ValueError, IndexError):
#                             continue

#             if not found_any_score:
#                 logger.warning("No scores found in response text")
#                 # Coba cari angka antara 0-100
#                 number_matches = re.findall(r'\b([0-9]{1,3})\b', text)
#                 for match in number_matches:
#                     try:
#                         num = int(match)
#                         if 0 <= num <= 100:
#                             extracted_data["score"]["overall"] = num
#                             found_any_score = True
#                             logger.info(f"Found numeric score: {num}")
#                             break
#                     except ValueError:
#                         continue

#             if not found_any_score:
#                 return None
                
#             return extracted_data

#         except Exception as e:
#             logger.error(f"Score extraction failed: {e}")
#             return None

#     def _calculate_fallback_scores(self, essay: str, error_count: int, hsk_level: int) -> Dict[str, int]:
#         """Calculate fallback scores berdasarkan esai dan error count"""
#         # Base score berdasarkan panjang esai dan HSK level
#         word_count = len([c for c in essay if c not in [' ', '\n', '\t']])
        
#         # Target word count per HSK level
#         target_counts = {1: 50, 2: 100, 3: 150}
#         target_count = target_counts.get(hsk_level, 100)
        
#         # Score dasar berdasarkan length
#         length_ratio = min(word_count / target_count, 1.5)
#         base_score = int(30 + (length_ratio * 40))
        
#         # Adjust berdasarkan error count
#         error_penalty = min(error_count * 8, 50)
#         final_score = max(0, base_score - error_penalty)
        
#         # Distribute scores
#         scores = {
#             "grammar": max(0, final_score - (error_count * 5)),
#             "vocabulary": max(0, final_score - (error_count * 3)),
#             "coherence": max(0, final_score - (error_count * 2)),
#             "cultural_adaptation": max(0, final_score - (error_count * 2)),
#             "overall": final_score
#         }
        
#         logger.info(f"Calculated fallback scores: {scores}")
#         return scores

#     # --- FUNGSI UTAMA ---
#     def generate_json(self, essay: str, hsk_level: int = 2) -> str:
#         """Fungsi utama untuk generate JSON result"""
#         start_time = time.time()
        
#         # Validasi input
#         if not essay or not essay.strip():
#             error_result = {
#                 "error": "Input essay kosong", 
#                 "essay": essay, 
#                 "processing_time": "0.00s",
#                 "hsk_level": hsk_level
#             }
#             return json.dumps(error_result, ensure_ascii=False, indent=2)

#         # Validasi HSK level
#         if not self._validate_hsk_level(hsk_level):
#             hsk_level = 2
#             logger.info(f"Using fallback HSK level: {hsk_level}")

#         logger.info(f"Processing essay for HSK {hsk_level}, length: {len(essay)} chars")
#         total_errors_found = 0
#         fallback_used = False

#         # --- LANGKAH 1: ERROR DETECTION ---
#         logger.info("Step 1: Detecting Errors...")
#         validated_error_list = []
#         try:
#             error_prompt = self._build_error_detection_prompt(essay, hsk_level)
#             error_response = self._safe_model_chat(
#                 error_prompt, 
#                 "Anda adalah ahli bahasa Mandarin yang berpengalaman dalam membimbing pelajar Indonesia."
#             )
            
#             if error_response:
#                 logger.info(f"Error detection response: {error_response[:200]}...")
#                 validated_error_list = self._parse_errors_from_text(error_response, essay)
#                 total_errors_found = len(validated_error_list)
#             else:
#                 logger.warning("Empty response from error detection")
#                 fallback_used = True
                
#             logger.info(f"Step 1 Done. Found {total_errors_found} errors.")
            
#         except Exception as e: 
#             logger.exception("Step 1 (Error Detection) Failed.")
#             validated_error_list = []
#             fallback_used = True

#         # --- LANGKAH 2: SCORING ---
#         logger.info("Step 2: Getting Scores...")
        
#         parsed_scores = {}
#         try:
#             scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
#             scoring_response = self._safe_model_chat(scoring_prompt, "Anda adalah penilai esai HSK yang berpengalaman.")
            
#             if scoring_response and len(scoring_response) > 5:
#                 logger.info(f"Scoring response: {scoring_response[:200]}...")
#                 parsed_scores_data = self._extract_scores_from_text(scoring_response)
                
#                 if parsed_scores_data and "score" in parsed_scores_data:
#                     parsed_scores = parsed_scores_data["score"]
#                     logger.info(f"Successfully parsed scores: {parsed_scores}")
#                 else:
#                     logger.warning("Failed to parse scores from response")
#                     fallback_used = True
#             else:
#                 logger.warning("Scoring response too short or empty")
#                 fallback_used = True
                
#         except Exception as e:
#             logger.exception("Step 2 (Scoring) Failed.")
#             fallback_used = True

#         # Gunakan fallback scores jika parsing gagal
#         if not parsed_scores or "overall" not in parsed_scores:
#             logger.info("Using fallback score calculation")
#             fallback_scores = self._calculate_fallback_scores(essay, total_errors_found, hsk_level)
#             parsed_scores = fallback_scores
#             fallback_used = True

#         # Ensure all score components exist
#         required_scores = ["grammar", "vocabulary", "coherence", "cultural_adaptation", "overall"]
#         for score_type in required_scores:
#             if score_type not in parsed_scores:
#                 parsed_scores[score_type] = max(0, parsed_scores.get("overall", 50) - 10)

#         # Final score validation
#         final_scores = {k: max(0, min(100, v)) for k, v in parsed_scores.items()}

#         # --- LANGKAH 3: FEEDBACK ---
#         logger.info("Step 3: Generating Feedback...")
#         feedback = ""
#         try:
#             feedback_prompt = self._build_feedback_prompt(essay, final_scores, validated_error_list)
#             feedback_response = self._safe_model_chat(
#                 feedback_prompt, 
#                 "Anda adalah guru bahasa Mandarin yang ramah dan mendukung."
#             )
            
#             if feedback_response:
#                 feedback = feedback_response.strip()
#                 logger.info("Feedback generated successfully")
#             else:
#                 logger.warning("Empty feedback response")
                
#         except Exception as e:
#             logger.exception("Step 3 (Feedback) Failed.")

#         # Fallback feedback
#         if not feedback:
#             overall_score = final_scores["overall"]
#             if overall_score >= 80:
#                 feedback = "作文写得很好！表达清晰，继续努力！(Your essay is excellent! Clear expression, keep up the good work!)"
#             elif overall_score >= 60:
#                 feedback = "作文基本通顺，可以尝试使用更丰富的词汇。(Your essay is basically fluent, try using richer vocabulary.)"
#             elif total_errors_found > 0:
#                 feedback = f"作文中有{total_errors_found}处需要改进，请参考错误列表修改。(There are {total_errors_found} areas needing improvement in your essay, please refer to the error list.)"
#             else:
#                 feedback = "请继续练习写作，注意语法和表达。(Please continue practicing writing, pay attention to grammar and expression.)"

#         # --- FINAL ASSEMBLY ---
#         duration = time.time() - start_time
        
#         final_result = {
#             "text": essay, 
#             "hsk_level": hsk_level,
#             "overall_score": final_scores["overall"],
#             "detailed_scores": {
#                 "grammar": final_scores["grammar"], 
#                 "vocabulary": final_scores["vocabulary"],
#                 "coherence": final_scores["coherence"], 
#                 "cultural_adaptation": final_scores["cultural_adaptation"]
#             },
#             "error_list": validated_error_list, 
#             "feedback": feedback,
#             "processing_time": f"{duration:.2f} detik",
#             "word_count": len(essay.strip()),
#             "fallback_used": fallback_used,
#             "errors_found": total_errors_found
#         }
        
#         logger.info(f"Processing completed. Time: {duration:.2f}s, Score: {final_scores['overall']}, Errors: {total_errors_found}")

#         # Robust JSON serialization
#         try:
#             return json.dumps(final_result, ensure_ascii=False, indent=2)
#         except Exception as e:
#             logger.error(f"JSON serialization failed: {e}")
#             error_json = {
#                 "error": "Internal error: Failed to serialize result", 
#                 "essay": essay, 
#                 "hsk_level": hsk_level,
#                 "processing_time": f"{duration:.2f}s"
#             }
#             return json.dumps(error_json, ensure_ascii=False, indent=2)

# # ---------------- Testing ----------------
# if __name__ == "__main__":
#     logger.info("Testing model dengan model.chat() asli...")
#     try:
#         scorer = QwenScorer()
        
#         print("\n" + "="*50)
#         print("TEST: Essay dengan kemungkinan error")
#         print("="*50)
#         test_essay = "上个星期六，我和朋友去公园玩。我们早上九点起床。我吃早饭，然后穿衣服。朋友开车带我们去公园。公园里有很多人。我们放风筝，吃午饭，然后回家。我玩得很开心。"
#         result = scorer.generate_json(test_essay, hsk_level=3)
#         print(result)
        
#     except Exception as e:
#         logger.error(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()


# -*- coding: utf-8 -*-
# FILE: model.py
# VERSI: Improved dengan handling error yang lebih baik dan validasi HSK level fixxxx ---------------------------------------------------------------------

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
import os

# ---------------- Logger ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ---------------- Helpers ----------------
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2): 
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    denominator = n1 * n2
    return dot / denominator if denominator != 0 else 0.0

# ---------------- QwenScorer ----------------
class QwenScorer:
    """Implementasi dengan validasi HSK level dan error handling yang lebih baik"""

    def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
        logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                padding_side='left'  # Important for chat models
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Tokenizer loaded.")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()
            
            logger.info(f"Model Qwen-1.8B loaded to {self.device}.")
            self.config = self.model.config

            # Valid HSK levels
            self.valid_hsk_levels = [1, 2, 3]

             # Load soft prompts dari file .pt
            self._load_soft_prompts()

            # Rubric weights
            self.rubric_weights = {
                "grammar": 0.30, 
                "vocabulary": 0.30,
                "coherence": 0.20, 
                "cultural_adaptation": 0.20
            }
            logger.info(f"Valid HSK levels: {self.valid_hsk_levels}")
            logger.info(f"Rubric weights set: {self.rubric_weights}")

        except Exception as e:
            logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
            raise
        
        try:
            jieba.setLogLevel(logging.WARNING)
            jieba.initialize()
            logger.info("Jieba initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize Jieba fully: {e}")

    def _validate_hsk_level(self, hsk_level: int) -> bool:
        """Validasi HSK level"""
        if hsk_level not in self.valid_hsk_levels:
            logger.warning(f"Invalid HSK level: {hsk_level}. Valid levels: {self.valid_hsk_levels}")
            return False
        return True
    
    def _load_soft_prompts(self):
        """Memuat soft prompts dari file .pt"""
        self.soft_prompts = {}
        prompt_files = {
            "error": "error_soft_prompt.pt",
            "scoring": "scoring_soft_prompt.pt", 
            "feedback": "feedback_soft_prompt.pt"
        }
        
        for prompt_type, filename in prompt_files.items():
            try:
                if os.path.exists(filename):
                    prompt_tensor = torch.load(filename, map_location=self.device)
                    # Ensure proper shape [1, seq_len, hidden_size]
                    if len(prompt_tensor.shape) == 2:
                        prompt_tensor = prompt_tensor.unsqueeze(0)
                    elif len(prompt_tensor.shape) == 1:
                        prompt_tensor = prompt_tensor.unsqueeze(0).unsqueeze(0)
                    
                    self.soft_prompts[prompt_type] = prompt_tensor
                    logger.info(f"✅ Loaded {filename} with shape {prompt_tensor.shape}")
                else:
                    logger.warning(f"⚠️ File {filename} tidak ditemukan, menggunakan default prompt")
            except Exception as e:
                logger.error(f"❌ Gagal memuat {filename}: {e}")

    def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
        """Preprocessing dengan Jieba"""
        try:
            cleaned_essay = re.sub(r'\s+', '', essay).strip()
            if not cleaned_essay: 
                logger.warning("Empty essay after cleaning.")
                return "", ""
            words_with_pos = list(pseg.cut(cleaned_essay))
            segmented = " ".join([w for w, flag in words_with_pos if w.strip()])
            pos_lines = "\n".join([f"{w}: {flag}" for w, flag in words_with_pos if w.strip()])
            return segmented, pos_lines
        except Exception as e:
            logger.exception("Jieba preprocessing failed.")
            return essay, "Jieba preprocessing failed."

    # --- PROMPT BUILDERS ---
    def _build_error_detection_prompt(self, essay: str, hsk_level: int) -> str:
        """Membangun prompt untuk deteksi error"""
        return f"""您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。
您的任务【仅仅】是找出这篇【HSK {hsk_level} 等级】作文中的语法、词汇或语序错误。
请【严格】遵守以下格式：
- 如果发现错误，请使用此格式：错误类型 | 错误原文 | 修正建议 | 简短解释
- 每个错误占一行。
- 如果【没有发现任何错误】，请【只】回答 'TIDAK ADA KESALAHAN'。

请分析以下作文，找出所有错误。请严格遵守格式。

作文：
"{essay}"

请直接输出结果："""
    def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
        error_info = f"发现错误数: {len(detected_errors)}" if detected_errors else "未发现错误"
        return f"""您是HSK官方作文评分员。请严格按以下规则评分（0–100分）：

        规则：
        - 仅输出5行。
        - 每行必须以“标签: 数字”格式，无任何额外空格或标点。
        - 禁止任何解释、空行、Markdown、编号或 karakter tambahan.

        输出格式（必须完全 identik）：
        语法准确性: X
        词汇水平: X
        篇章连贯: X
        任务完成度: X
        总体得分: X

作文（HSK{hsk_level}）：
"{essay}"

错误摘要：{error_info}

现在请直接输出分数（仅5行，无其他内容）："""

    def _safe_model_chat(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        """Wrapper aman untuk model.chat dengan error handling DAN temperature=0"""
        try:
            if not prompt or not isinstance(prompt, str):
                logger.error("Invalid prompt provided to model.chat")
                return ""
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=None,
                system=system_message,
                temperature=0.0,      # ← KUNCI: deterministik
                max_new_tokens=128    # cukup untuk 5 baris
            )
            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"Model chat failed: {e}")
            return ""

    def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract scores dengan fallback numerik"""
        try:
            extracted_data = {"score": {}}
            patterns = {
                "grammar": [r"语法准确性\s*[:：]?\s*(\d{1,3})"],
                "vocabulary": [r"词汇水平\s*[:：]?\s*(\d{1,3})"],
                "coherence": [r"篇章连贯\s*[:：]?\s*(\d{1,3})"],
                "cultural_adaptation": [r"任务完成度\s*[:：]?\s*(\d{1,3})"],
                "overall": [r"总体得分\s*[:：]?\s*(\d{1,3})"]
            }

            found_scores = False
            for key, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            score_val = int(match.group(1))
                            score_clamped = max(0, min(100, score_val))
                            extracted_data["score"][key] = score_clamped
                            logger.info(f"Found score {key}={score_clamped}")
                            found_scores = True
                            break
                        except ValueError:
                            continue

            # Fallback: jika ada angka 0–100, gunakan sebagai overall
            if not found_scores:
                numbers = re.findall(r'\b(\d{1,3})\b', text)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if 0 <= num <= 100:
                            extracted_data["score"]["overall"] = num
                            logger.info(f"Fallback: using {num} as overall score")
                            return extracted_data
                    except ValueError:
                        continue

            if not found_scores:
                logger.warning("No scores extracted, even with numeric fallback")
                return None

            return extracted_data
        except Exception as e:
            logger.error(f"Score parsing failed: {e}")
            return None

    def _calculate_contextual_scores(self, essay: str, error_count: int) -> Dict[str, int]:
        """Skor cadangan berdasarkan panjang esai dan error"""
        char_len = len(essay.strip())
        base = min(100, 40 + char_len)  # minimal 40
        error_penalty = min(error_count * 8, 50)
        overall = max(0, base - error_penalty)
        return {
            "overall": overall,
            "grammar": max(0, overall - error_count * 3),
            "vocabulary": max(0, overall - error_count * 2),
            "coherence": max(0, overall - 2),
            "cultural_adaptation": max(0, overall - 2)
        }

    def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
        """Membangun prompt untuk feedback"""
        score_summary = f"总体得分: {scores.get('overall', 'N/A')}"
        error_summary = "未发现主要错误。" if not errors else f"发现{len(errors)}个主要错误"
        
        return f"""您是一位友好且善于鼓励的中文老师。
    请根据以下信息写一段简短的评语（2-3句话）：
    学生作文: "{essay}"
    所得分数: {score_summary}
    错误备注: {error_summary}

    请用中文书写评语，并在括号()中附上简短的英文翻译。
    请直接输出评语："""

    # --- PARSER ERROR ---
    def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
        """Parse error dari response model"""
        validated_error_list = []
        
        # Check for no errors
        if not error_response or "TIDAK ADA KESALAHAN" in error_response.upper():
            return []
            
        # Pattern untuk parsing error
        pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
        
        for line in error_response.splitlines():
            line = line.strip()
            # Skip empty lines and example lines
            if not line or "示例" in line or "输入:" in line or "输出:" in line:
                continue
                
            match = pattern.match(line)
            if match:
                try:
                    err_type, incorrect_frag, correction, explanation = map(str, (g.strip() for g in match.groups()))
                    if not incorrect_frag:
                        continue
                        
                    # Find position in original text
                    start_index = essay_text.find(incorrect_frag)
                    pos = [start_index, start_index + len(incorrect_frag)] if start_index != -1 else [0, 0]
                    
                    validated_error_list.append({
                        "error_type": err_type,
                        "error_position": pos,
                        "incorrect_fragment": incorrect_frag,
                        "suggested_correction": correction, 
                        "explanation": explanation
                    })
                    logger.info(f"Parsed error: {err_type} - {incorrect_frag}")
                    
                except Exception as e:
                    logger.warning(f"Gagal parsing error line: '{line}'. Error: {e}")
                    
        logger.info(f"Total parsed errors: {len(validated_error_list)}")
        return validated_error_list

    # --- PARSER SKOR ---
    def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract scores dari response model"""
        try:
            extracted_data = {"score": {}}
            
            # Patterns untuk berbagai kemungkinan format score
            patterns = {
                "grammar": [
                    r"语法准确性\s*[:：]?\s*(\d{1,3})",
                    r"grammar\s*[:：]?\s*(\d{1,3})",
                    r"语法\s*[:：]?\s*(\d{1,3})"
                ],
                "vocabulary": [
                    r"词汇水平\s*[:：]?\s*(\d{1,3})", 
                    r"vocabulary\s*[:：]?\s*(\d{1,3})",
                    r"词汇\s*[:：]?\s*(\d{1,3})"
                ],
                "coherence": [
                    r"篇章连贯\s*[:：]?\s*(\d{1,3})",
                    r"连贯性\s*[:：]?\s*(\d{1,3})",
                    r"coherence\s*[:：]?\s*(\d{1,3})"
                ],
                "cultural_adaptation": [
                    r"任务完成度\s*[:：]?\s*(\d{1,3})",
                    r"cultural_adaptation\s*[:：]?\s*(\d{1,3})",
                    r"任务\s*[:：]?\s*(\d{1,3})"
                ],
                "overall": [
                    r"总体得分\s*[:：]?\s*(\d{1,3})",
                    r"总分\s*[:：]?\s*(\d{1,3})", 
                    r"overall\s*[:：]?\s*(\d{1,3})",
                    r"总体\s*[:：]?\s*(\d{1,3})"
                ]
            }

            found_scores = False
            for key, pattern_list in patterns.items():
                score_found = False
                for pattern in pattern_list:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            score_val = int(match.group(1))
                            score_clamped = max(0, min(100, score_val))
                            extracted_data["score"][key] = score_clamped
                            logger.info(f"Found score {key}={score_clamped}")
                            score_found = True
                            found_scores = True
                            break
                        except ValueError:
                            logger.warning(f"Value '{match.group(1)}' for '{key}' not integer.")
                
                if not score_found:
                    logger.debug(f"Score for '{key}' not found in text")

            if not found_scores:
                logger.warning("No scores could be extracted from text")
                return None

            return extracted_data

        except Exception as e:
            logger.error(f"Score parsing failed: {e}")
            return None

    def _safe_model_chat(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        """Wrapper aman untuk model.chat dengan error handling"""
        try:
            if not prompt or not isinstance(prompt, str):
                logger.error("Invalid prompt provided to model.chat")
                return ""
                
            response, _ = self.model.chat(
                self.tokenizer, 
                prompt, 
                history=None, 
                system=system_message
            )
            return response if response else ""
            
        except Exception as e:
            logger.error(f"Model chat failed: {e}")
            return ""

    # --- FUNGSI UTAMA ---
    def generate_json(self, essay: str, hsk_level: int = 2) -> str:
        """Fungsi utama untuk generate JSON result dengan validasi HSK"""
        start_time = time.time()
        
        # Validasi input
        if not essay or not essay.strip():
            error_result = {
                "error": "Input essay kosong", 
                "essay": essay, 
                "processing_time": "0.00s",
                "hsk_level": hsk_level
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        # Validasi HSK level
        if not self._validate_hsk_level(hsk_level):
            # Fallback ke level 2 jika invalid
            hsk_level = 2
            logger.info(f"Using fallback HSK level: {hsk_level}")

        logger.info(f"Processing essay for HSK {hsk_level}, length: {len(essay)} chars")

        # --- LANGKAH 1: ERROR DETECTION ---
        logger.info("Step 1: Detecting Errors...")
        validated_error_list = []
        try:
            error_prompt = self._build_error_detection_prompt(essay, hsk_level)
            error_response = self._safe_model_chat(
                error_prompt, 
                "您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。"
            )
            
            if error_response:
                logger.info(f"Error detection response received, length: {len(error_response)}")
                validated_error_list = self._parse_errors_from_text(error_response, essay)
            else:
                logger.warning("Empty response from error detection")
                
            logger.info(f"Step 1 Done. Found {len(validated_error_list)} errors.")
            
        except Exception as e: 
            logger.exception("Step 1 (Error Detection) Failed.")
            validated_error_list = []

        # --- LANGKAH 2: SCORING ---
        logger.info("Step 2: Getting Scores...")
        
        # Default scores
        default_scores = {
            "grammar": 0, "vocabulary": 0, "coherence": 0, 
            "cultural_adaptation": 0, "overall": 0
        }
        parsed_scores = default_scores.copy()
        
        try:
            scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
            scoring_response = self._safe_model_chat(scoring_prompt, "You are a helpful assistant.")
            
            if scoring_response:
                logger.info(f"Scoring response received, length: {len(scoring_response)}")
                parsed_scores_data = self._extract_scores_from_text(scoring_response)
                
                if parsed_scores_data and "score" in parsed_scores_data:
                    # Update hanya scores yang berhasil di-parse
                    for key in default_scores.keys():
                        if key in parsed_scores_data["score"]:
                            parsed_scores[key] = parsed_scores_data["score"][key]
                    logger.info(f"Parsed scores: {parsed_scores}")
                else:
                    logger.warning("No scores parsed from response, using defaults")
            else:
                logger.warning("Empty response from scoring")
                
        except Exception as e:
            logger.exception("Step 2 (Scoring) Failed.")

        # Calculate overall score if missing or zero but other scores exist
        grammar_s = parsed_scores["grammar"]
        vocab_s = parsed_scores["vocabulary"] 
        coherence_s = parsed_scores["coherence"]
        cultural_s = parsed_scores["cultural_adaptation"]
        overall_s = parsed_scores["overall"]

        # Jika overall 0 tapi ada komponen lain yang memiliki nilai, hitung ulang
        if overall_s == 0 and (grammar_s > 0 or vocab_s > 0 or coherence_s > 0 or cultural_s > 0):
            logger.info("Recalculating overall score from components...")
            calc_score = (grammar_s * self.rubric_weights["grammar"] +
                         vocab_s * self.rubric_weights["vocabulary"] +
                         coherence_s * self.rubric_weights["coherence"] +
                         cultural_s * self.rubric_weights["cultural_adaptation"])
            overall_s = max(0, min(100, int(round(calc_score))))
            parsed_scores["overall"] = overall_s
            logger.info(f"Recalculated overall score: {overall_s}")

        # --- LANGKAH 3: FEEDBACK ---
        logger.info("Step 3: Generating Feedback...")
        feedback = ""
        try:
            feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
            feedback_response = self._safe_model_chat(
                feedback_prompt, 
                "您是一位友好且善于鼓励的中文老师。"
            )
            
            if feedback_response:
                feedback = feedback_response.strip()
                logger.info("Feedback generated successfully")
            else:
                logger.warning("Empty response from feedback generation")
                
            # Fallback feedback berdasarkan scores dan errors
            if not feedback:
                if not validated_error_list and overall_s >= 80:
                    feedback = "作文写得很好，表达清晰流畅！继续保持！(Your essay is well-written with clear expression! Keep it up!)"
                elif not validated_error_list and overall_s >= 60:
                    feedback = "作文基本通顺，可以尝试使用更丰富的词汇。(Your essay is basically fluent, try using richer vocabulary.)"
                elif validated_error_list:
                    feedback = "作文中有一些需要改进的地方，请参考错误列表进行修改。(There are some areas for improvement in your essay, please refer to the error list.)"
                else:
                    feedback = "请继续练习写作，注意语法和词汇的正确使用。(Please continue practicing writing, pay attention to grammar and vocabulary usage.)"
                    
            logger.info("Step 3 Done.")
            
        except Exception as e:
            logger.exception("Step 3 (Feedback) Failed.")
            feedback = "由于系统错误无法生成详细反馈，请检查作文内容。(Unable to generate detailed feedback due to system error, please check your essay content.)"

        # --- FINAL ASSEMBLY ---
        duration = time.time() - start_time
        
        final_result = {
            "text": essay, 
            "hsk_level": hsk_level,
            "overall_score": parsed_scores["overall"],
            "detailed_scores": {
                "grammar": parsed_scores["grammar"], 
                "vocabulary": parsed_scores["vocabulary"],
                "coherence": parsed_scores["coherence"], 
                "cultural_adaptation": parsed_scores["cultural_adaptation"]
            },
            "error_list": validated_error_list, 
            "feedback": feedback,
            "processing_time": f"{duration:.2f} detik",
            "word_count": len(essay.strip())
        }
        
        logger.info(f"All steps completed. Time: {duration:.2f}s, Overall score: {parsed_scores['overall']}")

        # Robust JSON serialization
        try:
            return json.dumps(final_result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            error_json = {
                "error": "Internal error: Failed to serialize result", 
                "details": str(e), 
                "essay": essay, 
                "hsk_level": hsk_level,
                "processing_time": f"{duration:.2f}s"
            }
            return json.dumps(error_json, ensure_ascii=False, indent=2)

# ---------------- Testing ----------------
if __name__ == "__main__":
    logger.info("Testing improved model...")
    try:
        scorer = QwenScorer()
        
        print("\n" + "="*50)
        print("TEST 1: Valid Essay - HSK 2")
        print("="*50)
        test_essay_1 = "我喜欢学习中文。今天天气很好。我和朋友去公园。"
        result_1 = scorer.generate_json(test_essay_1, hsk_level=2)
        print(result_1)
        
        print("\n" + "="*50)
        print("TEST 2: Essay with Errors - HSK 3") 
        print("="*50)
        test_essay_2 = "我妹妹是十岁。我们住雅加达在。今天路很忙。"
        result_2 = scorer.generate_json(test_essay_2, hsk_level=3)
        print(result_2)
        
        print("\n" + "="*50)
        print("TEST 3: Invalid HSK Level (should fallback to 2)")
        print("="*50)
        test_essay_3 = "你好。我叫安娜。"
        result_3 = scorer.generate_json(test_essay_3, hsk_level=5)  # Invalid level
        print(result_3)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

        #----------------------------------------------------------------------------------------------------------------------------------------------

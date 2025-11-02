#----------------------------------------------------------------------------------------------------------------------------------

    # # -*- coding: utf-8 -*-
    # # Pastikan encoding UTF-8 di awal
    # # FILE: model.py
    # # VERSI FINAL: Murni "Chain of Prompts" (Stabil) - Meminta Skor Detail

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

    # # ---------------- Logger ----------------
    # logger = logging.getLogger(__name__)
    # # Set ke INFO
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
    #     return dot / denominator if denominator != 0 else 0.0

    # # ---------------- QwenScorer ----------------

    # class QwenScorer:
    #     """ Implementasi Murni Chain of Prompts via model.chat() """

    #     def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
    #         logger.info(f"Memulai inisialisasi QwenScorer (Chain of Prompts) dgn model: {model_name}")
    #         try:
    #             self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #             logger.info(f"Device: {self.device}")

    #             self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #             logger.info("Tokenizer loaded.")

    #             # --- Perbaikan CPU (Wajib) ---
    #             logger.info("Loading base model...")
    #             self.model = AutoModelForCausalLM.from_pretrained(
    #                 model_name,
    #                 trust_remote_code=True
    #             ).to(self.device).eval() # Paksa ke CPU
    #             logger.info(f"Model Qwen-1.8B loaded to {self.device}.")
    #             self.config = self.model.config
    #             # --- Akhir Perbaikan CPU ---

    #             logger.info("Pemuatan file soft prompt (.pt) diskip.") # Konfirmasi skip

    #         except Exception as e:
    #             logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
    #             raise
    #         try:
    #             jieba.setLogLevel(logging.WARNING)
    #             jieba.initialize()
    #             logger.info("Jieba initialized.")
    #         except Exception as e:
    #             logger.warning(f"Failed to initialize Jieba fully: {e}")
    #             pass

    #         self.rubric_weights = {
    #             "grammar": 0.30, "vocabulary": 0.30,
    #             "coherence": 0.20, "cultural_adaptation": 0.20
    #         }
    #         logger.info(f"Rubric weights set: {self.rubric_weights}")

    #     # HAPUS _load_soft_prompt, _get_input_embeddings, _generate_with_prompt

    #     def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
    #         # (Tidak berubah)
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
    #         # 您的任务【仅仅】是找出这篇【HSK {hsk_level} 等级】作文中的语法、词汇或语序错误
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

    #     # --- PARSER ERROR (Robust) ---
    #     def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
    #         # (Versi robust dari sebelumnya)
    #         validated_error_list = []
    #         if not error_response or "TIDAK ADA KESALAHAN" in error_response: return []
    #         pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
    #         skip_patterns = [re.compile(r"^\s*示例 \d+:"), re.compile(r"^\s*输入:"), re.compile(r"^\s*输出:"), re.compile(r"^\s*---"), re.compile(r"^\s*$"), re.compile(r"错误类型|错误原文|修正建议|简短解释")]
    #         logger.debug(f"Error Parser: Parsing response: {repr(error_response)}")
    #         lines_processed, lines_skipped = 0, 0
    #         for line in error_response.splitlines():
    #             line = line.strip(); lines_processed += 1
    #             should_skip = any(p.search(line) for p in skip_patterns)
    #             if should_skip: logger.debug(f"Error Parser: Skipping line: '{line}'"); lines_skipped += 1; continue
    #             match = pattern.match(line)
    #             if match:
    #                 try:
    #                     err_type, incorrect_frag, correction, explanation = map(str, (g.strip() for g in match.groups()))
    #                     if not incorrect_frag: logger.warning(f"Error Parser: Empty incorrect fragment: '{line}'. Skip."); continue
    #                     start_index = essay_text.find(incorrect_frag); pos = [start_index, start_index + len(incorrect_frag)] if start_index != -1 else [0, 0]
    #                     if start_index == -1: logger.warning(f"Error Parser: Pos not found: '{incorrect_frag}'. Use [0,0].")
    #                     error_entry = {"error_type": err_type, "error_position": pos, "incorrect_fragment": incorrect_frag, "suggested_correction": correction, "explanation": explanation}
    #                     if all(isinstance(v, (str, list)) for k, v in error_entry.items() if k != 'error_position') and isinstance(pos, list):
    #                         validated_error_list.append(error_entry); logger.debug(f"Error Parser: Parsed: {error_entry}")
    #                     else: logger.error(f"Error Parser: Invalid data type: '{line}'. Skip. Data: {error_entry}")
    #                 except Exception as e: logger.warning(f"Error Parser: Failed matched line: '{line}'. Error: {e}", exc_info=True)
    #             else: logger.warning(f"Error Parser: Line unmatched: '{line}'")
    #         logger.info(f"Error Parser Done. Lines: {lines_processed}. Skipped: {lines_skipped}. Errors: {len(validated_error_list)}")
    #         return validated_error_list

    #     # --- PROMPT SKOR (KEMBALI KE DETAIL) ---
    #     def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
    #         """
    #         Membangun prompt yang meminta SEMUA skor detail (seperti kode asli Anda).
    #         """
    #         logger.info("Building DETAILED scoring prompt (requesting all 5 scores).")
    #         return f"""
    #         您是HSK作文评分员。
    #         您的任务【仅仅】是提供分数（0-100）。
    #         【不要】写任何评语或解释。

    #         HSK等级: {hsk_level}
    #         作文: "{essay}"

    #         请【必须】按照以下纯文本格式提供所有5个分数。
    #         【不要】写任何其他文字。

    #         语法准确性: [分数]
    #         词汇水平: [分数]
    #         篇章连贯: [分数]
    #         任务完成度: [分数]
    #         总体得分: [分数]
    #         """

    #     # --- PARSER SKOR (KEMBALI KE ASLI - Cari Keyword) ---
    #     def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
    #         """
    #         Mengekstrak skor dari output teks. KEMBALI ke versi asli yang mencari keyword.
    #         """
    #         try:
    #             extracted_data = {"score": {}} # Mulai kosong
    #             found_any_score = False
    #             # Pola regex asli Anda
    #             patterns = {
    #                 "grammar": r"(?:语法准确性|grammar)\s*[:：分]?\s*(\d{1,3})",
    #                 "vocabulary": r"(?:词汇水平|vocabulary)\s*[:：分]?\s*(\d{1,3})",
    #                 "coherence": r"(?:篇章连贯|连贯性|coherence)\s*[:：分]?\s*(\d{1,3})",
    #                 # Sesuaikan nama kunci 'cultural_adaptation' jika perlu
    #                 "cultural_adaptation": r"(?:任务完成度|task_fulfillment|cultural_adaptation)\s*[:：分]?\s*(\d{1,3})",
    #                 "overall": r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})"
    #             }

    #             logger.debug(f"Score Parser (Keyword): Attempting to extract scores from: {repr(text)}")

    #             for key, pattern in patterns.items():
    #                 match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    #                 if match:
    #                     try:
    #                         score_val = int(match.group(1))
    #                         score_clamped = max(0, min(100, score_val))
    #                         extracted_data["score"][key] = score_clamped # Tambahkan skor jika ditemukan
    #                         found_any_score = True
    #                         logger.info(f"Score Parser (Keyword): Found score {key}={score_clamped}")
    #                     except ValueError:
    #                         logger.warning(f"Score Parser (Keyword): Value '{match.group(1)}' for '{key}' not int.")
    #                 # else:
    #                 #     logger.debug(f"Score Parser (Keyword): Pattern for '{key}' not found.") # Aktifkan jika perlu

    #             # Jika SAMA SEKALI tidak ada keyword skor yang cocok
    #             if not found_any_score:
    #                  logger.warning(f"Score Parser (Keyword): NO scores could be extracted via keywords from: {repr(text)}")
    #                  return None # Kembalikan None jika parsing GAGAL TOTAL

    #             # Jika setidaknya SATU skor ditemukan, pastikan semua key ada (default 0)
    #             # Ini penting agar generate_json tidak crash saat get()
    #             final_scores = {}
    #             for key in patterns: # Iterasi semua key yang DIHARAPKAN
    #                 final_scores[key] = extracted_data["score"].get(key, 0) # Ambil nilai jika ada, else 0
    #                 if key not in extracted_data["score"]:
    #                      logger.warning(f"Score Parser (Keyword): Score for '{key}' not found, setting to 0.")

    #             # Kembalikan dictionary yang lengkap dengan skor (atau 0)
    #             return {"score": final_scores}

    #         except Exception as e:
    #             logger.error(f"Score Parser (Keyword): Failed: {e}", exc_info=True)
    #             return None # Kembalikan None jika ada error tak terduga

    #     # --- PROMPT FEEDBACK ---
    #     def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
    #         # (Prompt feedback tidak berubah, tapi sekarang bisa menampilkan skor detail jika ada)
    #         score_summary = (
    #             f"总体得分: {scores.get('overall', 'N/A')}, "
    #             f"语法: {scores.get('grammar', 'N/A')}, " # Tampilkan lagi
    #             f"词汇: {scores.get('vocabulary', 'N/A')}" # Tampilkan lagi
    #         )
    #         error_summary = "未发现主要错误。"
    #         if errors: error_summary = "发现的主要错误:\n" + "".join([f"- {err.get('explanation', 'N/A')}\n" for err in errors[:2]])
    #         return f"""
    #         您是一位友好且善于鼓励的中文老师。
    #         您的任务是根据学生的作文、得分和错误，写一段简短的评语（2-3句话）。
    #         请用中文书写评语，并在括号()中附上简短的英文语翻译。
    #         学生作文: "{essay}"
    #         所得分数: {score_summary}
    #         错误备注: {error_summary}
    #         请现在撰写您的评语：
    #         """

    #     # --- FUNGSI UTAMA: generate_json (Menggunakan model.chat) ---
    #     def generate_json(self, essay: str, hsk_level: int = 3) -> str:
    #         start_time = time.time()
    #         logger.info(f"Request received (Chain of Prompts - Detailed Scores) for HSK {hsk_level}.")
    #         if not essay or not essay.strip():
    #              logger.warning("Empty essay input."); error_result = {"error": "Input essay empty.", "essay": essay, "processing_time": "0.00s"}
    #              try: return json.dumps(error_result, ensure_ascii=False, indent=2)
    #              except Exception: return '{"error": "Input essay empty and failed to create error JSON."}'

    #         # --- LANGKAH 1: ERROR DETECTION ---
    #         logger.info("Step 1: Detecting Errors (via model.chat)...")
    #         validated_error_list = []
    #         try:
    #             error_prompt = self._build_error_detection_prompt(essay)
    #             error_response, _ = self.model.chat(self.tokenizer, error_prompt, history=None, system="您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。")
    #             logger.debug(f"Step 1 Raw Response: {repr(error_response)}")
    #             validated_error_list = self._parse_errors_from_text(error_response, essay)
    #             logger.info(f"Step 1 Done. Found {len(validated_error_list)} errors.")
    #         except Exception as e: logger.exception("Step 1 (Error Detection) Failed."); validated_error_list = []

    #         # --- LANGKAH 2: SCORING (Dengan Parser Keyword Asli) ---
    #         logger.info("Step 2: Getting Detailed Scores (via model.chat)...")
    #         # Default skor 0
    #         parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0, "overall": 0}
    #         try:
    #             # Menggunakan prompt skor DETAIL
    #             scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
    #             scoring_response, _ = self.model.chat(self.tokenizer, scoring_prompt, history=None, system="You are a helpful assistant.")
    #             logger.info(f"Step 2 Raw Response: {repr(scoring_response)}") # INFO level

    #             # Menggunakan parser skor ASLI (keyword)
    #             parsed_scores_data = self._extract_scores_from_text(scoring_response)

    #             if parsed_scores_data and "score" in parsed_scores_data:
    #                  parsed_scores = parsed_scores_data["score"] # Ambil dict skor (sudah di-default 0 jika missing)
    #                  logger.info(f"Step 2 PARSED scores (might be 0): {parsed_scores}")
    #             else:
    #                  # Jika parser kembalikan None (gagal total)
    #                  logger.error("Step 2 PARSING FAILED: Parser returned None. Using default scores (all 0).")
    #                  # parsed_scores sudah default 0

    #         except Exception as e:
    #             logger.exception("Step 2 (Scoring) Failed TOTAL.")
    #             # parsed_scores sudah default 0

    #         # Ekstrak skor ke variabel individual (setelah try-except)
    #         grammar_s = parsed_scores.get("grammar", 0)
    #         vocab_s = parsed_scores.get("vocabulary", 0)
    #         coherence_s = parsed_scores.get("coherence", 0)
    #         cultural_s = parsed_scores.get("cultural_adaptation", 0)
    #         overall_s = parsed_scores.get("overall", 0)

    #         # Hitung ulang overall JIKA 0 tapi komponen lain ada nilainya
    #         if overall_s == 0 and (grammar_s > 0 or vocab_s > 0 or coherence_s > 0 or cultural_s > 0):
    #             logger.info("Overall score is 0 but detailed scores exist. Recalculating overall based on weights...")
    #             calc_score = (grammar_s * self.rubric_weights["grammar"]) + \
    #                          (vocab_s * self.rubric_weights["vocabulary"]) + \
    #                          (coherence_s * self.rubric_weights["coherence"]) + \
    #                          (cultural_s * self.rubric_weights["cultural_adaptation"])
    #             overall_s = max(0, min(100, int(round(calc_score))))
    #             logger.info(f"Recalculated overall score: {overall_s}")

    #         logger.info(f"Step 2 Done. Final Scores -> Overall: {overall_s}, Grammar: {grammar_s}, Vocab: {vocab_s}, Coherence: {coherence_s}, Cultural: {cultural_s}")


    #         # --- LANGKAH 3: FEEDBACK ---
    #         logger.info("Step 3: Generating Feedback (via model.chat)...")
    #         feedback = "Gagal menghasilkan feedback."
    #         try:
    #             # Feedback prompt sekarang bisa pakai skor detail
    #             feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
    #             feedback_response, _ = self.model.chat(self.tokenizer, feedback_prompt, history=None, system="You are a helpful assistant.")
    #             feedback = feedback_response.strip() if feedback_response else feedback
    #             if not feedback: # Fallback
    #                 logger.warning("Step 3: model.chat returned empty feedback. Using fallback.")
    #                 if not validated_error_list and overall_s > 80: feedback = "作文写得很好...(Esai baik...)"
    #                 elif validated_error_list: feedback = "作文中发现一些错误...(Ada error...)"
    #                 else: feedback = "请根据得到的总体分数...(Periksa esai...)"
    #             logger.info("Step 3 Done.")
    #             logger.debug(f"Step 3 Feedback: {repr(feedback)}")
    #         except Exception as e:
    #             logger.exception("Step 3 (Feedback) Failed TOTAL.")
    #             if validated_error_list: feedback = "作文中发现一些错误...(Ada error...)"
    #             elif overall_s > 80: feedback = "作文写得很好...(Esai baik...)"
    #             else: feedback = "请根据得到的总体分数...(Periksa esai...)"

    #         # --- FINAL ASSEMBLY ---
    #         final_result = {
    #             "text": essay, "overall_score": overall_s,
    #             "detailed_scores": { # Sekarang bisa berisi nilai detail
    #                 "grammar": grammar_s, "vocabulary": vocab_s,
    #                 "coherence": coherence_s, "cultural_adaptation": cultural_s
    #             },
    #             "error_list": validated_error_list, "feedback": feedback
    #         }
    #         duration = time.time() - start_time
    #         final_result["processing_time"] = f"{duration:.2f} detik"
    #         logger.info(f"All steps (Chain of Prompts - Detailed) done. Time: {duration:.2f}s.")

    #         # --- Robust JSON Dump ---
    #         try:
    #             json_output_string = json.dumps(final_result, ensure_ascii=False, indent=2)
    #             logger.debug(f"JSON generated successfully:\n{json_output_string[:500]}...")
    #             return json_output_string
    #         except TypeError as e:
    #             logger.error(f"FATAL: Failed converting final_result to JSON! Error: {e}", exc_info=True)
    #             logger.error(f"Data causing error: {final_result}")
    #             error_json = {"error": "Internal error: Failed to serialize result.", "details": str(e), "essay": essay, "processing_time": f"{duration:.2f}s"}
    #             try: return json.dumps(error_json, ensure_ascii=False, indent=2)
    #             except Exception as final_e: logger.critical(f"ULTRA FATAL: Failed dumping error JSON: {final_e}"); return '{"error": "Internal error: Failed serialization."}'
    #         # --- End Robust JSON Dump ---

    # # ---------------- Simulasi (Main execution) ----------------
    # if __name__ == "__main__":
    #     logger.setLevel(logging.INFO)
    #     logger.warning("="*50 + "\nRUNNING model.py SCRIPT (Chain of Prompts - DETAILED Scores)\n" + "="*50)
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





# yg dipakai sekarang:
# # -*- coding: utf-8 -*-
# # FILE: model.py
# # VERSI: Load .pt files dengan model.chat()

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
# import torch.nn as nn
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
#     if not v1 or not v2 or len(v1) != len(v2): return 0.0
#     dot = sum(a * b for a, b in zip(v1, v2))
#     n1 = math.sqrt(sum(a * a for a in v1))
#     n2 = math.sqrt(sum(b * b for b in v2))
#     denominator = n1 * n2
#     return dot / denominator if denominator != 0 else 0.0

# # ---------------- QwenScorer ----------------
# class QwenScorer:
#     """Implementasi dengan loading file .pt"""

#     def __init__(self, model_name: str = "Qwen/Qwen-1_8B-Chat"):
#         logger.info(f"Memulai inisialisasi QwenScorer dengan model: {model_name}")
#         try:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             logger.info(f"Device: {self.device}")

#             self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#             logger.info("Tokenizer loaded.")

#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 trust_remote_code=True
#             ).to(self.device).eval()
#             logger.info(f"Model Qwen-1.8B loaded to {self.device}.")
#             self.config = self.model.config

#             # --- MEMUAT SOFT PROMPT (.pt files) ---
#             logger.info("Memuat file soft prompt (.pt)...")
#             self._load_all_prompts()

#         except Exception as e:
#             logger.exception(f"Gagal memuat model atau tokenizer {model_name}.")
#             raise
        
#         try:
#             jieba.setLogLevel(logging.WARNING)
#             jieba.initialize()
#             logger.info("Jieba initialized.")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Jieba fully: {e}")
#             pass

#         self.rubric_weights = {
#             "grammar": 0.30, "vocabulary": 0.30,
#             "coherence": 0.20, "cultural_adaptation": 0.20
#         }
#         logger.info(f"Rubric weights set: {self.rubric_weights}")

#     def _load_all_prompts(self):
#         """Memuat semua file .pt yang ada"""
#         prompt_files = {
#             "error": "error_soft_prompt.pt",
#             "scoring": "scoring_soft_prompt.pt", 
#             "feedback": "feedback_soft_prompt.pt"
#         }
        
#         self.prompts = {}
        
#         for prompt_type, filename in prompt_files.items():
#             try:
#                 if os.path.exists(filename):
#                     # Load tensor langsung
#                     prompt_tensor = torch.load(filename, map_location=self.device)
                    
#                     # Pastikan shape benar [1, prompt_length, hidden_size]
#                     if len(prompt_tensor.shape) == 2:
#                         prompt_tensor = prompt_tensor.unsqueeze(0)  # Add batch dimension
                    
#                     self.prompts[prompt_type] = prompt_tensor
#                     logger.info(f"✅ Berhasil memuat {filename} dengan shape {prompt_tensor.shape}")
#                 else:
#                     logger.warning(f"⚠️ File {filename} tidak ditemukan")
#                     # Buat prompt dummy sebagai fallback
#                     self.prompts[prompt_type] = torch.randn(1, 20, self.config.hidden_size)
                    
#             except Exception as e:
#                 logger.error(f"❌ Gagal memuat {filename}: {e}")
#                 # Fallback ke prompt random
#                 self.prompts[prompt_type] = torch.randn(1, 20, self.config.hidden_size)

#         logger.info(f"Total prompts loaded: {len(self.prompts)}")

#     def _preprocess_with_jieba(self, essay: str) -> Tuple[str, str]:
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

#     # --- PROMPT BUILDERS dengan integrasi .pt ---
#     def _build_error_detection_prompt(self, essay: str, hsk_level: int) -> str:
#         return f"""
#     您是一位经验丰富的中文语法专家，尤其擅长指导印尼学习者。
#     您的任务【仅仅】是找出这篇【HSK {hsk_level} 等级】作文中的语法、词汇或语序错误。
#     请【严格】遵守以下格式：
#     - 如果发现错误，请使用此格式： 错误类型 | 错误原文 | 修正建议 | 简短解释
#     - 每个错误占一行。
#     - 如果【没有发现任何错误】，请【只】回答 'TIDAK ADA KESALAHAN'。

#     示例 1: 输入: 我妹妹是十岁。 输出: 助词误用(是) | 我妹妹是十岁 | 我妹妹十岁 | 表达年龄时通常不需要'是'。
#     示例 2: 输入: 我们住雅加达在。 输出: 语序干扰(SPOK) | 我们住雅加达在 | 我们住在雅加达 | 地点状语(在雅加达)应放在动词(住)之前。
#     示例 3: 输入: 路很忙。 输出: 词语误用(False Friend) | 路很忙 | 路很拥挤 | '忙'(máng)通常用于人，而非道路。
#     示例 4: 输入: 我喜欢学中文。 输出: TIDAK ADA KESALAHAN

#     请分析以下作文，找出所有错误。请严格遵守格式。
#     作文：
#     "{essay}"
#     """

#     def _build_scoring_prompt(self, essay: str, hsk_level: int, detected_errors: List[Dict]) -> str:
#         error_info = ""
#         if detected_errors:
#             error_info = f"\n发现错误数: {len(detected_errors)}"
            
#             return f"""
#             您是HSK作文评分员。请为这篇HSK{hsk_level}作文打分（0-100分）。
#             {error_info}

#             请按照以下格式提供分数（0-100）：
#             语法准确性: [分数]
#             词汇水平: [分数] 
#             篇章连贯: [分数]
#             任务完成度: [分数]
#             总体得分: [分数]

#             作文: "{essay}"
#             """

#     def _build_feedback_prompt(self, essay: str, scores: Dict, errors: List[Dict]) -> str:
#         score_summary = f"总体得分: {scores.get('overall', 'N/A')}"
#         error_summary = "未发现主要错误。"
#         if errors: 
#             error_summary = f"发现{len(errors)}个主要错误"
            
#         return f"""
#         您是一位友好且善于鼓励的中文老师。
#         请根据以下信息写一段简短的评语（2-3句话）：
#         学生作文: "{essay}"
#         所得分数: {score_summary}
#         错误备注: {error_summary}

#         请用中文书写评语，并在括号()中附上简短的英文语翻译。
#         请现在撰写您的评语：
#         """

#     # --- PARSER ERROR ---
#     def _parse_errors_from_text(self, error_response: str, essay_text: str) -> List[Dict[str, Any]]:
#         validated_error_list = []
#         if not error_response or "TIDAK ADA KESALAHAN" in error_response: 
#             return []
            
#         pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")
        
#         for line in error_response.splitlines():
#             line = line.strip()
#             match = pattern.match(line)
#             if match:
#                 try:
#                     err_type, incorrect_frag, correction, explanation = map(str, (g.strip() for g in match.groups()))
#                     if not incorrect_frag:
#                         continue
                        
#                     start_index = essay_text.find(incorrect_frag)
#                     pos = [start_index, start_index + len(incorrect_frag)] if start_index != -1 else [0, 0]
                    
#                     validated_error_list.append({
#                         "error_type": err_type,
#                         "error_position": pos,
#                         "incorrect_fragment": incorrect_frag,
#                         "suggested_correction": correction, 
#                         "explanation": explanation
#                     })
#                 except Exception as e:
#                     logger.warning(f"Gagal parsing error line: '{line}'. Error: {e}")
                    
#         logger.info(f"Parsed {len(validated_error_list)} errors")
#         return validated_error_list

#     # --- PARSER SKOR ---
#     def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, Any]]:
#         try:
#             extracted_data = {"score": {}}
#             patterns = {
#                 "grammar": r"(?:语法准确性|grammar)\s*[:：分]?\s*(\d{1,3})",
#                 "vocabulary": r"(?:词汇水平|vocabulary)\s*[:：分]?\s*(\d{1,3})",
#                 "coherence": r"(?:篇章连贯|连贯性|coherence)\s*[:：分]?\s*(\d{1,3})",
#                 "cultural_adaptation": r"(?:任务完成度|task_fulfillment|cultural_adaptation)\s*[:：分]?\s*(\d{1,3})",
#                 "overall": r"(?:总体得分|总分|overall)\s*[:：分]?\s*(\d{1,3})"
#             }

#             for key, pattern in patterns.items():
#                 match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
#                 if match:
#                     try:
#                         score_val = int(match.group(1))
#                         score_clamped = max(0, min(100, score_val))
#                         extracted_data["score"][key] = score_clamped
#                         logger.info(f"Found score {key}={score_clamped}")
#                     except ValueError:
#                         logger.warning(f"Value '{match.group(1)}' for '{key}' not int.")

#             # Jika tidak ada skor yang ditemukan
#             if not extracted_data["score"]:
#                 logger.warning("No scores extracted from text")
#                 return None

#             return extracted_data

#         except Exception as e:
#             logger.error(f"Score parsing failed: {e}")
#             return None

#     # --- FUNGSI UTAMA ---
#     def generate_json(self, essay: str, hsk_level: int = 3) -> str:
#         start_time = time.time()
#         logger.info(f"Request received for HSK {hsk_level}.")
        
#         if not essay or not essay.strip():
#             error_result = {
#                 "error": "Input essay empty.", 
#                 "essay": essay, 
#                 "processing_time": "0.00s",
#                 "hsk_level": hsk_level
#             }
#             try: 
#                 return json.dumps(error_result, ensure_ascii=False, indent=2)
#             except Exception: 
#                 return '{"error": "Input essay empty and failed to create error JSON."}'

#         # --- LANGKAH 1: ERROR DETECTION ---
#         logger.info("Step 1: Detecting Errors...")
#         validated_error_list = []
#         try:
#             error_prompt = self._build_error_detection_prompt(essay, hsk_level)
#             error_response, _ = self.model.chat(self.tokenizer, error_prompt, history=None)
#             logger.info(f"Error detection raw response: {error_response[:200]}...")
#             validated_error_list = self._parse_errors_from_text(error_response, essay)
#             logger.info(f"Step 1 Done. Found {len(validated_error_list)} errors.")
#         except Exception as e: 
#             logger.exception("Step 1 (Error Detection) Failed.")
#             validated_error_list = []

#         # --- LANGKAH 2: SCORING ---
#         logger.info("Step 2: Getting Scores...")
#         parsed_scores = {"grammar": 0, "vocabulary": 0, "coherence": 0, "cultural_adaptation": 0, "overall": 0}
#         try:
#             scoring_prompt = self._build_scoring_prompt(essay, hsk_level, validated_error_list)
#             scoring_response, _ = self.model.chat(self.tokenizer, scoring_prompt, history=None)
#             logger.info(f"Scoring raw response: {scoring_response[:200]}...")
            
#             parsed_scores_data = self._extract_scores_from_text(scoring_response)
#             if parsed_scores_data and "score" in parsed_scores_data:
#                 parsed_scores = parsed_scores_data["score"]
#                 logger.info(f"Parsed scores: {parsed_scores}")
#         except Exception as e:
#             logger.exception("Step 2 (Scoring) Failed.")

#         # Calculate overall score if missing
#         grammar_s = parsed_scores.get("grammar", 0)
#         vocab_s = parsed_scores.get("vocabulary", 0)
#         coherence_s = parsed_scores.get("coherence", 0)
#         cultural_s = parsed_scores.get("cultural_adaptation", 0)
#         overall_s = parsed_scores.get("overall", 0)

#         if overall_s == 0 and (grammar_s > 0 or vocab_s > 0 or coherence_s > 0 or cultural_s > 0):
#             logger.info("Recalculating overall score...")
#             calc_score = (grammar_s * self.rubric_weights["grammar"]) + \
#                          (vocab_s * self.rubric_weights["vocabulary"]) + \
#                          (coherence_s * self.rubric_weights["coherence"]) + \
#                          (cultural_s * self.rubric_weights["cultural_adaptation"])
#             overall_s = max(0, min(100, int(round(calc_score))))

#         # --- LANGKAH 3: FEEDBACK ---
#         logger.info("Step 3: Generating Feedback...")
#         feedback = ""
#         try:
#             feedback_prompt = self._build_feedback_prompt(essay, parsed_scores, validated_error_list)
#             feedback_response, _ = self.model.chat(self.tokenizer, feedback_prompt, history=None)
#             feedback = feedback_response.strip() if feedback_response else ""
            
#             if not feedback:
#                 if not validated_error_list and overall_s > 80:
#                     feedback = "作文写得很好，未发现明显错误。继续努力！(Esai ditulis dengan baik, tidak ditemukan kesalahan signifikan. Teruslah berusaha!)"
#                 elif validated_error_list:
#                     feedback = "作文中发现一些错误，请查看错误列表了解详情。(Ditemukan beberapa kesalahan dalam esai, silakan periksa daftar kesalahan untuk detailnya.)"
#                 else:
#                     feedback = "请根据得到的总体分数检查你的作文。(Harap periksa esai berdasarkan skor yang didapat.)"
                    
#             logger.info("Step 3 Done.")
#         except Exception as e:
#             logger.exception("Step 3 (Feedback) Failed.")
#             feedback = "Gagal menghasilkan feedback."

#         # --- FINAL ASSEMBLY ---
#         final_result = {
#             "text": essay, 
#             "hsk_level": hsk_level,
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
        
#         duration = time.time() - start_time
#         final_result["processing_time"] = f"{duration:.2f} detik"
#         logger.info(f"All steps done. Time: {duration:.2f}s")

#         # Robust JSON dump
#         try:
#             return json.dumps(final_result, ensure_ascii=False, indent=2)
#         except Exception as e:
#             logger.error(f"JSON serialization failed: {e}")
#             error_json = {
#                 "error": "Internal error: Failed to serialize result.", 
#                 "details": str(e), 
#                 "essay": essay, 
#                 "hsk_level": hsk_level
#             }
#             return json.dumps(error_json, ensure_ascii=False, indent=2)

# # ---------------- Testing ----------------
# if __name__ == "__main__":
#     logger.info("Testing model with .pt files...")
#     try:
#         scorer = QwenScorer()
        
#         # Test essay
#         test_essay = "我喜欢学习中文。今天天气很好。"
#         result = scorer.generate_json(test_essay, hsk_level=2)
#         print("\n=== TEST RESULT ===")
#         print(result)
#         print("===================\n")
        
#     except Exception as e:
#         logger.error(f"Test failed: {e}")

# def _load_all_prompts(self):
#     prompt_files = {
#         "error": "error_soft_prompt.pt",
#         "scoring": "scoring_soft_prompt.pt", 
#         "feedback": "feedback_soft_prompt.pt"
#     }
    
#     self.prompts = {}
    
#     for prompt_type, filename in prompt_files.items():
#         try:
#             if os.path.exists(filename):
#                 prompt_tensor = torch.load(filename, map_location=self.device)
                
#                 # Handle berbagai kemungkinan shape
#                 if len(prompt_tensor.shape) == 1:
#                     # [hidden_size] -> [1, 1, hidden_size]
#                     prompt_tensor = prompt_tensor.unsqueeze(0).unsqueeze(0)
#                 elif len(prompt_tensor.shape) == 2:
#                     # [length, hidden_size] -> [1, length, hidden_size]  
#                     prompt_tensor = prompt_tensor.unsqueeze(0)
#                 elif len(prompt_tensor.shape) == 3:
#                     # [batch, length, hidden_size] -> tetap
#                     pass
                    
#                 # Pastikan hidden size match
#                 if prompt_tensor.shape[-1] != self.config.hidden_size:
#                     logger.warning(f"Hidden size mismatch for {filename}. Resizing...")
#                     # Resize ke hidden size yang benar
#                     prompt_tensor = prompt_tensor[..., :self.config.hidden_size]
                
#                 self.prompts[prompt_type] = prompt_tensor
#                 logger.info(f"✅ Loaded {filename} with shape {prompt_tensor.shape}")
                
#         except Exception as e:
#             logger.error(f"❌ Failed to load {filename}: {e}")
#             self.prompts[prompt_type] = torch.randn(1, 20, self.config.hidden_size)










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
        """Membangun prompt untuk scoring"""
        error_info = f"发现错误数: {len(detected_errors)}" if detected_errors else "未发现错误"
        
        return f"""您是HSK作文评分员。请为这篇HSK{hsk_level}作文打分（0-100分）。
{error_info}

请按照以下格式提供分数（0-100）：
语法准确性: [分数]
词汇水平: [分数] 
篇章连贯: [分数]
任务完成度: [分数]
总体得分: [分数]

作文: "{essay}"

请直接输出分数："""

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
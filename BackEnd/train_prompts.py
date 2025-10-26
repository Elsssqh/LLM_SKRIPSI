# -*- coding: utf-8 -*-
# FILE: train_prompts.py
# Ini adalah skrip untuk PELATIHAN "Soft Prompts"
# Berdasarkan teori dari Paper 3 ("The Power of Scale for Parameter-Efficient Prompt Tuning")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import logging
from torch.optim import AdamW

# ---------------- Logger ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# ---------------- TENTUKAN KONFIGURASI ----------------
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
PROMPT_LENGTH = 20  # Paper 3 menemukan 20+ token bekerja baik [cite: 223]
LEARNING_RATE = 0.3 # Paper 3 menggunakan LR tinggi seperti 0.3 [cite: 139]
BATCH_SIZE = 8
EPOCHS = 3

# ---------------- MOCK DATASET (GANTI DENGAN DATA ANDA) ----------------
# Anda HARUS memiliki dataset berlabel untuk melatih prompt 
# Contoh ini hanya untuk melatih 'scoring_prompt'
# Format: (input_teks, output_teks_yang_diharapkan)
# SCORING_TRAIN_DATA = [
#     ("HSK: 3 Esai: 林美美好朋友，你好！我聽說你最近找到了工作，你真棒！恭喜，恭喜！我替你很高興！我聽說你打算開一個慶祝會。對不起，我要參加，可是沒有空。你開一個慶祝會的時候我不能會參加，是因為我在外國做工作。可惜我不能參加。我回台灣以後，我們應該找時間可以好好地聊聊。我有很多問題要問你。你找到了什麼工作？你什麼時候開始上班？祝身體健康張愛文。", "语法准确性: 65\n词汇水平: 90\n篇章连贯: 85\n任务完成度: 90\n总体得分: 78"),
#     ("HSK: 3 Esai: 我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係，沒有辦法參加。真的不好意思！下下個禮拜，如果你有空的話，我們一起喝咖啡", "语法准确性: 60\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 72"),
#     ("HSK: 4 Esai: 祝你找到了工作。最近找到工作很難，可是你容易找到。我很高興！我聽說你下個星期六要開一個慶祝會。可是我不能參加，因為我計畫跟我的父母到花蓮去旅行。我下次跟你見面的時候，請你吃飯。", "语法准确性: 70\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 80\n总体得分: 76"),
#     ("HSK: 4 Esai: 你好！我是張愛文。我聽到你找到工作。恭喜恭喜！我很高興你找到了工作！我寫卡是為了恭喜你，還有我要說我不能去慶祝會。我對你很抱歉。我也很想去，可是沒有辦法。是因為慶祝會的日子是我母親的生日。所以那個時候我跟我家人都要去韓國玩。我下個禮拜天才回來。我希望你了解我。我答應你，我回來的時候，我一定要帶送給你的禮物。那下次見面吧。", "语法准确性: 75\n词汇水平: 85\n篇章连贯: 85\n任务完成度: 85\n总体得分: 80"),
#     ("HSK: 3 Esai: 恭喜你剛剛找到了一個工作！我為你真高興。謝謝你邀請我來你慶祝會，但是我不能參加，因為我已經跟我的男朋友約好了一起去旅行。我們在你開晚會的那一天就出去，可是我真希望回來以後再跟你見面。恭喜發財！", "语法准确性: 68\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 74"),
#     ("HSK: 2 Esai: 我不能去你的派對，因為我生病了。醫生說我需要休息一週。祝你新工作一切順利！", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 65\n任务完成度: 60\n总体得分: 58"),
#     ("HSK: 4 Esai: 親愛的美美：恭喜你找到新工作！我很想去你的慶祝會，但那天我正好要參加一個重要的面試，實在無法分身。希望你能理解。等你有空，我們再約！", "语法准确性: 80\n词汇水平: 90\n篇章连贯: 90\n任务完成度: 85\n总体得分: 85"),
#     ("HSK: 2 Esai: 對不起我不能去。因為我有事情。祝你工作好。", "语法准确性: 40\n词汇水平: 40\n篇章连贯: 45\n任务完成度: 45\n总体得分: 42"),
#     ("HSK: 4 Esai: 聽說你找到工作了，真為你開心！本來很想參加你的慶祝會，但公司臨時安排我出差，實在無法推辭。回來後一定補請你吃飯！", "语法准确性: 85\n词汇水平: 90\n篇章连贯: 90\n任务完成度: 90\n总体得分: 88"),
#     ("HSK: 3 Esai: 我不能去因為我媽媽生病了。我要照顧她。對不起。祝你找到好工作。", "语法准确性: 55\n词汇水平: 60\n篇章连贯: 65\n任务完成度: 65\n总体得分: 60"),
#     ("HSK: 3 Esai: 恭喜你！我本來要參加你的party，但是我有考試。所以不能去。希望你玩得開心。", "语法准确性: 60\n词汇水平: 65\n篇章连贯: 70\n任务完成度: 70\n总体得分: 65"),
#     ("HSK: 5 Esai: 很抱歉無法出席您的慶祝會。因本人需陪同家人前往醫院進行例行檢查，時間衝突，深感遺憾。謹祝新職順遂，鵬程萬里！", "语法准确性: 90\n词汇水平: 95\n篇章连贯: 90\n任务完成度: 95\n总体得分: 92"),
#     ("HSK: 2 Esai: 我想去但是不能。因為我有約會。對不起。", "语法准确性: 45\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 48"),
#     ("HSK: 4 Esai: 祝賀你獲得新職位！因已預訂機票回鄉探親，無法參與慶祝活動，萬分抱歉。期待日後相聚！", "语法准确性: 85\n词汇水平: 90\n篇章连贯: 85\n任务完成度: 90\n总体得分: 87"),
#     ("HSK: 2 Esai: 我不能去，因為我睡覺。", "语法准确性: 30\n词汇水平: 40\n篇章连贯: 40\n任务完成度: 40\n总体得分: 35"),
#     ("HSK: 3 Esai: 聽說你找到工作，真好！可惜我那天要加班，不能去你的慶祝會。下次我請你吃飯！", "语法准确性: 70\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 75"),
#     ("HSK: 4 Esai: 恭喜！因個人行程衝突，無法出席，敬請見諒。祝事業蒸蒸日上！", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82"),
#     ("HSK: 2 Esai: 我不能去，因為我忘記帶錢。", "语法准确性: 35\n词汇水平: 45\n篇章连贯: 45\n任务完成度: 45\n总体得分: 40"),
#     ("HSK: 4 Esai: 很高興聽到你找到工作！但很遺憾，我已答應家人那天去掃墓，無法參加。祝你新工作一切順利！", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 85\n任务完成度: 85\n总体得分: 83"),
#     ("HSK: 2 Esai: 對不起，我去不了。因為我有事。再見。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 45"),
#     ("HSK: 4 Esai: 恭喜你！因需參加國際會議，無法出席慶祝會，深表歉意。祝前程似錦！", "语法准确性: 85\n词汇水平: 90\n篇章连贯: 85\n任务完成度: 85\n总体得分: 86"),
#     ("HSK: 2 Esai: 我想去，但我媽媽不讓我。", "语法准确性: 45\n词汇水平: 55\n篇章连贯: 55\n任务完成度: 55\n总体得分: 50"),
#     ("HSK: 3 Esai: 祝你新工作順利！因已預約牙醫，無法參加慶祝會，抱歉！", "语法准确性: 65\n词汇水平: 75\n篇章连贯: 75\n任务完成度: 70\n总体得分: 70"),
#     ("HSK: 3 Esai: 聽說你找到工作，太棒了！但我那天要帶狗去看獸醫，不能去。對不起！", "语法准确性: 60\n词汇水平: 65\n篇章连贯: 65\n任务完成度: 60\n总体得分: 62"),
#     ("HSK: 4 Esai: 恭喜！因航班延誤滯留機場，無法趕上慶祝會，萬分抱歉。祝一切順利！", "语法准确性: 75\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 78"),
#     ("HSK: 2 Esai: 我不能去，因為我討厭人多。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 44"),
#     ("HSK: 4 Esai: 祝賀你！因需照顧生病的家人，無法出席，深感抱歉。祝新工作順心如意！", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 85\n任务完成度: 85\n总体得分: 84"),
#     ("HSK: 2 Esai: 對不起，我不能去。因為我有遊戲要打。", "语法准确性: 30\n词汇水平: 45\n篇章连贯: 45\n任务完成度: 45\n总体得分: 38"),
#     ("HSK: 3 Esai: 很高興你找到工作！因已承諾協助朋友搬家，無法參加慶祝會，抱歉。期待下次相聚！", "语法准确性: 70\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 76"),
#     ("HSK: 5 Esai: 恭喜！因宗教節日需參與家庭儀式，無法出席，敬請見諒。祝事業興隆！", "语法准确性: 85\n词汇水平: 90\n篇章连贯: 90\n任务完成度: 90\n总体得分: 89"),
#     ("HSK: 2 Esai: 我不能去，因為我懶得去。", "语法准确性: 25\n词汇水平: 35\n篇章连贯: 40\n任务完成度: 40\n总体得分: 30"),
#     ("HSK: 4 Esai: 祝你新工作一切順利！因需參加親戚婚禮，無法出席慶祝會，深感遺憾。", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 80\n总体得分: 81"),
#     ("HSK: 3 Esai: 對不起，我有約會，不能去。", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 60\n任务完成度: 60\n总体得分: 55"),
#     ("HSK: 3 Esai: 恭喜找到工作！因突發家庭緊急事件，無法參加，萬分抱歉。祝好！", "语法准确性: 70\n词汇水平: 75\n篇章连贯: 75\n任务完成度: 75\n总体得分: 74"),
#     ("HSK: 2 Esai: 我不能去，因為我要看電視。", "语法准确性: 30\n词汇水平: 40\n篇章连贯: 45\n任务完成度: 45\n总体得分: 36"),
#     ("HSK: 4 Esai: 聽說你找到工作，真為你高興！但因需陪同父母就醫，無法出席慶祝會，抱歉。祝事業順利！", "语法准确性: 80\n词汇水平: 90\n篇章连贯: 85\n任务完成度: 85\n总体得分: 85"),
#     ("HSK: 3 Esai: 對不起，我去不了。因為我有事要做。", "语法准确性: 50\n词汇水平: 55\n篇章连贯: 55\n任务完成度: 55\n总体得分: 52"),
#     ("HSK: 3 Esai: 恭喜！因需準備重要考試，無法參加慶祝會，深感抱歉。祝你工作愉快！", "语法准确性: 75\n词汇水平: 80\n篇章连贯: 75\n任务完成度: 80\n总体得分: 77"),
#     ("HSK: 2 Esai: 我不能去，因為我心情不好。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 55\n总体得分: 46"),
#     ("HSK: 4 Esai: 祝賀你獲得新職！因已承諾擔任婚禮伴郎，無法出席，萬分抱歉。祝前程似錦！", "语法准确性: 85\n词汇水平: 90\n篇章连贯: 90\n任务完成度: 85\n总体得分: 88"),
#     ("HSK: 3 Esai: 對不起，我不能去。因為我有課。", "语法准确性: 55\n词汇水平: 65\n篇章连贯: 65\n任务完成度: 60\n总体得分: 60"),
#     ("HSK: 4 Esai: 恭喜！因需照顧新生兒，無法出席慶祝會，深感遺憾。祝一切順利！", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 85\n任务完成度: 80\n总体得分: 83"),
#     ("HSK: 2 Esai: 我不能去，因為我忘記了。", "语法准确性: 45\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 50\n总体得分: 48"),
#     ("HSK: 4 Esai: 很高興你找到工作！因需參加線上國際研討會，無法出席，抱歉。祝事業成功！", "语法准确性: 75\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 80\n总体得分: 80"),
#     ("HSK: 3 Esai: 對不起，我不能去。因為我有約。", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 60\n任务完成度: 55\n总体得分: 54"),
#     ("HSK: 4 Esai: 恭喜！因需進行年度健康檢查，無法出席慶祝會，敬請見諒。祝身體健康，工作順利！", "语法准确性: 75\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 79"),
#     ("HSK: 2 Esai: 我不能去，因為我沒興趣。", "语法准确性: 25\n词汇水平: 35\n篇章连贯: 40\n任务完成度: 40\n总体得分: 32"),
#     ("HSK: 4 Esai: 祝你新工作順利！因需陪同孩子參加學校活動，無法出席，抱歉。期待日後相聚！", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82"),
#     ("HSK: 3 Esai: 對不起，我不能去。因為我累了。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 55\n总体得分: 47"),
#     ("HSK: 5 Esai: 恭喜你找到理想工作！因已預訂家庭旅遊，無法參加慶祝會，萬分抱歉。祝事業蒸蒸日上，萬事如意！", "语法准确性: 85\n词汇水平: 95\n篇章连贯: 90\n任务完成度: 90\n总体得分: 90")
# ]
# MOCK_TRAINING_DATA = SCORING_TRAIN_DATA 
# ERROR_TRAIN_DATA = [
#     ("林美美好朋友，你好！我聽說你最近找到了工作，你真棒！恭喜，恭喜！我替你很高興！我聽說你打算開一個慶祝會。對不起，我要參加，可是沒有空。你開一個慶祝會的時候我不能會參加，是因為我在外國做工作。可惜我不能參加。我回台灣以後，我們應該找時間可以好好地聊聊。我有很多問題要問你。你找到了什麼工作？你什麼時候開始上班？祝身體健康張愛文。", "表达不自然 | 林美美好朋友 | 林美美，你好！ | 直接称呼朋友时，通常将名字和称谓分开\n词语误用/表达不准确 | 我要參加 | 我想參加 | “要参加”表示意愿或计划，与“没有空”矛盾\n语法错误(助动词重复) | 我不能會參加 | 我不能參加 | “不能”已表示否定和能力，无需再加“会”\n表达冗余 | 在外國做工作 | 在外國工作 | “工作”本身可作动词，无需再加“做”\n语法错误/表达冗余 | 找時間可以好好地聊聊 | 找時間好好地聊聊 | “找时间”后直接接动词短语即可\n格式错误/语序不当 | 祝身體健康張 | 祝 身體健康\n張 | 署名应在祝福语之后另起一行"),
#     ("我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係，沒有辦法參加。真的不好意思！下下個禮拜，如果你有空的話，我們一起喝咖啡", "表达不自然 | 因為我回國看我的父母的關係 | 因為我要回國看父母 | “...的關係” 是日語直譯，中文應簡化\n标点缺失 | （句尾） | 句尾加句號 | 中文句子結束需用句號\n格式不完整 | （结尾） | 加“祝好！XXX” | 非正式邀請/回覆也應有基本結尾禮貌格式"),
#     ("祝你找到了工作。最近找到工作很難，可是你容易找到。我很高興！我聽說你下個星期六要開一個慶祝會。可是我不能參加，因為我計畫跟我的父母到花蓮去旅行。我下次跟你見面的時候，請你吃飯。", "时态表达不当 | 祝你找到了工作 | 祝你找到工作！ | “祝”後接未完成或一般狀態，不應用“了”\n用词不准确 | 你容易找到 | 你這麼快就找到了 | “容易找到”聽起來像能力強，而非運氣好\n冗余表达 | 到花蓮去旅行 | 去花蓮旅行 | “去”已表方向，“到...去”重複"),
#     ("你好！我是張愛文。我聽到你找到工作。恭喜恭喜！我很高興你找到了工作！我寫卡是為了恭喜你，還有我要說我不能去慶祝會。我對你很抱歉。我也很想去，可是沒有辦法。是因為慶祝會的日子是我母親的生日。所以那個時候我跟我家人都要去韓國玩。我下個禮拜天才回來。我希望你了解我。我答應你，我回來的時候，我一定要帶送給你的禮物。那下次見面吧。", "句式重复啰嗦 | 我聽到你找到工作。恭喜恭喜！我很高興你找到了工作！ | 聽說你找到工作了，真為你高興！恭喜！ | 兩句意思重複，可合併精簡\n用词不地道 | 我寫卡是為了恭喜你 | 我寫這張卡片是想恭喜你 | “寫卡”口語中不常用\n逻辑连接生硬 | 所以那個時候 | 所以那天 | “那個時候”太口語且模糊"),
#     ("恭喜你剛剛找到了一個工作！我為你真高興。謝謝你邀請我來你慶祝會，但是我不能參加，因為我已經跟我的男朋友約好了一起去旅行。我們在你開晚會的那一天就出去，可是我真希望回來以後再跟你見面。恭喜發財！", "量词冗余 | 一個工作 | 一份工作 | “工作”通常用量詞“份”，而非“個”\n介词误用 | 來你慶祝會 | 參加你的慶祝會 | “來”不能直接接活動名稱\n用词不当（文化） | 恭喜發財 | 祝你工作順利！ | “恭喜發財”多用於春節，不適合就職"),
#     ("我不能去你的派對，因為我生病了。醫生說我需要休息一週。祝你新工作一切順利！", "用词不当 | 派對 | 慶祝會 | “派對”偏口語且西化\n句子过短/衔接弱 | 三句獨立無連接 | 加“由於...因此...” | 段落缺乏邏輯銜接"),
#     ("親愛的美美：恭喜你找到新工作！我很想去你的慶祝會，但那天我正好要參加一個重要的面試，實在無法分身。希望你能理解。等你有空，我們再約！", "格式小瑕疵 | 親愛的美美： | 親愛的美美：\n | 書信開頭後應換行\n用词稍口语 | 分身 | 抽身 | “分身”多用於口語"),
#     ("對不起我不能去。因為我有事情。祝你工作好。", "表达过于模糊 | 我有事情 | 我有重要的家庭安排 | “有事情”太籠統\n祝福语不完整 | 工作好 | 工作順利 | “工作好”不符合中文祝福習慣"),
#     ("聽說你找到工作了，真為你開心！本來很想參加你的慶祝會，但公司臨時安排我出差，實在無法推辭。回來後一定補請你吃飯！", "無明顯錯誤 | - | - | -"),
#     ("我不能去因為我媽媽生病了。我要照顧她。對不起。祝你找到好工作。", "句子碎片化 | 三個短句無連接 | 因為媽媽生病了，我必須在家照顧她，所以無法參加。 | 中文偏好複句\n祝福语时序错误 | 祝你找到好工作 | 祝你工作順利 | 對方已找到工作"),
#     ("恭喜你！我本來要參加你的party，但是我有考試。所以不能去。希望你玩得開心。", "混用中英文 | party | 慶祝會 | 正式中文應避免夾雜英文\n連接詞單一 | 所以 | 因此 | 可多樣化連接方式"),
#     ("很抱歉無法出席您的慶祝會。因本人需陪同家人前往醫院進行例行檢查，時間衝突，深感遺憾。謹祝新職順遂，鵬程萬里！", "無明顯錯誤 | - | - | -"),
#     ("我想去但是不能。因為我有約會。對不起。", "表达幼稚 | 我想去但是不能。因為我有約會。 | 很遺憾因已有私人約會，無法出席。 | 過於口語化\n缺乏禮貌緩衝 | 直接說“不能” | 很遺憾 | 應先表達遺憾"),
#     ("祝賀你獲得新職位！因已預訂機票回鄉探親，無法參與慶祝活動，萬分抱歉。期待日後相聚！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我睡覺。", "理由不恰當 | 因為我睡覺 | 因已有其他安排 | “睡覺”作為理由顯得不尊重\n句子不完整 | 單一句子 | 應包含問候與祝福 | 社交書信需基本結構"),
#     ("聽說你找到工作，真好！可惜我那天要加班，不能去你的慶祝會。下次我請你吃飯！", "用词稍口语 | 真好 | 真為你高興 | “真好”較隨意\n缺少署名 | 無署名 | 加“你的朋友，XXX” | 非正式信件也應有基本署名"),
#     ("恭喜！因個人行程衝突，無法出席，敬請見諒。祝事業蒸蒸日上！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我忘記帶錢。", "理由不恰當 | 因為我忘記帶錢 | 因已有其他安排 | 經濟理由不宜公開\n缺乏禮貌 | 直接陳述 | 應先致歉 | 應以“很抱歉”開頭"),
#     ("很高興聽到你找到工作！但很遺憾，我已答應家人那天去掃墓，無法參加。祝你新工作一切順利！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我去不了。因為我有事。再見。", "表达模糊 | 我有事 | 我有重要安排 | “有事”太籠統\n結尾不當 | 再見 | 祝好 | “再見”用於口語告別"),
#     ("恭喜你！因需參加國際會議，無法出席慶祝會，深表歉意。祝前程似錦！", "無明顯錯誤 | - | - | -"),
#     ("我想去，但我媽媽不讓我。", "表达幼稚 | 我媽媽不讓我 | 家中另有安排 | 成人應避免“媽媽不讓”\n缺乏正式結構 | 無問候無祝福 | 應補充 | 社交書信需基本要素"),
#     ("祝你新工作順利！因已預約牙醫，無法參加慶祝會，抱歉！", "理由稍弱 | 預約牙醫 | 有重要醫療預約 | “牙醫”略顯瑣碎\n缺少情感緩衝 | 直接陳述 | 很遺憾因... | 應先表達遺憾"),
#     ("聽說你找到工作，太棒了！但我那天要帶狗去看獸醫，不能去。對不起！", "理由不正式 | 帶狗去看獸醫 | 有家庭安排 | 個人瑣事不宜作為正式理由\n語氣過於隨意 | 太棒了 | 真為你高興 | 可更正式"),
#     ("恭喜！因航班延誤滯留機場，無法趕上慶祝會，萬分抱歉。祝一切順利！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我討厭人多。", "理由不恰當 | 因為我討厭人多 | 因個人原因無法出席 | 負面情緒不宜直述\n缺乏禮貌 | 無歉意 | 應先致歉 | 應表達遺憾"),
#     ("祝賀你！因需照顧生病的家人，無法出席，深感抱歉。祝新工作順心如意！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我不能去。因為我有遊戲要打。", "理由極不恰當 | 有遊戲要打 | 有其他安排 | 遊戲作為理由極不尊重\n缺乏基本結構 | 無問候無祝福 | 應補充 | 不符合書信規範"),
#     ("很高興你找到工作！因已承諾協助朋友搬家，無法參加慶祝會，抱歉。期待下次相聚！", "用词稍口语 | 幫助朋友搬家 | 協助友人遷居 | 可更正式\n缺少署名 | 無署名 | 應添加 | 非正式信件也需署名"),
#     ("恭喜！因宗教節日需參與家庭儀式，無法出席，敬請見諒。祝事業興隆！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我懶得去。", "態度不尊重 | 因為我懶得去 | 因個人原因無法出席 | “懶得”極不禮貌\n無基本禮貌 | 無問候無祝福 | 應完整結構 | 不符合溝通基本原則"),
#     ("祝你新工作一切順利！因需參加親戚婚禮，無法出席慶祝會，深感遺憾。", "無明顯錯誤 | - | - | -"),
#     ("對不起，我有約會，不能去。", "表达模糊 | 有約會 | 有重要私人安排 | “約會”可能誤解\n缺乏結構 | 無問候無祝福 | 應補充 | 社交書信需基本要素"),
#     ("恭喜找到工作！因突發家庭緊急事件，無法參加，萬分抱歉。祝好！", "表达稍模糊 | 家庭緊急事件 | 家中有急事需處理 | 可稍具體\n祝福语简略 | 祝好 | 祝工作順利 | 可更貼合主題"),
#     ("我不能去，因為我要看電視。", "理由不恰當 | 因為我要看電視 | 因已有其他安排 | 娛樂活動不宜作為正式理由\n缺乏禮貌 | 無歉意 | 應先致歉 | 應表達遺憾"),
#     ("聽說你找到工作，真為你高興！但因需陪同父母就醫，無法出席慶祝會，抱歉。祝事業順利！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我去不了。因為我有事要做。", "表达模糊 | 有事要做 | 有重要安排 | “有事”太籠統\n缺乏細節與情感 | 無具體說明 | 應稍具體 | 可增加誠意"),
#     ("恭喜！因需準備重要考試，無法參加慶祝會，深感抱歉。祝你工作愉快！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我心情不好。", "理由不恰當 | 因為我心情不好 | 因個人原因無法出席 | 情緒問題不宜公開\n缺乏禮貌結構 | 無問候無祝福 | 應補充 | 不符合書信規範"),
#     ("祝賀你獲得新職！因已承諾擔任婚禮伴郎，無法出席，萬分抱歉。祝前程似錦！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我不能去。因為我有課。", "表达稍简略 | 有課 | 有重要的課程安排 | 可稍正式\n缺少祝福 | 無祝福 | 應添加 | 社交書信需祝福語"),
#     ("恭喜！因需照顧新生兒，無法出席慶祝會，深感遺憾。祝一切順利！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我忘記了。", "理由不尊重 | 因為我忘記了 | 因時間衝突 | “忘記”顯得不重視\n缺乏誠意 | 無歉意強度 | 應表達深感抱歉 | 應加強歉意"),
#     ("很高興你找到工作！因需參加線上國際研討會，無法出席，抱歉。祝事業成功！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我不能去。因為我有約。", "表达模糊 | 有約 | 有重要約定 | “有約”太口語且模糊\n缺少細節 | 無具體說明 | 應稍具體 | 可增加誠意"),
#     ("恭喜！因需進行年度健康檢查，無法出席慶祝會，敬請見諒。祝身體健康，工作順利！", "無明顯錯誤 | - | - | -"),
#     ("我不能去，因為我沒興趣。", "態度極不禮貌 | 因為我沒興趣 | 因個人原因無法出席 | 直接表達“沒興趣”極不尊重\n無基本禮貌 | 無問候無祝福 | 應完整結構 | 不符合社交規範"),
#     ("祝你新工作順利！因需陪同孩子參加學校活動，無法出席，抱歉。期待日後相聚！", "無明顯錯誤 | - | - | -"),
#     ("對不起，我不能去。因為我累了。", "理由不正式 | 因為我累了 | 因身體不適需休息 | “累了”太口語\n缺乏禮貌緩衝 | 無遺憾表達 | 應先表達遺憾 | 應以“很遺憾”開頭"),
#     ("恭喜你找到理想工作！因已預訂家庭旅遊，無法參加慶祝會，萬分抱歉。祝事業蒸蒸日上，萬事如意！", "無明顯錯誤 | - | - | -")
# ]

# FEEDBACK_TRAIN_DATA = [
#     ("Esai: 林美美好朋友，你好！我聽說你最近找到了工作...\nSkor: 78\nKesalahan: 6", "作文情感真挚，祝贺之意表达清楚，词汇使用丰富。但有几处语法和表达不够自然，如“我要参加，可是没有空”逻辑矛盾，建议改为“本来想参加”。另外，署名格式需规范。整体表现良好，稍加修改会更出色。（文章情感真挚，祝贺之意表达清楚，词汇使用丰富。但有几处语法和表达不够自然，如“我要参加，可是没有空”逻辑矛盾，建议改为“本来想参加”。另外，署名格式需规范。整体表现良好，稍加修改会更出色。）"),
#     ("Esai: 我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係...\nSkor: 72\nKesalahan: 3", "你的理由表達清楚，但句式受外語影響較明顯，如「...的關係」不符合中文習慣。建議簡化因果表達，並補充結尾禮貌用語。（Alasanmu jelas, tetapi struktur kalimat terpengaruh bahasa asing, seperti frasa '...的關係' yang tidak lazim dalam bahasa Mandarin. Disarankan gunakan ekspresi sebab-akibat yang lebih alami dan tambahkan salam penutup sopan。）"),
#     ("Esai: 祝你找到了工作。最近找到工作很難...\nSkor: 76\nKesalahan: 3", "祝賀語真誠，但「祝你找到了工作」時態不當，應去掉「了」。此外，「容易找到」建議改為「這麼快就找到」更貼切。整體語意流暢，情感自然。（Ucapan selamatmu tulus, tetapi penggunaan '了' setelah '祝' tidak tepat. Selain itu, '容易找到' lebih baik diganti dengan '這麼快就找到' agar lebih sesuai konteks. Secara keseluruhan, esai lancar dan ekspresi emosional alami。）"),
#     ("Esai: 你好！我是張愛文。我聽到你找到工作...\nSkor: 80\nKesalahan: 3", "內容完整、情感誠懇，但部分句子重複（如多次提到「找到工作」）。建議精簡開頭，並使用更地道的表達如「寫這張卡片」。整體表現優秀，具備良好的溝通意識。（Isi lengkap dan tulus, tetapi ada pengulangan (misalnya menyebut '找到工作' berkali-kali). Disarankan memadatkan pembukaan dan gunakan frasa lebih alami seperti '寫這張卡片'. Secara keseluruhan sangat baik dan menunjukkan kesadaran komunikasi yang baik。）"),
#     ("Esai: 恭喜你剛剛找到了一個工作！我為你真高興...\nSkor: 74\nKesalahan: 3", "祝賀之意明確，但「一個工作」應為「一份工作」，「來你慶祝會」搭配不當。結尾「恭喜發財」雖熱情，但用於就職場合稍顯不妥，建議改用「工作順利」。（Ucapan selamat jelas, tetapi '一個工作' seharusnya '一份工作', dan frasa '來你慶祝會' tidak tepat. Penutup '恭喜發財' terlalu spesifik untuk Tahun Baru Imlek; untuk ucapan kerja, lebih baik gunakan '祝你工作順利'.）"),
#     ("Esai: 我不能去你的派對，因為我生病了...\nSkor: 58\nKesalahan: 2", "內容簡潔但過於簡略，缺乏情感表達。建議使用更正式的詞彙如「聚會」，並增加過渡詞使行文流暢。（Isi terlalu singkat dan kurang ekspresi emosional. Gunakan kosakata lebih formal seperti '聚會' dan tambahkan kata penghubung agar alur lebih lancar。）"),
#     ("Esai: 親愛的美美：恭喜你找到新工作！...\nSkor: 85\nKesalahan: 2", "書信格式正確，語氣得體，理由合理。僅需注意書面語與口語的區分，整體表現優秀。（Format surat tepat, nada sopan, alasan masuk akal. Hanya perlu bedakan sedikit antara bahasa lisan dan tulis. Secara keseluruhan sangat baik。）"),
#     ("Esai: 對不起我不能去。因為我有事情...\nSkor: 42\nKesalahan: 2", "表達過於簡略且模糊，缺乏具體理由和誠意。建議說明具體原因，並使用標準祝福語如「工作順利」。（Ekspresi terlalu singkat dan samar, kurang alasan spesifik. Gunakan ucapan selamat standar seperti '工作順利' untuk menunjukkan kesopanan。）"),
#     ("Esai: 聽說你找到工作了，真為你開心！...\nSkor: 88\nKesalahan: 0", "理由充分，語氣真誠，用詞得體，結構完整。是一篇優秀的社交書信。（Alasan masuk akal, nada tulus, pilihan kata tepat, struktur lengkap. Ini adalah contoh surat sosial yang sangat baik。）"),
#     ("Esai: 我不能去因為我媽媽生病了...\nSkor: 60\nKesalahan: 2", "理由合理但表達過於碎片化。建議合併句子，並根據情境調整祝福語為「工作順利」。（Alasan masuk akal, tetapi kalimat terlalu terpotong-potong. Gabungkan menjadi kalimat kompleks dan sesuaikan ucapan selamat dengan situasi。）"),
#     ("Esai: 恭喜你！我本來要參加你的party...\nSkor: 65\nKesalahan: 2", "內容清晰，但應避免使用英文詞如「party」。建議使用純中文詞彙，並豐富連接詞。（Isi jelas, tetapi hindari kata Inggris seperti 'party'. Gunakan kosakata Mandarin murni dan variasikan kata penghubung。）"),
#     ("Esai: 很抱歉無法出席您的慶祝會...\nSkor: 92\nKesalahan: 0", "用詞典雅，格式正式，理由得體，祝福語恰當。展現高階中文書寫能力。（Pilihan kata elegan, format formal, alasan sopan, ucapan selamat tepat. Menunjukkan kemampuan menulis Mandarin tingkat lanjut。）"),
#     ("Esai: 我想去但是不能。因為我有約會...\nSkor: 48\nKesalahan: 2", "表達過於直白且口語化，缺乏書面禮貌。建議使用「很遺憾」開頭，並說明更具體的理由。（Terlalu langsung dan lisan. Gunakan frasa sopan seperti '很遺憾' dan berikan alasan lebih spesifik。）"),
#     ("Esai: 祝賀你獲得新職位！因已預訂機票...\nSkor: 87\nKesalahan: 0", "語言簡潔得體，理由充分，情感真摯。格式與用詞均符合正式書信標準。（Bahasa ringkas namun sopan, alasan jelas, emosi tulus. Memenuhi standar surat formal。）"),
#     ("Esai: 我不能去，因為我睡覺。\nSkor: 35\nKesalahan: 2", "理由不恰當且缺乏基本禮貌結構。社交場合應避免以「睡覺」為藉口，並補充問候與祝福。（Alasan tidak sopan dan struktur surat tidak lengkap. Hindari alasan seperti 'tidur' dalam konteks sosial formal。）"),
#     ("Esai: 聽說你找到工作，真好！可惜我那天...\nSkor: 75\nKesalahan: 2", "語氣友好，理由合理。建議使用更完整的祝福語和署名，使信件更完整。（Nada ramah, alasan masuk akal. Tambahkan salam penutup dan nama pengirim agar surat lebih lengkap。）"),
#     ("Esai: 恭喜！因個人行程衝突...\nSkor: 82\nKesalahan: 0", "簡潔有力，用詞正式，符合商務書信風格。表現成熟穩重。（Ringkas, formal, sesuai gaya surat bisnis. Menunjukkan kedewasaan。）"),
#     ("Esai: 我不能去，因為我忘記帶錢。\nSkor: 40\nKesalahan: 2", "理由不妥且缺乏禮貌。建議使用中性理由如“已有安排”，並以歉意開頭。（Alasan tidak pantas dan kurang sopan. Gunakan alasan netral dan mulai dengan permintaan maaf。）"),
#     ("Esai: 很高興聽到你找到工作！但很遺憾...\nSkor: 83\nKesalahan: 0", "理由文化得體（掃墓），情感真誠，結構完整。展現良好的文化適應能力。（Alasan sesuai budaya (ziarah kubur), emosi tulus, struktur lengkap. Menunjukkan adaptasi budaya yang baik。）"),
#     ("Esai: 對不起，我去不了。因為我有事...\nSkor: 45\nKesalahan: 2", "表達過於簡略且模糊。建議具體化理由，並使用書面結尾如「祝好」。（Terlalu singkat dan samar. Jelaskan alasan dan gunakan penutup tertulis seperti '祝好'。）"),
#     ("Esai: 恭喜你！因需參加國際會議...\nSkor: 86\nKesalahan: 0", "理由正式，用詞典雅，祝福語恰當。適合職場或學術場合。（Alasan formal, pilihan kata elegan, cocok untuk konteks profesional。）"),
#     ("Esai: 我想去，但我媽媽不讓我。\nSkor: 50\nKesalahan: 2", "表達顯得孩子氣。建議使用中性理由如「家中另有安排」，並補充問候與祝福。（Ekspresi terlalu kekanak-kanakan. Gunakan alasan netral dan lengkapi struktur surat。）"),
#     ("Esai: 祝你新工作順利！因已預約牙醫...\nSkor: 70\nKesalahan: 2", "理由合理但稍顯瑣碎。建議泛化為“醫療預約”，並以“很遺憾”開頭以示禮貌。（Alasan masuk akal tapi terlalu detail. Umumkan sebagai 'janji medis' dan awali dengan '很遺憾'。）"),
#     ("Esai: 聽說你找到工作，太棒了！但我那天...\nSkor: 62\nKesalahan: 2", "理由過於個人化且不正式。建議使用更得體的理由如“家庭安排”。（Alasan terlalu pribadi. Gunakan alasan lebih umum dan sopan seperti '安排 keluarga'。）"),
#     ("Esai: 恭喜！因航班延誤滯留機場...\nSkor: 78\nKesalahan: 0", "理由合理且不可抗力，語氣誠懇，結構完整。表現得體。（Alasan masuk akal (force majeure), nada tulus, struktur lengkap。）"),
#     ("Esai: 我不能去，因為我討厭人多。\nSkor: 44\nKesalahan: 2", "理由不妥且顯露負面情緒。建議使用中性表述如「因個人原因」，並表達遺憾。（Alasan tidak pantas dan menunjukkan emosi negatif. Gunakan frasa netral dan ungkapkan penyesalan。）"),
#     ("Esai: 祝賀你！因需照顧生病的家人...\nSkor: 84\nKesalahan: 0", "理由得體，情感真誠，用詞優雅。展現同理心與禮貌。（Alasan sopan, emosi tulus, menunjukkan empati dan kesopanan。）"),
#     ("Esai: 對不起，我不能去。因為我有遊戲要打。\nSkor: 38\nKesalahan: 2", "理由極不恰當且缺乏基本禮貌。遊戲不應作為缺席正式場合的理由。（Alasan sangat tidak pantas. Hindari menyebut 'main game' dalam konteks sosial formal。）"),
#     ("Esai: 很高興你找到工作！因已承諾協助朋友...\nSkor: 76\nKesalahan: 2", "理由合理，語氣友好。建議稍提升用詞正式度，並補充署名。（Alasan masuk akal, nada ramah. Tingkatkan sedikit formalitas dan tambahkan nama。）"),
#     ("Esai: 恭喜！因宗教節日需參與家庭儀式...\nSkor: 89\nKesalahan: 0", "理由文化得體，尊重傳統，用詞正式。展現高度文化適應力。（Alasan sesuai nilai budaya dan agama, menunjukkan adaptasi budaya yang tinggi。）"),
#     ("Esai: 我不能去，因為我懶得去。\nSkor: 30\nKesalahan: 2", "態度極不禮貌且缺乏基本溝通素養。應避免負面表述，並遵循書信基本格式。（Sikap sangat tidak sopan. Hindari ekspresi negatif dan ikuti format dasar komunikasi tertulis。）"),
#     ("Esai: 祝你新工作一切順利！因需參加親戚婚禮...\nSkor: 81\nKesalahan: 0", "理由得體（婚禮），情感真誠，結構完整。表現成熟。（Alasan sesuai budaya (pernikahan), struktur lengkap, menunjukkan kedewasaan。）"),
#     ("Esai: 對不起，我有約會，不能去。\nSkor: 55\nKesalahan: 2", "表達過於簡略。建議說明“重要安排”，並補充問候與祝福。（Terlalu singkat. Jelaskan sebagai '安排 penting' dan lengkapi struktur surat。）"),
#     ("Esai: 恭喜找到工作！因突發家庭緊急事件...\nSkor: 74\nKesalahan: 2", "理由合理，語氣誠懇。建議祝福語更貼合“新工作”主題。（Alasan masuk akal, nada tulus. Sesuaikan ucapan selamat dengan konteks pekerjaan baru。）"),
#     ("Esai: 我不能去，因為我要看電視。\nSkor: 36\nKesalahan: 2", "理由不 pantas untuk acara sosial formal. Gunakan alasan netral dan tunjukkan rasa hormat.（Alasan tidak pantas. Gunakan frasa netral dan tunjukkan kesopanan。）"),
#     ("Esai: 聽說你找到工作，真為你高興！但因需陪同父母...\nSkor: 85\nKesalahan: 0", "理由得體（孝道），情感真誠，用詞得當。展現良好價值觀。（Alasan mencerminkan nilai filial, emosi tulus, pilihan kata tepat。）"),
#     ("Esai: 對不起，我去不了。因為我有事要做。\nSkor: 52\nKesalahan: 2", "表達過於模糊。建議稍具體化理由以增加誠意，如“重要家庭安排”。（Terlalu samar. Berikan sedikit detail seperti '安排 keluarga penting' untuk menunjukkan keseriusan。）"),
#     ("Esai: 恭喜！因需準備重要考試...\nSkor: 77\nKesalahan: 0", "理由合理（考試），語氣誠懇，結構完整。適合學生身份。（Alasan masuk akal (ujian), cocok untuk pelajar, struktur lengkap。）"),
#     ("Esai: 我不能去，因為我心情不好。\nSkor: 46\nKesalahan: 2", "理由涉及個人情緒，不宜在社交場合提及。建議使用中性理由。（Alasan bersifat pribadi dan emosional. Gunakan alasan netral dalam konteks sosial。）"),
#     ("Esai: 祝賀你獲得新職！因已承諾擔任婚禮伴郎...\nSkor: 88\nKesalahan: 0", "理由得體且具責任感，用詞正式，祝福語典雅。表現優秀。（Alasan menunjukkan tanggung jawab, pilihan kata formal, ucapan selamat elegan。）"),
#     ("Esai: 對不起，我不能去。因為我有課。\nSkor: 60\nKesalahan: 2", "理由合理（學生身份），但應補充祝福語，並稍提升正式度。（Alasan masuk akal untuk pelajar, tetapi tambahkan ucapan selamat dan sedikit formalitas。）"),
#     ("Esai: 恭喜！因需照顧新生兒...\nSkor: 83\nKesalahan: 0", "理由溫馨得體，情感真誠，展現家庭價值觀。用詞恰當。（Alasan hangat dan sesuai budaya, menunjukkan nilai keluarga。）"),
#     ("Esai: 我不能去，因為我忘記了。\nSkor: 48\nKesalahan: 2", "“忘記”作為理由顯得不重視對方活動。建議使用“時間衝突”等中性表述。（Mengatakan 'lupa' menunjukkan ketidakhormatan. Gunakan frasa netral seperti 'konflik jadwal'。）"),
#     ("Esai: 很高興你找到工作！因需參加線上國際研討會...\nSkor: 80\nKesalahan: 0", "理由正式且具專業性，適合職場人士。結構完整，語氣得體。（Alasan profesional, cocok untuk konteks kerja, struktur lengkap。）"),
#     ("Esai: 對不起，我不能去。因為我有約。\nSkor: 54\nKesalahan: 2", "表達過於模糊。“有約”應具體化為“重要約定”以增加可信度。（Terlalu samar. '有約' sebaiknya dijelaskan sebagai 'janji penting'。）"),
#     ("Esai: 恭喜！因需進行年度健康檢查...\nSkor: 79\nKesalahan: 0", "理由合理且積極（關注健康），祝福語完整。表現穩重。（Alasan positif (periksa kesehatan), ucapan selamat lengkap, menunjukkan kedewasaan。）"),
#     ("Esai: 我不能去，因為我沒興趣。\nSkor: 32\nKesalahan: 2", "Sikap sangat tidak sopan. Hindari ekspresi negatif seperti 'tidak tertarik'. Gunakan frasa netral dan lengkapi struktur surat.（Sikap sangat tidak sopan. Hindari ekspresi negatif. Gunakan frasa netral dan lengkapi struktur surat。）"),
#     ("Esai: 祝你新工作順利！因需陪同孩子參加學校活動...\nSkor: 82\nKesalahan: 0", "理由得體（keluarga & pendidikan anak），情感真誠， struktur lengkap. Sangat baik.（Alasan sesuai nilai keluarga, emosi tulus, struktur lengkap. Sangat baik。）"),
#     ("Esai: 對不起，我不能去。因為我累了。\nSkor: 47\nKesalahan: 2", "“Lelah” sebagai alasan terlalu informal. Gunakan “kondisi tubuh kurang fit” dan awali dengan rasa penyesalan.（Alasan 'lelah' terlalu informal. Gunakan frasa lebih sopan dan awali dengan '很遺憾'。）"),
#     ("Esai: 恭喜你找到理想工作！因已預訂家庭旅遊...\nSkor: 90\nKesalahan: 0", "Bahasa elegan, alasan masuk akal (keluarga), ucapan selamat lengkap dan berkelas. Sangat baik.（Bahasa elegan, alasan masuk akal, ucapan selamat lengkap dan berkelas. Sangat baik。）")
# ]
SCORING_TRAIN_DATA = [
    ("HSK: 3 Esai: 林美美好朋友，你好！我聽說你最近找到了工作，你真棒！恭喜，恭喜！我替你很高興！我聽說你打算開一個慶祝會。對不起，我要參加，可是沒有空。你開一個慶祝會的時候我不能會參加，是因為我在外國做工作。可惜我不能參加。我回台灣以後，我們應該找時間可以好好地聊聊。我有很多問題要問你。你找到了什麼工作？你什麼時候開始上班？祝身體健康張愛文。", "语法准确性: 65\n词汇水平: 90\n篇章连贯: 85\n任务完成度: 90\n总体得分: 78"),
    ("HSK: 3 Esai: 我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係，沒有辦法參加。真的不好意思！下下個禮拜，如果你有空的話，我們一起喝咖啡", "语法准确性: 60\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 72"),
    ("HSK: 3 Esai: 恭喜你剛剛找到了一個工作！我為你真高興。謝謝你邀請我來你慶祝會，但是我不能參加，因為我已經跟我的男朋友約好了一起去旅行。我們在你開晚會的那一天就出去，可是我真希望回來以後再跟你見面。恭喜發財！", "语法准确性: 68\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 74"),
    ("HSK: 2 Esai: 我不能去你的派對，因為我生病了。醫生說我需要休息一週。祝你新工作一切順利！", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 65\n任务完成度: 60\n总体得分: 58"),
    ("HSK: 2 Esai: 對不起我不能去。因為我有事情。祝你工作好。", "语法准确性: 40\n词汇水平: 40\n篇章连贯: 45\n任务完成度: 45\n总体得分: 42"),
    ("HSK: 3 Esai: 我不能去因為我媽媽生病了。我要照顧她。對不起。祝你找到好工作。", "语法准确性: 55\n词汇水平: 60\n篇章连贯: 65\n任务完成度: 65\n总体得分: 60"),
    ("HSK: 3 Esai: 恭喜你！我本來要參加你的party，但是我有考試。所以不能去。希望你玩得開心。", "语法准确性: 60\n词汇水平: 65\n篇章连贯: 70\n任务完成度: 70\n总体得分: 65"),
    ("HSK: 2 Esai: 我想去但是不能。因為我有約會。對不起。", "语法准确性: 45\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 48"),
    ("HSK: 2 Esai: 我不能去，因為我睡覺。", "语法准确性: 30\n词汇水平: 40\n篇章连贯: 40\n任务完成度: 40\n总体得分: 35"),
    ("HSK: 3 Esai: 聽說你找到工作，真好！可惜我那天要加班，不能去你的慶祝會。下次我請你吃飯！", "语法准确性: 70\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 75"),
    ("HSK: 2 Esai: 我不能去，因為我忘記帶錢。", "语法准确性: 35\n词汇水平: 45\n篇章连贯: 45\n任务完成度: 45\n总体得分: 40"),
    ("HSK: 2 Esai: 對不起，我去不了。因為我有事。再見。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 45"),
    ("HSK: 2 Esai: 我想去，但我媽媽不讓我。", "语法准确性: 45\n词汇水平: 55\n篇章连贯: 55\n任务完成度: 55\n总体得分: 50"),
    ("HSK: 3 Esai: 祝你新工作順利！因已預約牙醫，無法參加慶祝會，抱歉！", "语法准确性: 65\n词汇水平: 75\n篇章连贯: 75\n任务完成度: 70\n总体得分: 70"),
    ("HSK: 3 Esai: 聽說你找到工作，太棒了！但我那天要帶狗去看獸醫，不能去。對不起！", "语法准确性: 60\n词汇水平: 65\n篇章连贯: 65\n任务完成度: 60\n总体得分: 62"),
    ("HSK: 2 Esai: 我不能去，因為我討厭人多。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 50\n任务完成度: 50\n总体得分: 44"),
    ("HSK: 2 Esai: 對不起，我不能去。因為我有遊戲要打。", "语法准确性: 30\n词汇水平: 45\n篇章连贯: 45\n任务完成度: 45\n总体得分: 38"),
    ("HSK: 3 Esai: 很高興你找到工作！因已承諾協助朋友搬家，無法參加慶祝會，抱歉。期待下次相聚！", "语法准确性: 70\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 75\n总体得分: 76"),
    ("HSK: 2 Esai: 我不能去，因為我懶得去。", "语法准确性: 25\n词汇水平: 35\n篇章连贯: 40\n任务完成度: 40\n总体得分: 30"),
    ("HSK: 3 Esai: 對不起，我有約會，不能去。", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 60\n任务完成度: 60\n总体得分: 55"),
    ("HSK: 3 Esai: 恭喜找到工作！因突發家庭緊急事件，無法參加，萬分抱歉。祝好！", "语法准确性: 70\n词汇水平: 75\n篇章连贯: 75\n任务完成度: 75\n总体得分: 74"),
    ("HSK: 2 Esai: 我不能去，因為我要看電視。", "语法准确性: 30\n词汇水平: 40\n篇章连贯: 45\n任务完成度: 45\n总体得分: 36"),
    ("HSK: 3 Esai: 對不起，我去不了。因為我有事要做。", "语法准确性: 50\n词汇水平: 55\n篇章连贯: 55\n任务完成度: 55\n总体得分: 52"),
    ("HSK: 3 Esai: 恭喜！因需準備重要考試，無法參加慶祝會，深感抱歉。祝你工作愉快！", "语法准确性: 75\n词汇水平: 80\n篇章连贯: 75\n任务完成度: 80\n总体得分: 77"),
    ("HSK: 2 Esai: 我不能去，因為我心情不好。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 55\n总体得分: 46"),
    ("HSK: 3 Esai: 對不起，我不能去。因為我有課。", "语法准确性: 55\n词汇水平: 65\n篇章连贯: 65\n任务完成度: 60\n总体得分: 60"),
    ("HSK: 2 Esai: 我不能去，因為我忘記了。", "语法准确性: 45\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 50\n总体得分: 48"),
    ("HSK: 3 Esai: 對不起，我不能去。因為我有約。", "语法准确性: 50\n词汇水平: 60\n篇章连贯: 60\n任务完成度: 55\n总体得分: 54"),
    ("HSK: 2 Esai: 我不能去，因為我沒興趣。", "语法准确性: 25\n词汇水平: 35\n篇章连贯: 40\n任务完成度: 40\n总体得分: 32"),
    ("HSK: 3 Esai: 對不起，我不能去。因為我累了。", "语法准确性: 40\n词汇水平: 50\n篇章连贯: 55\n任务完成度: 55\n总体得分: 47"),
    ("HSK: 1 Esai: 我很好，謝謝！你呢？", "语法准确性: 90\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 85"),
    ("HSK: 1 Esai: 我叫小明。我十八歲。", "语法准确性: 90\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 82"),
    ("HSK: 2 Esai: 我去學校。我在學校學習中文。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 2 Esai: 我和朋友一起吃飯。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 84"),
    ("HSK: 2 Esai: 我在家看電視。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 81"),
    ("HSK: 2 Esai: 對不起，我遲到了。", "语法准确性: 90\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 85"),
    ("HSK: 2 Esai: 我喜歡吃飯。我不喜歡喝咖啡。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 84"),
    ("HSK: 3 Esai: 因為我有考試，所以不能去。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 3 Esai: 我本來想去，但是我沒空。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 3 Esai: 我要加班，不能去聚會。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 81"),
    ("HSK: 3 Esai: 我約了朋友，不能去。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 81"),
    ("HSK: 3 Esai: 我身體不舒服，要休息。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 80"),
    ("HSK: 3 Esai: 我要幫媽媽做飯。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 3 Esai: 我坐飛機回國看爸爸。", "语法准确性: 80\n词汇水平: 85\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 3 Esai: 我的狗生病了，我很擔心。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 80\n总体得分: 81"),
    ("HSK: 3 Esai: 我忘記帶作業，對不起。", "语法准确性: 75\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 80\n总体得分: 79"),
    ("HSK: 3 Esai: 我想去，但是沒時間。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 85\n总体得分: 82"),
    ("HSK: 3 Esai: 我要去買東西，不能去。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 81"),
    ("HSK: 3 Esai: 我心情不好，不想出去。", "语法准确性: 80\n词汇水平: 80\n篇章连贯: 85\n任务完成度: 80\n总体得分: 81"),
    ("HSK: 3 Esai: 我要去上課，不能玩。", "语法准确性: 85\n词汇水平: 80\n篇章连贯: 80\n任务完成度: 85\n总体得分: 82")
]



ERROR_TRAIN_DATA = [
    ("林美美好朋友，你好！我聽說你最近找到了工作，你真棒！恭喜，恭喜！我替你很高興！我聽說你打算開一個慶祝會。對不起，我要參加，可是沒有空。你開一個慶祝會的時候我不能會參加，是因為我在外國做工作。可惜我不能參加。我回台灣以後，我們應該找時間可以好好地聊聊。我有很多問題要問你。你找到了什麼工作？你什麼時候開始上班？祝身體健康張愛文。", "表达不自然 | 林美美好朋友 | 林美美，你好！ | 直接称呼朋友时，通常将名字和称谓分开\n词语误用/表达不准确 | 我要參加 | 我想參加 | “要参加”表示意愿或计划，与“没有空”矛盾\n语法错误(助动词重复) | 我不能會參加 | 我不能參加 | “不能”已表示否定和能力，无需再加“会”\n表达冗余 | 在外國做工作 | 在外國工作 | “工作”本身可作动词，无需再加“做”\n语法错误/表达冗余 | 找時間可以好好地聊聊 | 找時間好好地聊聊 | “找时间”后直接接动词短语即可\n格式错误/语序不当 | 祝身體健康張 | 祝 身體健康\n張 | 署名应在祝福语之后另起一行"),
    ("我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係，沒有辦法參加。真的不好意思！下下個禮拜，如果你有空的話，我們一起喝咖啡", "表达不自然 | 因為我回國看我的父母的關係 | 因為我要回國看父母 | “...的關係” 是日語直譯，中文應簡化\n标点缺失 | （句尾） | 句尾加句號 | 中文句子結束需用句號\n格式不完整 | （结尾） | 加“祝好！XXX” | 非正式邀請/回覆也應有基本結尾禮貌格式"),
    ("恭喜你剛剛找到了一個工作！我為你真高興。謝謝你邀請我來你慶祝會，但是我不能參加，因為我已經跟我的男朋友約好了一起去旅行。我們在你開晚會的那一天就出去，可是我真希望回來以後再跟你見面。恭喜發財！", "量词冗余 | 一個工作 | 一份工作 | “工作”通常用量詞“份”，而非“個”\n介词误用 | 來你慶祝會 | 參加你的慶祝會 | “來”不能直接接活動名稱\n用词不当（文化） | 恭喜發財 | 祝你工作順利！ | “恭喜發財”多用於春節，不適合就職"),
    ("我不能去你的派對，因為我生病了。醫生說我需要休息一週。祝你新工作一切順利！", "用词不当 | 派對 | 慶祝會 | “派對”偏口語且西化\n句子过短/衔接弱 | 三句獨立無連接 | 加“由於...因此...” | 段落缺乏邏輯銜接"),
    ("對不起我不能去。因為我有事情。祝你工作好。", "表达过于模糊 | 我有事情 | 我有重要的家庭安排 | “有事情”太籠統\n祝福语不完整 | 工作好 | 工作順利 | “工作好”不符合中文祝福習慣"),
    ("我不能去因為我媽媽生病了。我要照顧她。對不起。祝你找到好工作。", "句子碎片化 | 三個短句無連接 | 因為媽媽生病了，我必須在家照顧她，所以無法參加。 | 中文偏好複句\n祝福语时序错误 | 祝你找到好工作 | 祝你工作順利 | 對方已找到工作"),
    ("恭喜你！我本來要參加你的party，但是我有考試。所以不能去。希望你玩得開心。", "混用中英文 | party | 慶祝會 | 正式中文應避免夾雜英文\n連接詞單一 | 所以 | 因此 | 可多樣化連接方式"),
    ("我想去但是不能。因為我有約會。對不起。", "表达幼稚 | 我想去但是不能。因為我有約會。 | 很遺憾因已有私人約會，無法出席。 | 過於口語化\n缺乏禮貌緩衝 | 直接說“不能” | 很遺憾 | 應先表達遺憾"),
    ("我不能去，因為我睡覺。", "理由不恰當 | 因為我睡覺 | 因已有其他安排 | “睡覺”作為理由顯得不尊重\n句子不完整 | 單一句子 | 應包含問候與祝福 | 社交書信需基本結構"),
    ("聽說你找到工作，真好！可惜我那天要加班，不能去你的慶祝會。下次我請你吃飯！", "用词稍口语 | 真好 | 真為你高興 | “真好”較隨意\n缺少署名 | 無署名 | 加“你的朋友，XXX” | 非正式信件也應有基本署名"),
    ("我不能去，因為我忘記帶錢。", "理由不恰當 | 因為我忘記帶錢 | 因已有其他安排 | 經濟理由不宜公開\n缺乏禮貌 | 直接陳述 | 應先致歉 | 應以“很抱歉”開頭"),
    ("對不起，我去不了。因為我有事。再見。", "表达模糊 | 我有事 | 我有重要安排 | “有事”太籠統\n結尾不當 | 再見 | 祝好 | “再見”用於口語告別"),
    ("我想去，但我媽媽不讓我。", "表达幼稚 | 我媽媽不讓我 | 家中另有安排 | 成人應避免“媽媽不讓”\n缺乏正式結構 | 無問候無祝福 | 應補充 | 社交書信需基本要素"),
    ("祝你新工作順利！因已預約牙醫，無法參加慶祝會，抱歉！", "理由稍弱 | 預約牙醫 | 有重要醫療預約 | “牙醫”略顯瑣碎\n缺少情感緩衝 | 直接陳述 | 很遺憾因... | 應先表達遺憾"),
    ("聽說你找到工作，太棒了！但我那天要帶狗去看獸醫，不能去。對不起！", "理由不正式 | 帶狗去看獸醫 | 有家庭安排 | 個人瑣事不宜作為正式理由\n語氣過於隨意 | 太棒了 | 真為你高興 | 可更正式"),
    ("我不能去，因為我討厭人多。", "理由不恰當 | 因為我討厭人多 | 因個人原因無法出席 | 負面情緒不宜直述\n缺乏禮貌 | 無歉意 | 應先致歉 | 應表達遺憾"),
    ("對不起，我不能去。因為我有遊戲要打。", "理由極不恰當 | 有遊戲要打 | 有其他安排 | 遊戲作為理由極不尊重\n缺乏基本結構 | 無問候無祝福 | 應補充 | 不符合書信規範"),
    ("很高興你找到工作！因已承諾協助朋友搬家，無法參加慶祝會，抱歉。期待下次相聚！", "用词稍口语 | 幫助朋友搬家 | 協助友人遷居 | 可更正式\n缺少署名 | 無署名 | 應添加 | 非正式信件也需署名"),
    ("我不能去，因為我懶得去。", "態度不尊重 | 因為我懶得去 | 因個人原因無法出席 | “懶得”極不禮貌\n無基本禮貌 | 無問候無祝福 | 應完整結構 | 不符合溝通基本原則"),
    ("對不起，我有約會，不能去。", "表达模糊 | 有約會 | 有重要私人安排 | “約會”可能誤解\n缺乏結構 | 無問候無祝福 | 應補充 | 社交書信需基本要素"),
    ("恭喜找到工作！因突發家庭緊急事件，無法參加，萬分抱歉。祝好！", "表达稍模糊 | 家庭緊急事件 | 家中有急事需處理 | 可稍具體\n祝福语简略 | 祝好 | 祝工作順利 | 可更貼合主題"),
    ("我不能去，因為我要看電視。", "理由不恰當 | 因為我要看電視 | 因已有其他安排 | 娛樂活動不宜作為正式理由\n缺乏禮貌 | 無歉意 | 應先致歉 | 應表達遺憾"),
    ("對不起，我去不了。因為我有事要做。", "表达模糊 | 有事要做 | 有重要安排 | “有事”太籠統\n缺乏細節與情感 | 無具體說明 | 應稍具體 | 可增加誠意"),
    ("恭喜！因需準備重要考試，無法參加慶祝會，深感抱歉。祝你工作愉快！", "無明顯錯誤 | - | - | -"),
    ("我不能去，因為我心情不好。", "理由不恰當 | 因為我心情不好 | 因個人原因無法出席 | 情緒問題不宜公開\n缺乏禮貌結構 | 無問候無祝福 | 應補充 | 不符合書信規範"),
    ("對不起，我不能去。因為我有課。", "表达稍简略 | 有課 | 有重要的課程安排 | 可稍正式\n缺少祝福 | 無祝福 | 應添加 | 社交書信需祝福語"),
    ("我不能去，因為我忘記了。", "理由不尊重 | 因為我忘記了 | 因時間衝突 | “忘記”顯得不重視\n缺乏誠意 | 無歉意強度 | 應表達深感抱歉 | 應加強歉意"),
    ("對不起，我不能去。因為我有約。", "表达模糊 | 有約 | 有重要約定 | “有約”太口語且模糊\n缺少細節 | 無具體說明 | 應稍具體 | 可增加誠意"),
    ("我不能去，因為我沒興趣。", "態度極不禮貌 | 因為我沒興趣 | 因個人原因無法出席 | 直接表達“沒興趣”極不尊重\n無基本禮貌 | 無問候無祝福 | 應完整結構 | 不符合社交規範"),
    ("對不起，我不能去。因為我累了。", "理由不正式 | 因為我累了 | 因身體不適需休息 | “累了”太口語\n缺乏禮貌緩衝 | 無遺憾表達 | 應先表達遺憾 | 應以“很遺憾”開頭"),
    ("我很好，謝謝！你呢？", "無明顯錯誤 | - | - | -"),
    ("我叫小明。我十八歲。", "無明顯錯誤 | - | - | -"),
    ("我去學校。我在學校學習中文。", "句子衔接稍弱 | 兩句獨立 | 可加“因為”或“所以” | 但對HSK 2 masih dapat diterima"),
    ("我和朋友一起吃飯。", "無明顯錯誤 | - | - | -"),
    ("我在家看電視。", "無明顯錯誤 | - | - | -"),
    ("對不起，我遲到了。", "無明顯錯誤 | - | - | -"),
    ("我喜歡吃飯。我不喜歡喝咖啡。", "無明顯錯誤 | - | - | -"),
    ("因為我有考試，所以不能去。", "無明顯錯誤 | - | - | -"),
    ("我本來想去，但是我沒空。", "無明顯錯誤 | - | - | -"),
    ("我要加班，不能去聚會。", "用词稍正式 | 聚會 | 慶祝會 | Tapi masih dalam jangkauan HSK 3"),
    ("我約了朋友，不能去。", "表達稍模糊 | 朋友 | (可 diterima di HSK 3) | Tidak perlu koreksi"),
    ("我身體不舒服，要休息。", "無明顯錯誤 | - | - | -"),
    ("我要幫媽媽做飯。", "無明顯錯誤 | - | - | -"),
    ("我坐飛機回國看爸爸。", "無明顯錯誤 | - | - | -"),
    ("我的狗生病了，我很擔心。", "無明顯錯誤 | - | - | -"),
    ("我忘記帶作業，對不起。", "無明顯錯誤 | - | - | -"),
    ("我想去，但是沒時間。", "無明顯錯誤 | - | - | -"),
    ("我要去買東西，不能去。", "表達稍模糊 | 不能去 | 不能去（那裡） | Tapi masih alami untuk HSK 3"),
    ("我心情不好，不想出去。", "無明顯錯誤 | - | - | -"),
    ("我要去上課，不能玩。", "無明顯錯誤 | - | - | -")
]

FEEDBACK_TRAIN_DATA = [
    ("Esai: 林美美好朋友，你好！我聽說你最近找到了工作...\nSkor: 78\nKesalahan: 6", "作文情感真挚，祝贺之意表达清楚，词汇使用丰富。但有几处语法和表达不够自然，如“我要参加，可是没有空”逻辑矛盾，建议改为“本来想参加”。另外，署名格式需规范。整体表现良好，稍加修改会更出色。（文章情感真挚，祝贺之意表达清楚，词汇使用丰富。但有几处语法和表达不够自然，如“我要参加，可是没有空”逻辑矛盾，建议改为“本来想参加”。另外，署名格式需规范。整体表现良好，稍加修改会更出色。）"),
    ("Esai: 我下個禮拜的慶祝會去不了。因為我回國看我的父母的關係...\nSkor: 72\nKesalahan: 3", "你的理由表達清楚，但句式受外語影響較明顯，如「...的關係」不符合中文習慣。建議簡化因果表達，並補充結尾禮貌用語。（Alasanmu jelas, tetapi struktur kalimat terpengaruh bahasa asing, seperti frasa '...的關係' yang tidak lazim dalam bahasa Mandarin. Disarankan gunakan ekspresi sebab-akibat yang lebih alami dan tambahkan salam penutup sopan。）"),
    ("Esai: 恭喜你剛剛找到了一個工作！我為你真高興...\nSkor: 74\nKesalahan: 3", "祝賀之意明確，但「一個工作」應為「一份工作」，「來你慶祝會」搭配不當。結尾「恭喜發財」雖熱情，但用於就職場合稍顯不妥，建議改用「工作順利」。（Ucapan selamat jelas, tetapi '一個工作' seharusnya '一份工作', dan frasa '來你慶祝會' tidak tepat. Penutup '恭喜發財' terlalu spesifik untuk Tahun Baru Imlek; untuk ucapan kerja, lebih baik gunakan '祝你工作順利'.）"),
    ("Esai: 我不能去你的派對，因為我生病了...\nSkor: 58\nKesalahan: 2", "內容簡潔但過於簡略，缺乏情感表達。建議使用更正式的詞彙如「聚會」，並增加過渡詞使行文流暢。（Isi terlalu singkat dan kurang ekspresi emosional. Gunakan kosakata lebih formal seperti '聚會' dan tambahkan kata penghubung agar alur lebih lancar。）"),
    ("Esai: 對不起我不能去。因為我有事情...\nSkor: 42\nKesalahan: 2", "表達過於簡略且模糊，缺乏具體理由和誠意。建議說明具體原因，並使用標準祝福語如「工作順利」。（Ekspresi terlalu singkat dan samar, kurang alasan spesifik. Gunakan ucapan selamat standar seperti '工作順利' untuk menunjukkan kesopanan。）"),
    ("Esai: 我不能去因為我媽媽生病了...\nSkor: 60\nKesalahan: 2", "理由合理但表達過於碎片化。建議合併句子，並根據情境調整祝福語為「工作順利」。（Alasan masuk akal, tetapi kalimat terlalu terpotong-potong. Gabungkan menjadi kalimat kompleks dan sesuaikan ucapan selamat dengan situasi。）"),
    ("Esai: 恭喜你！我本來要參加你的party...\nSkor: 65\nKesalahan: 2", "內容清晰，但應避免使用英文詞如「party」。建議使用純中文詞彙，並豐富連接詞。（Isi jelas, tetapi hindari kata Inggris seperti 'party'. Gunakan kosakata Mandarin murni dan variasikan kata penghubung。）"),
    ("Esai: 我想去但是不能。因為我有約會...\nSkor: 48\nKesalahan: 2", "表達過於直白且口語化，缺乏書面禮貌。建議使用「很遺憾」開頭，並說明更具體的理由。（Terlalu langsung dan lisan. Gunakan frasa sopan seperti '很遺憾' dan berikan alasan lebih spesifik。）"),
    ("Esai: 我不能去，因為我睡覺。\nSkor: 35\nKesalahan: 2", "理由不恰當且缺乏基本禮貌結構。社交場合應避免以「睡覺」為藉口，並補充問候與祝福。（Alasan tidak sopan dan struktur surat tidak lengkap. Hindari alasan seperti 'tidur' dalam konteks sosial formal。）"),
    ("Esai: 聽說你找到工作，真好！可惜我那天...\nSkor: 75\nKesalahan: 2", "語氣友好，理由合理。建議使用更完整的祝福語和署名，使信件更完整。（Nada ramah, alasan masuk akal. Tambahkan salam penutup dan nama pengirim agar surat lebih lengkap。）"),
    ("Esai: 我不能去，因為我忘記帶錢。\nSkor: 40\nKesalahan: 2", "理由不妥且缺乏禮貌。建議使用中性理由如“已有安排”，並以歉意開頭。（Alasan tidak pantas dan kurang sopan. Gunakan alasan netral dan mulai dengan permintaan maaf。）"),
    ("Esai: 對不起，我去不了。因為我有事...\nSkor: 45\nKesalahan: 2", "表達過於簡略且模糊。建議具體化理由，並使用書面結尾如「祝好」。（Terlalu singkat dan samar. Jelaskan alasan dan gunakan penutup tertulis seperti '祝好'。）"),
    ("Esai: 我想去，但我媽媽不讓我。\nSkor: 50\nKesalahan: 2", "表達顯得孩子氣。建議使用中性理由如「家中另有安排」，並補充問候與祝福。（Ekspresi terlalu kekanak-kanakan. Gunakan alasan netral dan lengkapi struktur surat。）"),
    ("Esai: 祝你新工作順利！因已預約牙醫...\nSkor: 70\nKesalahan: 2", "理由合理但稍顯瑣碎。建議泛化為“醫療預約”，並以“很遺憾”開頭以示禮貌。（Alasan masuk akal tapi terlalu detail. Umumkan sebagai 'janji medis' dan awali dengan '很遺憾'。）"),
    ("Esai: 聽說你找到工作，太棒了！但我那天...\nSkor: 62\nKesalahan: 2", "理由過於個人化且不正式。建議使用更得體的理由如“家庭安排”。（Alasan terlalu pribadi. Gunakan alasan lebih umum dan sopan seperti '安排 keluarga'。）"),
    ("Esai: 我不能去，因為我討厭人多。\nSkor: 44\nKesalahan: 2", "理由不妥且顯露負面情緒。建議使用中性表述如「因個人原因」，並表達遺憾。（Alasan tidak pantas dan menunjukkan emosi negatif. Gunakan frasa netral dan ungkapkan penyesalan。）"),
    ("Esai: 對不起，我不能去。因為我有遊戲要打。\nSkor: 38\nKesalahan: 2", "理由極不恰當且缺乏基本禮貌。遊戲不應作為缺席正式場合的理由。（Alasan sangat tidak pantas. Hindari menyebut 'main game' dalam konteks sosial formal。）"),
    ("Esai: 很高興你找到工作！因已承諾協助朋友...\nSkor: 76\nKesalahan: 2", "理由合理，語氣友好。建議稍提升用詞正式度，並補充署名。（Alasan masuk akal, nada ramah. Tingkatkan sedikit formalitas dan tambahkan nama。）"),
    ("Esai: 我不能去，因為我懶得去。\nSkor: 30\nKesalahan: 2", "態度極不禮貌且缺乏基本溝通素養。應避免負面表述，並遵循書信基本格式。（Sikap sangat tidak sopan. Hindari ekspresi negatif dan ikuti format dasar komunikasi tertulis。）"),
    ("Esai: 對不起，我有約會，不能去。\nSkor: 55\nKesalahan: 2", "表達過於簡略。建議說明“重要安排”，並補充問候與祝福。（Terlalu singkat. Jelaskan sebagai '安排 penting' dan lengkapi struktur surat。）"),
    ("Esai: 恭喜找到工作！因突發家庭緊急事件...\nSkor: 74\nKesalahan: 2", "理由合理，語氣誠懇。建議祝福語更貼合“新工作”主題。（Alasan masuk akal, nada tulus. Sesuaikan ucapan selamat dengan konteks pekerjaan baru。）"),
    ("Esai: 我不能去，因為我要看電視。\nSkor: 36\nKesalahan: 2", "理由 tidak pantas untuk acara sosial formal. Gunakan alasan netral dan tunjukkan rasa hormat.（Alasan tidak pantas. Gunakan frasa netral dan tunjukkan kesopanan。）"),
    ("Esai: 對不起，我去不了。因為我有事要做。\nSkor: 52\nKesalahan: 2", "表達過於模糊。建議稍具體化理由以增加誠意，如“重要家庭安排”。（Terlalu samar. Berikan sedikit detail seperti '安排 keluarga penting' untuk menunjukkan keseriusan。）"),
    ("Esai: 恭喜！因需準備重要考試...\nSkor: 77\nKesalahan: 0", "理由合理（考試），語氣誠懇，結構完整。適合學生身份。（Alasan masuk akal (ujian), cocok untuk pelajar, struktur lengkap。）"),
    ("Esai: 我不能去，因為我心情不好。\nSkor: 46\nKesalahan: 2", "理由涉及個人情緒，不宜在社交場合提及。建議使用中性理由。（Alasan bersifat pribadi dan emosional. Gunakan alasan netral dalam konteks sosial。）"),
    ("Esai: 對不起，我不能去。因為我有課。\nSkor: 60\nKesalahan: 2", "理由合理（學生身份），但應補充祝福語，並稍提升正式度。（Alasan masuk akal untuk pelajar, tetapi tambahkan ucapan selamat dan sedikit formalitas。）"),
    ("Esai: 我不能去，因為我忘記了。\nSkor: 48\nKesalahan: 2", "“忘記”作為理由顯得不重視對方活動。建議使用“時間衝突”等中性表述。（Mengatakan 'lupa' menunjukkan ketidakhormatan. Gunakan frasa netral seperti 'konflik jadwal'。）"),
    ("Esai: 對不起，我不能去。因為我有約。\nSkor: 54\nKesalahan: 2", "表達過於模糊。“有約”應具體化為“重要約定”以增加可信度。（Terlalu samar. '有約' sebaiknya dijelaskan sebagai 'janji penting'。）"),
    ("Esai: 我不能去，因為我沒興趣。\nSkor: 32\nKesalahan: 2", "Sikap sangat tidak sopan. Hindari ekspresi negatif seperti 'tidak tertarik'. Gunakan frasa netral dan lengkapi struktur surat.（Sikap sangat tidak sopan. Hindari ekspresi negatif. Gunakan frasa netral dan lengkapi struktur surat。）"),
    ("Esai: 對不起，我不能去。因為我累了。\nSkor: 47\nKesalahan: 2", "“Lelah” sebagai alasan terlalu informal. Gunakan “kondisi tubuh kurang fit” dan awali dengan rasa penyesalan.（Alasan 'lelah' terlalu informal. Gunakan frasa lebih sopan dan awali dengan '很遺憾'。）"),
    ("Esai: 我很好，謝謝！你呢？\nSkor: 85\nKesalahan: 0", "Kalimat sederhana, sopan, dan interaktif. Sangat sesuai untuk HSK 1.（Kalimat sederhana, sopan, dan interaktif. Sangat sesuai untuk HSK 1。）"),
    ("Esai: 我叫小明。我十八歲。\nSkor: 82\nKesalahan: 0", "Perkenalan diri yang jelas dan akurat. Struktur kalimat HSK 1 sempurna.（Perkenalan diri yang jelas dan akurat. Struktur kalimat HSK 1 sempurna。）"),
    ("Esai: 我去學校。我在學校學習中文。\nSkor: 82\nKesalahan: 1", "Aktivitas harian dijelaskan dengan baik. Bisa digabung jadi satu kalimat untuk lebih lancar.（Aktivitas harian dijelaskan dengan baik. Bisa digabung jadi satu kalimat untuk lebih lancar。）"),
    ("Esai: 我和朋友一起吃飯。\nSkor: 84\nKesalahan: 0", "Kalimat lengkap, konteks sosial tepat, sesuai HSK 2.（Kalimat lengkap, konteks sosial tepat, sesuai HSK 2。）"),
    ("Esai: 我在家看電視。\nSkor: 81\nKesalahan: 0", "Deskripsi aktivitas rumah yang alami dan sesuai level.（Deskripsi aktivitas rumah yang alami dan sesuai level。）"),
    ("Esai: 對不起，我遲到了。\nSkor: 85\nKesalahan: 0", "Ungkapan sopan yang umum dan tepat.（Ungkapan sopan yang umum dan tepat。）"),
    ("Esai: 我喜歡吃飯。我不喜歡喝咖啡。\nSkor: 84\nKesalahan: 0", "Kontras preferensi diungkapkan dengan jelas dan alami.（Kontras preferensi diungkapkan dengan jelas dan alami。）"),
    ("Esai: 因為我有考試，所以不能去。\nSkor: 82\nKesalahan: 0", "Penggunaan konjungsi '因為...所以...' tepat untuk HSK 3.（Penggunaan konjungsi '因為...所以...' tepat untuk HSK 3。）"),
    ("Esai: 我本來想去，但是我沒空。\nSkor: 82\nKesalahan: 0", "Ekspresi niat dan alasan ditampilkan dengan baik.（Ekspresi niat dan alasan ditampilkan dengan baik。）"),
    ("Esai: 我要加班，不能去聚會。\nSkor: 81\nKesalahan: 0", "Alasan umum dan relevan, kosakata sesuai HSK 3.（Alasan umum dan relevan, kosakata sesuai HSK 3。）"),
    ("Esai: 我約了朋友，不能去。\nSkor: 81\nKesalahan: 0", "Kalimat ringkas tapi jelas.（Kalimat ringkas tapi jelas。）"),
    ("Esai: 我身體不舒服，要休息。\nSkor: 80\nKesalahan: 0", "Alasan kesehatan yang wajar dan sopan.（Alasan kesehatan yang wajar dan sopan。）"),
    ("Esai: 我要幫媽媽做飯。\nSkor: 82\nKesalahan: 0", "Menunjukkan nilai keluarga, sesuai konteks HSK 3.（Menunjukkan nilai keluarga, sesuai konteks HSK 3。）"),
    ("Esai: 我坐飛機回國看爸爸。\nSkor: 82\nKesalahan: 0", "Aktivitas keluarga yang relevan, struktur baik.（Aktivitas keluarga yang relevan, struktur baik。）"),
    ("Esai: 我的狗生病了，我很擔心。\nSkor: 81\nKesalahan: 0", "Ekspresi emosi dan konteks hewan peliharaan sesuai level.（Ekspresi emosi dan konteks hewan peliharaan sesuai level。）"),
    ("Esai: 我忘記帶作業，對不起。\nSkor: 79\nKesalahan: 0", "Permintaan maaf yang relevan untuk konteks pelajar.（Permintaan maaf yang relevan untuk konteks pelajar。）"),
    ("Esai: 我想去，但是沒時間。\nSkor: 82\nKesalahan: 0", "Kalimat kontras yang alami dan umum.（Kalimat kontras yang alami dan umum。）"),
    ("Esai: 我要去買東西，不能去。\nSkor: 81\nKesalahan: 0", "Alasan sederhana dan masuk akal.（Alasan sederhana dan masuk akal。）"),
    ("Esai: 我心情不好，不想出去。\nSkor: 81\nKesalahan: 0", "Ekspresi perasaan pribadi yang wajar untuk HSK 3.（Ekspresi perasaan pribadi yang wajar untuk HSK 3。）"),
    ("Esai: 我要去上課，不能玩。\nSkor: 82\nKesalahan: 0", "Prioritas pelajar diungkapkan dengan jelas.（Prioritas pelajar diungkapkan dengan jelas。）")
]
MOCK_TRAINING_DATA = SCORING_TRAIN_DATA


class EssayScoreDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        
        # Gabungkan input dan target untuk Causal LM
        # Model harus belajar memprediksi target_text setelah melihat input_text
        full_text = f"{input_text}\n\n{target_text}"
        
        tokenized = self.tokenizer(full_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        
        # Buat labels, abaikan loss untuk bagian input_text
        input_ids = tokenized.input_ids.squeeze(0)
        
        # Tokenize input_text saja untuk tahu panjangnya
        input_only_tokenized = self.tokenizer(input_text, return_tensors="pt")
        input_length = input_only_tokenized.input_ids.shape[1]
        
        labels = input_ids.clone()
        # Setel token input menjadi -100 agar diabaikan oleh loss function
        labels[:input_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "labels": labels
        }

class PromptTuningModel(nn.Module):
    """
    Wrapper Model yang menerapkan Prompt Tuning sesuai Paper 3.
    Model ini MEMBEKUKAN base_model dan HANYA melatih soft_prompt.
    """
    def __init__(self, model_name: str, prompt_length: int):
        super(PromptTuningModel, self).__init__()
        logger.info(f"Memuat base model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, device_map="cpu", torch_dtype="auto"
        )
        self.config = self.base_model.config
        self.prompt_length = prompt_length
        
        logger.info("Membekukan seluruh parameter base model...")
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        embed_dim = self.config.hidden_size
        self.soft_prompt_embeddings = nn.Parameter(
            torch.randn(1, self.prompt_length, embed_dim, dtype=self.base_model.dtype)
        )
        logger.info(f"Membuat soft prompt. Parameter trainable: {self.soft_prompt_embeddings.numel()}")

        nn.init.xavier_uniform_(self.soft_prompt_embeddings)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Helper untuk mendapatkan embedding dari token ID."""
        
        if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
             return self.base_model.transformer.wte(input_ids)
        elif hasattr(self.base_model, 'get_input_embeddings'):
             return self.base_model.get_input_embeddings()(input_ids)
        else:
             raise NotImplementedError("Tidak dapat menemukan layer input embedding.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass yang menerapkan mekanisme Prompt Tuning dari Paper 3.
        """
        # Dapatkan embedding dari input asli
        inputs_embeds = self.get_input_embeddings(input_ids)
        batch_size = inputs_embeds.shape[0]

        # 3. GABUNGKAN SOFT PROMPT DENGAN EMBEDDING INPUT 
        # [P_e; X_e]
        prompt_embeds = self.soft_prompt_embeddings.expand(batch_size, -1, -1).to(inputs_embeds.device)
        combined_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # 4. BUAT ATTENTION MASK BARU UNTUK PROMPT
        prompt_mask = torch.ones(batch_size, self.prompt_length, dtype=attention_mask.dtype).to(attention_mask.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 5. SESUAIKAN LABELS (tambahkan padding -100 untuk prompt)
        if labels is not None:
            prompt_labels = torch.full((batch_size, self.prompt_length), -100, dtype=labels.dtype).to(labels.device)
            combined_labels = torch.cat([prompt_labels, labels], dim=1)
        else:
            combined_labels = None
            
        # 6. JALANKAN FORWARD PASS HANYA DENGAN EMBEDDING
        # Gradien HANYA akan mengalir ke self.soft_prompt_embeddings [cite: 92, 96]
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        
        return outputs

# ---------------- FUNGSI PELATIHAN ----------------
def train():
    logger.info("="*50)
    logger.info("MEMULAI PELATIHAN 'PROMPT TUNING' (BERDASARKAN PAPER 3)")
    logger.info("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inisialisasi Tokenizer dengan aman
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token  # <-- tambahkan baris ini
        else:
            # Fallback jika eos_token_id pun tidak ada (tidak biasa untuk Qwen)
            tokenizer.pad_token = "<|endoftext|>"

    dataset = EssayScoreDataset(SCORING_TRAIN_DATA, tokenizer)  # pastikan SCORING_TRAIN_DATA sudah didefinisikan
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PromptTuningModel(MODEL_NAME, PROMPT_LENGTH)
    model.to(device)
    model.train()

    optimizer = AdamW([model.soft_prompt_embeddings], lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    logger.info("Memulai training loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} selesai. Rata-rata Loss: {avg_loss:.4f}")

    output_path = "Scoring_soft_prompt.pt"
    torch.save(model.soft_prompt_embeddings, output_path)
    logger.info(f"Pelatihan selesai. Soft prompt disimpan di: {output_path}")
    
    # 7. SIMPAN HANYA PARAMETER PROMPT YANG SUDAH DILATIH
    output_path = "Scoring_soft_prompt.pt"
    # Kita hanya menyimpan parameter prompt, bukan seluruh model [cite: 59]
    torch.save(model.soft_prompt_embeddings.data, output_path)
    logger.info(f"Pelatihan selesai. Soft prompt disimpan di: {output_path}")

if __name__ == "__main__":
    MOCK_TRAINING_DATA = SCORING_TRAIN_DATA
    train()  # ← uncomment ini
    logger.warning("Skrip ini adalah demonstrasi. Jalankan 'train()' dengan data latih nyata.")
    logger.info("Contoh ini hanya melatih 'scoring_prompt'.")
    logger.info("Anda perlu membuat dataset dan melatih 'error_prompt' dan 'feedback_prompt' secara terpisah.")

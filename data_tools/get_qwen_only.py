# -*- coding: utf-8 -*-
import os
import re
import json
import copy
import argparse
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
import librosa
import scipy.signal
from tqdm import tqdm
from qwen_asr import Qwen3ForcedAligner

# 导入文本正则化工具
from data_tools.en_punc import english_text_normalization
from data_tools.zh_punc import chinese_text_normalization

# ==========================================
# 全局启发式声学优化参数 (Heuristic Acoustic Parameters)
# ==========================================
HOP_LENGTH_MS = 10       # 10ms一帧，即100Hz精度
MIN_DUR_FRAMES = 4       # 字的最短持续帧数 (40ms)
SEARCH_RATIO = 0.10      # 边界优化的最大搜索范围 (占当前/相邻字时长的比例)
MIN_SEARCH_MS = 0.03     # 最小搜索范围保底 30ms
THRESHOLD = 0.003        # 判定为静音/低响度的能量阈值


def normalize_text(text: str, language: str) -> str:
    """与 MFA 流程完全一致的文本正则化操作。"""
    if language == "en":
        target_text, _ = english_text_normalization(text)
    elif language == "zh":
        target_text, _ = chinese_text_normalization(text)
        target_text = re.sub(r"[^\w\s]|_", ' ', target_text)           # 标点替换为空格
        target_text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', target_text) # 中文字符两边加空格
        target_text = re.sub(r'\s+', ' ', target_text).strip()         # 多余空格归一化
    else:
        raise ValueError(f"不支持的语言类型: {language}")
    return target_text


def extract_energy_curve(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """提取音频绝对值的平均值 (100Hz池化) 并进行平滑，用于表征响度/能量曲线。"""
    y, sr = librosa.load(audio_path, sr=sr)
    hop_length = int(sr * (HOP_LENGTH_MS / 1000.0))
    
    num_frames = len(y) // hop_length
    frames = np.abs(y[:num_frames * hop_length]).reshape(num_frames, hop_length)
    energy_curve = np.mean(frames, axis=1)  
    
    # Savitzky-Golay 滤波平滑处理
    energy_curve = scipy.signal.savgol_filter(energy_curve, window_length=7, polyorder=2)
    
    return y, energy_curve, sr, hop_length


def find_optimized_frame(
    f_start: int, f_end: int, f_left: int, f_right: int, 
    energy_curve: np.ndarray, low_pos: bool = True, prefer_left: bool = False
) -> Tuple[int, str]:
    """基于帧级能量的边界优化搜索。"""
    f_current = f_start if prefer_left else f_end
    cur = f_current
    max_expand_steps = 1000

    for i in range(max_expand_steps):
        segment = energy_curve[f_left:f_right+1]
        below_thresh_indices = np.where(segment < THRESHOLD)[0]

        if len(below_thresh_indices) > 0:
            idx = below_thresh_indices[-1] if prefer_left else below_thresh_indices[0]
            new_pos = f_left + idx
            if new_pos == cur: return cur, 'thresh'
            cur = new_pos
            
            if prefer_left and cur == f_right:
                f_left = f_right
                f_right = min(f_end, f_left + 2)
                if f_end == f_right: return cur, 'thresh'
            elif (not prefer_left) and cur == f_left:
                f_right = f_left
                f_left = max(f_start, f_right - 2)
                if f_start == f_left: return cur, 'thresh'
            else:
                return cur, 'thresh'
        elif i == 0 and low_pos: 
            return f_left + int(np.argmin(segment)), 'low'
        else: 
            return cur, 'thresh'


def optimize_word_boundaries(
    audio_features: Tuple[np.ndarray, np.ndarray, int, int], 
    words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """利用响度曲线对字级时间戳进行边界自适应优化 (Loudness-based Boundary Optimization)。"""
    if not words:
        return []
        
    opt_words = copy.deepcopy(words)
    y, energy_curve, sr, hop_length = audio_features
    max_frames = len(energy_curve) - 1
    min_search_frames = int((MIN_SEARCH_MS * sr) / hop_length)

    for w in opt_words:
        w['f_start'] = min(max(0, int(w['start'] * sr / hop_length)), max_frames)
        w['f_end'] = min(max(0, int(w['end'] * sr / hop_length)), max_frames)
        if w['f_end'] > w['f_start']:
            w['peak'] = w['f_start'] + int(np.argmax(energy_curve[w['f_start']:w['f_end'] + 1]))
        else:
            w['peak'] = w['f_start']

    for i in range(len(opt_words)):
        w = opt_words[i]
        orig_dur_f = w['f_end'] - w['f_start']
        search_right_f = max(int(orig_dur_f * SEARCH_RATIO), min_search_frames)
        
        # 优化左边界
        if i == 0:
            search_left_f = max(int(orig_dur_f * SEARCH_RATIO), min_search_frames) 
            left_limit = max(0, w['f_start'] - search_left_f) 
            f_low_pos, end_start_equal = True, False
        else:
            w_prev = opt_words[i-1]
            dur_prev_f = w_prev['f_end'] - w_prev['f_start'] 
            search_left_f = max(int(dur_prev_f * SEARCH_RATIO), min_search_frames) 
            if w['f_start'] < w_prev['f_end']: w['f_start'] = w_prev['f_end']
            search_left_f = min(search_left_f, w['f_start'] - w_prev['f_end']) 
            left_limit = max(w['f_start'] - search_left_f, w_prev['peak']) 
            f_low_pos = True
            end_start_equal = (abs(w['f_start'] - w_prev['f_end']) < 0.001)
            
        right_limit = min(w['f_start'] + search_right_f, w['peak']) 
        w['f_start'], find_type = find_optimized_frame(
            w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, low_pos=f_low_pos, prefer_left=True
        )
        
        if i > 0 and end_start_equal and find_type == 'low':
            if w['f_start'] > w_prev['f_end']: w_prev['f_end'] = w['f_start']

        # 优化右边界
        curr_dur_f = max(0, w['f_end'] - w['f_start']) 
        search_left_f = min(max(int(curr_dur_f * SEARCH_RATIO), min_search_frames), curr_dur_f) 
        if i < len(opt_words) - 1:
            w_next = opt_words[i+1]
            dur_next_f = w_next['f_end'] - w_next['f_start']  
            search_right_f = max(int(dur_next_f * SEARCH_RATIO), min_search_frames) 
            right_limit = min(w['f_end'] + search_right_f, w_next['peak']) 
        else:
            search_right_f = max(int(curr_dur_f * SEARCH_RATIO), min_search_frames) 
            right_limit = min(max_frames, w['f_end'] + search_right_f) 
            
        left_limit = max(w['f_end'] - search_left_f, w['peak']) 
        w['f_end'], _ = find_optimized_frame(
            w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, prefer_left=False
        )

    for w in opt_words:
        w['f_end'] = max(w['f_end'], w['f_start'] + MIN_DUR_FRAMES) 
        w['start'] = round(float(w['f_start'] * hop_length / sr), 2)
        w['end'] = round(float(w['f_end'] * hop_length / sr), 2)
        for key in ['f_start', 'f_end', 'peak', 'f_dur']: w.pop(key, None)

    return opt_words


def load_processed_utts(jsonl_path: str) -> set:
    processed = set()
    if not os.path.exists(jsonl_path): return processed
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): processed.add(json.loads(line)['utt'])
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Qwen3FA Alignment & Optimization Pipeline")
    parser.add_argument("--json_path", type=str, required=True, help="输入的原始 JSONL 文件路径")
    parser.add_argument("--wav_folder", type=str, required=True, help="音频文件夹路径")
    parser.add_argument("--output_file", type=str, required=True, help="结果保存的 JSONL 文件路径 (.only.qwen.jsonl)")
    parser.add_argument("--qwen_path", type=str, required=True, help="Qwen3FA 预训练模型路径")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"], help="语言类型")
    args = parser.parse_args()

    lan_map = {"en": "English", "zh": "Chinese"}
    current_lan = lan_map[args.language]
    print(f"✅ 语言设置为: {current_lan}")

    # 1. 加载模型
    print(f"🚀 加载 Qwen3FA 模型: {args.qwen_path}")
    qwen_model = Qwen3ForcedAligner.from_pretrained(
        args.qwen_path, dtype=torch.bfloat16, device_map="cuda:0"
    )

    # 2. 读取数据并过滤已处理
    raw_entries = []
    with open(args.json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): raw_entries.append(json.loads(line))
            
    processed_utts = load_processed_utts(args.output_file)
    tasks = [item for item in raw_entries if item['key'] not in processed_utts]
    print(f"📊 总计 {len(raw_entries)} 条，已处理 {len(processed_utts)} 条，剩余 {len(tasks)} 条待处理。")

    # 3. 主循环
    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for data in tqdm(tasks, desc="单模型对齐与优化"):
            audio_path = os.path.join(args.wav_folder, data['audio'])
            original_text = data['txt']
            
            # --- A. 文本正则化 ---
            try:
                norm_text = normalize_text(original_text, args.language)
                norm_text_list = norm_text.split()
            except Exception as e:
                print(f"⚠️ 文本正则化失败 (utt={data['key']}): {e}")
                continue

            # --- B. Qwen3FA 推理 ---
            try:
                qwen_results = qwen_model.align(audio=audio_path, text=norm_text, language=current_lan)
                qwen_words = [
                    {"word": item.text, "start": round(item.start_time, 2), "end": round(item.end_time, 2)}
                    for item in qwen_results[0].items
                ]
            except Exception as e:
                print(f"⚠️ Qwen3FA 推理失败 (utt={data['key']}): {e}")
                continue

            # --- C. 标签修正 (强制对齐正则化后的文本) ---
            # Qwen3FA 可能会拆分或遗漏某些词，尽量将其 word 字段替换为规范化后的文本
            if len(norm_text_list) == len(qwen_words):
                for i in range(len(qwen_words)):
                    qwen_words[i]['word'] = norm_text_list[i]
            else:
                # 长度不匹配时，保留 Qwen 的原始输出，但不影响后续的声学优化
                pass

            # --- D. 响度边界优化 ---
            try:
                audio_features = extract_energy_curve(audio_path)
                optimized_words = optimize_word_boundaries(audio_features, qwen_words)
            except Exception as e:
                print(f"⚠️ 响度优化失败 (utt={data['key']}): {e}")
                optimized_words = qwen_words # 失败则回退到原始输出

            # --- E. 写入结果 ---
            result_item = {
                "utt": data['key'],
                "audio_path": data['audio'],
                "text": original_text,
                "mfa_text": norm_text,
                "words": optimized_words,
                "confidence": True  # 单模型默认置信度为 True
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            
    print("🎉 单模型提取与优化全部完成！")
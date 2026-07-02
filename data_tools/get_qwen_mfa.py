import torch
import librosa
import json
import numpy as np
import matplotlib.pyplot as plt
import zhplot
from qwen_asr import Qwen3ForcedAligner
from tqdm import tqdm
import copy
import scipy.signal
import sys
sys.path.append("/workspace/project-cpfs-1000109/niesihang/wordvoice")

# 配置参数
HOP_LENGTH_MS = 10 # 10ms一帧，即100Hz精度
MIN_DUR_FRAMES = 4  # 最短 4 帧
SEARCH_RATIO = 0.10 # 扩充当前/相邻字时长的比例
MIN_SEARCH_MS = 0.03 # 最小搜索范围保底 30ms 
THRESHOLD = 0.003 # 能量阈值


def process_audio_abs_mean(audio_path, sr=16000):
    """提取音频绝对值的平均值 (100Hz池化) 并增加前后各1帧的平滑"""
    y, sr = librosa.load(audio_path, sr=sr)
    
    hop_length = int(sr * (HOP_LENGTH_MS / 1000.0))
    # 优化：避免不必要的切片拷贝，直接对不足一帧的部分进行reshape计算
    num_frames = len(y) // hop_length
    frames = np.abs(y[:num_frames * hop_length]).reshape(num_frames, hop_length)
    energy_curve = np.mean(frames, axis=1)  
    
    # 平滑处理
    energy_curve = scipy.signal.savgol_filter(energy_curve, window_length=7, polyorder=2)
    
    return y, energy_curve, sr, hop_length


def find_optimized_frame(f_start, f_end, f_left, f_right, energy_curve, low_pos=True, prefer_left=False):
    """
    基于帧(Frame)的优化寻找。传入的f_left和f_right必须已经是严格限制好的安全边界。
    """
    if prefer_left:
        f_current = f_start
    else:
        f_current = f_end

    # 第一步：在当前边界内寻找第一个低于阈值的位置
    cur = f_current
    max_expand_steps = 1000 # 控制最多扩展次数，防止极端死循环
    step_pos = False
    for i in range(max_expand_steps):
        segment = energy_curve[f_left:f_right+1]
        below_thresh_indices = np.where(segment < THRESHOLD)[0]

        if len(below_thresh_indices) > 0:
            # 存在低于阈值的位置，选择最靠近当前边界的一个
            idx = below_thresh_indices[-1] if prefer_left else below_thresh_indices[0]
            new_pos = f_left + idx
            if new_pos == cur: return cur, 'thresh' # 收敛
            cur = new_pos
            # 如果贴边界 → 扩展
            if prefer_left and cur == f_right:
                # 向右扩展
                f_left = f_right
                f_right = min(f_end, f_left + 2) # 每次扩展2帧
                if f_end == f_right:
                    return cur, 'thresh'
            elif (not prefer_left) and cur == f_left:
                # 向左扩展
                f_right = f_left
                f_left = max(f_start, f_right - 2) # 每次扩展2帧
                if f_start == f_left:
                    return cur, 'thresh'
            else:
                return cur, 'thresh'
        elif i == 0 and low_pos: # 第一次没有找到，直接返回最低点
            return f_left + np.argmin(segment), 'low'
        else: # 扩展后未找到，返回原边界
            return cur, 'thresh'


def process_single_data(audio_eng, record_words, idx=0):
    # 单条数据边界优化
    # 复制备份
    orig_record_words = copy.deepcopy(record_words)
    opt_record_words = copy.deepcopy(record_words)
    y, energy_curve, sr, hop_length = audio_eng # [原始波形曲线, 能量曲线, 采样率, 每帧样本数]
    
    # 计算绝对平均值
    words = opt_record_words # mfa原始标注结果
    max_frames = len(energy_curve) - 1 # 最大帧索引
    min_search_frames = int((MIN_SEARCH_MS * sr) / hop_length) # 最小搜索帧数

    # 预处理：将时间全部转为帧，并计算Peak
    for w in words:
        w['f_start'] = min(max(0, int(w['start'] * sr / hop_length)), max_frames)
        w['f_end'] = min(max(0, int(w['end'] * sr / hop_length)), max_frames)
        
        if w['f_end'] > w['f_start']:
            w['peak'] = w['f_start'] + np.argmax(energy_curve[w['f_start']:w['f_end'] + 1])
        else:
            w['peak'] = w['f_start']

    # 核心优化逻辑 (纯帧操作)
    for i in range(len(words)):
        # 依次优化每个字
        w = words[i]
        orig_dur_f = w['f_end'] - w['f_start'] # 计算字时长
        search_right_f = max(int(orig_dur_f * SEARCH_RATIO), min_search_frames)
        # 优化左边界 (start)
        # 搜索范围计算
        if i == 0:
            # 第一个字特殊处理
            search_left_f = max(int(orig_dur_f * SEARCH_RATIO), min_search_frames) 
            left_limit = max(0, w['f_start'] - search_left_f) 
            f_low_pos = True
            end_start_equal = False
        else:
            # 非第一个字，受前一个字限制
            w_prev = words[i-1]
            dur_prev_f = w_prev['f_end'] - w_prev['f_start'] 
            search_left_f = max(int(dur_prev_f * SEARCH_RATIO), min_search_frames) 
            if w['f_start'] < w_prev['f_end']: 
                w['f_start'] = w_prev['f_end']
            search_left_f = min(search_left_f, w['f_start'] - w_prev['f_end']) 
            left_limit = max(w['f_start'] - search_left_f, w_prev['peak']) 
            if w['f_start'] == w_prev['f_end']: 
                f_low_pos = True
            else: 
                f_low_pos = True
            end_start_equal = (abs(w['f_start'] - w_prev['f_end']) < 0.001)
        right_limit = min(w['f_start'] + search_right_f, w['peak']) 
        w['f_start'], find_type = find_optimized_frame(w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, low_pos=f_low_pos, prefer_left=True)
        
        # 防撕裂：如果前一个字存在且对齐，推齐
        if i > 0 and end_start_equal and find_type == 'low':
            if w['f_start'] > w_prev['f_end']:
                w_prev['f_end'] = w['f_start']

        # 优化右边界 (end)
        curr_dur_f = max(0, w['f_end'] - w['f_start']) 
        search_left_f = min(max(int(curr_dur_f * SEARCH_RATIO), min_search_frames), curr_dur_f) 
        if i < len(words) - 1:
            w_next = words[i+1]
            dur_next_f = w_next['f_end'] - w_next['f_start']  
            search_right_f = max(int(dur_next_f * SEARCH_RATIO), min_search_frames) 
            right_limit = min(w['f_end'] + search_right_f, w_next['peak']) 
        else:
            search_right_f = max(int(curr_dur_f * SEARCH_RATIO), min_search_frames) 
            right_limit = min(max_frames, w['f_end'] + search_right_f) 
        left_limit = max(w['f_end'] - search_left_f, w['peak']) 
        w['f_end'], _ = find_optimized_frame(w['f_start'], w['f_end'], left_limit, right_limit, energy_curve, prefer_left=False)

    # 帧转回时间，并清理多余字段
    for w in words:
        w['f_end'] = max(w['f_end'], w['f_start'] + MIN_DUR_FRAMES) 
        # 【优化点】强制使用 float 转换，解决 Numpy 类型的序列化报错，无需再调用 to_serializable
        w['start'] = round(float(w['f_start'] * hop_length / sr), 2)
        w['end'] = round(float(w['f_end'] * hop_length / sr), 2)
        # 清理多余字段
        w.pop('f_start', None); w.pop('f_end', None); w.pop('peak', None); w.pop('f_dur', None)

    return opt_record_words


def check_timestamp_consistency(
    mfa_words: list, 
    qwen_words: list, 
    duration_tolerance: float = 0.5, 
    boundary_tolerance: float = 0.4,
    min_boundary_abs: float = 0.04
) -> bool:
    """
    对比两个时间戳标注结果的一致性。
    
    参数:
    mfa_words: list, MFA的时间戳字典列表
    qwen_words: list, Qwen的时间戳字典列表
    duration_tolerance: float, 词时长容忍度，例如0.5表示qwen时长只能在mfa时长的 [50%, 150%] 之间 (标准二)
    boundary_tolerance: float, 边界偏移容忍度，例如0.5表示起止边界偏移不能超过mfa时长的 50% (标准三)
    min_boundary_abs: float, 最小绝对边界容忍度(秒)。对于极短的词，按时长比例计算的阈值可能过于苛刻，引入一个最小绝对容错值。
    
    返回:
    bool: 结果一致返回 True，否则返回 False
    """
    for mfa, qwen in zip(mfa_words, qwen_words):
        # 基础校验：对应的词文本应该相同（可选，防错位）
        if mfa['word'] != qwen['word']:
            return False
            
        m_start, m_end = mfa['start'], mfa['end']
        q_start, q_end = qwen['start'], qwen['end']
        
        m_duration = m_end - m_start
        q_duration = q_end - q_start
        
        # 标准一：时间戳不能完全错开（必须有交集）
        # 如果 mfa的end <= qwen的start，或者 qwen的end <= mfa的start，说明完全错开
        if m_end <= q_start or q_end <= m_start:
            return False
            
        # 标准二：qwen的时长不能超过或小于mfa时长的固定百分比
        # 允许的时长范围：m_duration * (1 - duration_tolerance) 到 m_duration * (1 + duration_tolerance)
        min_duration = m_duration * (1 - duration_tolerance)
        max_duration = m_duration * (1 + duration_tolerance)
        if not (min_duration <= q_duration <= max_duration):
            return False
            
        # 标准三：qwen的start和end边界不能超过由mfa字时长决定的阈值
        # 动态阈值 = mfa时长 * boundary_tolerance。
        # 为了防止mfa时长极短（如0.05秒）导致容错率几乎为0，取动态阈值和最小绝对阈值之间的最大值
        boundary_threshold = max(m_duration * boundary_tolerance, min_boundary_abs)
        
        start_diff = abs(q_start - m_start)
        end_diff = abs(q_end - m_end)
        
        if start_diff > boundary_threshold or end_diff > boundary_threshold:
            return False
            
    return True

def load_processed_utts(jsonl_path):
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                processed.add(item['utt'])
    return processed

def merge_apostrophe(words):
    """英文连读合并处理"""
    merged = []
    i = 0
    while i < len(words):
        w = words[i]
        if i + 1 < len(words):
            next_w = words[i + 1]
            if next_w['word'].startswith("'"):
                new_word = w['word'] + next_w['word']
                merged_word = {"word": new_word, "start": w['start'], "end": next_w['end']}
                merged.append(merged_word)
                i += 2
                continue
        merged.append(w)
        i += 1
    return merged

if __name__ == "__main__":
    import os
    import re

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/workspace/project-cpfs-1000109/niesihang/wordvoice/datasets/LEMAS/train/zh/zh000.mfa.jsonl", help="输入的 JSONL 文件路径")
    parser.add_argument("--wav_folder", type=str, default="/workspace/project-cpfs-1000109/niesihang/wordvoice/datasets/LEMAS/train_wav/zh", help="音频文件所在的文件夹路径")
    parser.add_argument("--output_qwen_file", type=str, default="/workspace/project-cpfs-1000109/niesihang/wordvoice/datasets/LEMAS/train/zh/zh000.mfa.qwen.jsonl", help="Qwen 对齐结果保存的 JSONL 文件路径")
    parser.add_argument("--qwen_path", type=str, default="/workspace/project-cpfs-1000109/niesihang/wordvoice/checkpoints/Qwen3FA", help="Qwen 模型路径")
    parser.add_argument("--language", type=str, default="en", help="语言类型，默认为 english") # ["en", "zh"]
    parser.add_argument("--batch_size", type=int, default=500, help="每批处理多少条数据 (推荐 200-500)")
    args = parser.parse_args()

    # 加载模型（全局只需加载一次）
    print(f"✅ Loaded Qwen3FA model from {args.qwen_path}")
    qwen_model = Qwen3ForcedAligner.from_pretrained(
        args.qwen_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    def qwen3fa_infer(audio_paths, texts, lan="Chinese"):
        align_results = qwen_model.align(
            audio=audio_paths,
            text=texts,
            language=lan,
        )
        return align_results

    lan_map = {"en": "English", "zh": "Chinese"}
    if args.language not in lan_map:
        raise ValueError(f"Unsupported language: {args.language}. Supported languages: {list(lan_map.keys())}")
    print(f"✅ Set language to {args.language}")

    # 读取 mfa 结果
    ori_data = []
    with open(args.json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ori_data.append(item)
    print(f"✅ Loaded MFA data from {args.json_path}, total {len(ori_data)} items")

    # 断点续传：加载已处理的 utt，避免重复计算
    print(f"⏳ Checking for already processed utterances in {args.output_qwen_file}...")
    try:
        processed_utts = load_processed_utts(args.output_qwen_file)
    except Exception as e:
        print(f"加载已处理数据失败: {e}")
        processed_utts = set()
    raw_data = []
    for item in tqdm(ori_data, miniters=20000): # xxh
        if item['utt'] in processed_utts:
            continue
        raw_data.append(item)
    del ori_data
    print(f"✅ Total {len(raw_data)} items to process after filtering already processed utterances")

    good_count = 0
    results = []
    with open(args.output_qwen_file, "a", encoding="utf-8") as f_out:
        for idx, data in tqdm(enumerate(raw_data), miniters=10000):
            # 对齐预测
            audio_path = os.path.join(args.wav_folder, data['audio_path'])
            norm_text = data['mfa_text']
            try:
                qwen_words = qwen3fa_infer(audio_path, norm_text, lan=lan_map[args.language])
                qwen_words = [
                    {
                        "word": item.text,
                        "start": round(item.start_time, 2),
                        "end": round(item.end_time, 2)
                    }
                    for item in qwen_words[0].items
                ]
            except Exception as e:
                print(f"⚠️ Qwen3FA inference failed for utt={data['utt']}: {e}")
                continue

            # 边界算法优化
            mfa_text_list = data['mfa_text'].split()
            mfa_words = data['words']


            if args.language == "zh":
                # 优化中文标签
                try:
                    assert len(mfa_text_list) == len(mfa_words) == len(qwen_words), f"文本和标签长度不匹配: {data['utt']}"
                except AssertionError as e:
                    print(f"⚠️ {e}")
                    continue
                for i, w in enumerate(mfa_words):
                    mfa_words[i]['word'] = mfa_text_list[i]
                    qwen_words[i]['word'] = mfa_text_list[i]
            else:
                # 优化英文标签
                if len(mfa_text_list) != len(mfa_words):
                    mfa_words = merge_apostrophe(mfa_words)
                if len(mfa_text_list) != len(mfa_words):
                    print(f"⚠️ WARNING: utt={data['utt']} mfa_text 与 mfa_words 仍不一致，已丢弃")
                    continue
                else:
                    for i, w in enumerate(mfa_words):
                        w['word'] = mfa_text_list[i]
                    data['words'] = mfa_words

            audio_eng = process_audio_abs_mean(audio_path)
            optimized_mfa_words = process_single_data(audio_eng, mfa_words, f'{idx}_mfa')
            optimized_qwen_words = process_single_data(audio_eng, qwen_words, f'{idx}_qwen')

            # 标注结果一致性评估
            confidence = check_timestamp_consistency(optimized_mfa_words, optimized_qwen_words)
            # if confidence == True:
            #     print(idx, data['utt'], "✅ Timestamp consistency check passed")
            good_count += confidence

            f_out.write(json.dumps({
                "utt": data['utt'],
                "audio_path": data['audio_path'],
                "text": data['text'],
                "mfa_text": data['mfa_text'],
                "mfa_words": optimized_mfa_words,
                "qwen_words": optimized_qwen_words,
                "confidence": confidence
            }, ensure_ascii=False) + "\n")
    print(f"✅ Good alignments: {good_count} / {len(raw_data)}, Accuracy: {good_count / len(raw_data) * 100:.2f}%")


# -*- coding: utf-8 -*-
import os
import re
import json
import math
import shutil
import tempfile
import subprocess
import gc
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

import torch
import torchaudio
from tqdm import tqdm

# ==========================================
# 优化 PyTorch 线程设置，防止数据处理时 CPU 爆炸
# ==========================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 假设这些是你项目中的自定义模块
from data_tools.en_punc import english_text_normalization
from data_tools.zh_punc import chinese_text_normalization


class MFAPipeline:
    """
    Montreal Forced Aligner (MFA) 自动化对齐管道。
    负责音频预处理、文本准备、MFA 调用及结果解析。
    """
    def __init__(self, dictionary_path: str, acoustic_model_path: str, batch_size: int = 200, num_jobs: int = 4):
        self.dictionary_path = dictionary_path
        self.model_path = acoustic_model_path
        self.batch_size = batch_size
        self.num_jobs = num_jobs
        self.target_sr = 16000
        
        if not shutil.which("mfa"):
            raise RuntimeError("❌ 找不到 mfa 命令，请先安装 MFA 并将其添加到环境变量中。")

    def align_dataset(self, task_list: List[Dict[str, Any]], output_json_path: str, wav_folder: str):
        """
        主控函数：分批处理整个数据集并追加保存结果。
        """
        total_batches = math.ceil(len(task_list) / self.batch_size)
        print(f"🚀 开始 MFA 对齐，共 {len(task_list)} 条数据，分为 {total_batches} 个批次...")

        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(task_list))
            current_batch = task_list[start_idx:end_idx]
            
            print(f"📦 处理批次 {i+1}/{total_batches} (包含 {len(current_batch)} 条数据)...")
            start_time = time.time()
            
            # 执行单批次对齐
            batch_results = self._process_single_batch(current_batch, wav_folder)

            # 实时追加保存，防止意外中断导致数据丢失
            print(f"💾 正在保存批次结果到 {output_json_path} ...")
            with open(output_json_path, 'a', encoding='utf-8') as f:
                for item in batch_results:
                    json_str = json.dumps(item, ensure_ascii=False)
                    f.write(json_str + '\n')
                    
            print(f"⏱️ 批次 {i+1} 耗时: {time.time() - start_time:.2f} s\n")

    def _process_single_batch(self, batch_data: List[Dict[str, Any]], wav_folder: str) -> List[Dict[str, Any]]:
        """
        处理单个批次：创建临时目录 -> 预处理音视频 -> 运行 MFA -> 解析结果 -> 清理内存
        """
        resamplers = {}
        batch_output = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root_temp = Path(temp_dir)
            input_dir = root_temp / "input"
            output_dir = root_temp / "output"
            input_dir.mkdir()
            output_dir.mkdir()
        
            valid_utts = set()
            
            # --- 1. 准备数据 (音频重采样/防爆音 & 生成 .lab 文件) ---
            with torch.no_grad():
                for item in tqdm(batch_data, desc="预处理音频与文本", miniters=100):
                    utt_id = item['utt']
                    src_audio = Path(wav_folder) / item['audio_path']
                    dest_audio = input_dir / f"{utt_id}.wav"
                    dest_lab = input_dir / f"{utt_id}.lab"
                    
                    if not src_audio.exists():
                        print(f"⚠️ 跳过: 音频文件不存在 {src_audio}")
                        continue

                    # 处理音频
                    success = self._process_single_audio(src_audio, dest_audio, resamplers)
                    if success:
                        # 写入文本
                        with open(dest_lab, 'w', encoding='utf-8') as f:
                            f.write(item['target_text'])
                        valid_utts.add(utt_id)

            # --- 2. 调用 MFA 命令行 ---
            cmd = [
                "mfa", "align",
                str(input_dir),
                self.dictionary_path,
                self.model_path,
                str(output_dir),
                "--output_format", "json", 
                "--clean",
                "--single_speaker",  # 跳过说话人自适应，加速对齐
                "--num_jobs", str(self.num_jobs),
                "--quiet"
            ]
            
            try:
                subprocess.run(cmd, check=True) # 如果要屏蔽输出：stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            except subprocess.CalledProcessError:
                print("⚠️ 本批次 MFA 运行有部分失败（通常是 OOV 或音频极短），尝试读取已生成的结果...")

            # --- 3. 解析 MFA 输出结果 ---
            for item in batch_data:
                utt_id = item['utt']
                result_item = {
                    'utt': item['utt'],
                    'audio_path': item['audio_path'],
                    'text': item['text'],
                    'mfa_text': item['target_text'],
                    'words': []
                }

                if utt_id in valid_utts:
                    json_file = output_dir / f"{utt_id}.json"
                    if json_file.exists():
                        words_alignment = self._parse_mfa_json(json_file, item['target_text'])
                        if words_alignment is not None:
                            result_item['words'] = words_alignment
                        else:
                            print(f"⚠️ 跳过 {utt_id}: 对齐词数与目标文本不匹配。")
                    else:
                        print(f"⚠️ 未生成对齐结果: {utt_id}")

                batch_output.append(result_item)

        # 强制垃圾回收，防止内存泄漏
        gc.collect()
        return batch_output

    def _process_single_audio(self, src_audio: Path, dest_audio: Path, resamplers: Dict[int, Any]) -> bool:
        """
        处理单条音频：重采样至 16kHz、转单声道、音量归一化。
        若失败则尝试软链接或生成静音占位符。
        """
        try:
            audio, sample_rate = torchaudio.load(src_audio, backend="soundfile")
            
            # 转单声道
            if audio.shape[0] > 1:
                audio = audio[0:1, :]
                
            # 重采样
            if sample_rate != self.target_sr:
                if sample_rate not in resamplers:
                    resamplers[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=self.target_sr
                    )
                audio = resamplers[sample_rate](audio)
            
            # 归一化防爆音
            max_val = audio.abs().max()
            if max_val > 1.0: 
                audio /= max_val
                
            torchaudio.save(dest_audio, audio, self.target_sr, encoding="PCM_S", bits_per_sample=16)
            del audio
            return True

        except Exception as e:
            print(f"❌ 音频预处理失败 {src_audio}: {e}")
            try:
                if src_audio.exists():
                    os.symlink(src_audio.resolve(), dest_audio)
                else:
                    raise RuntimeError("源音频不存在")
            except Exception:
                # 极端情况下生成 1/16000s 的静音防止 MFA 崩溃
                dummy_audio = torch.zeros(1, 1)
                torchaudio.save(dest_audio, dummy_audio, self.target_sr, encoding="PCM_S", bits_per_sample=16)
                del dummy_audio
            return False

    def _parse_mfa_json(self, json_file: Path, target_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        解析 MFA 输出的 JSON 文件，处理 <unk> 映射以及英文撇号 (') 导致的词数偏移问题。
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                mfa_data = json.load(f)
            
            words_entries = mfa_data.get('tiers', {}).get('words', {}).get('entries', [])
            target_words = target_text.split()
            
            formatted_words = []
            apostrophe_count = 0  # 记录因撇号被 MFA 拆分的特殊词数量
            
            for idx, entry in enumerate(words_entries):
                start_time, end_time, mfa_word = entry[0], entry[1], entry[2]
                
                if mfa_word.startswith("'"):
                    apostrophe_count += 1
                    
                # 还原被 MFA 标记为 <unk> 的原词
                if mfa_word == '<unk>':
                    mfa_word = target_words[idx - apostrophe_count]
                    
                formatted_words.append({
                    "word": mfa_word,
                    "start": round(start_time, 2),
                    "end": round(end_time, 2)
                })

            # 校验解析后的词数是否与原文本一致
            if (len(target_words) + apostrophe_count) == len(words_entries):
                return formatted_words
            else:
                return None
                
        except Exception as e:
            print(f"❌ 解析 MFA JSON 出错 {json_file.name}: {e}")
            return None


def load_processed_utts(jsonl_path: str) -> Set[str]:
    """加载已处理的 utterance ID，支持断点续传。"""
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                processed.add(item['utt'])
    return processed


def normalize_text(text: str, language: str) -> str:
    """根据语言执行对应的文本正则化操作。"""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WordVoice MFA Alignment Data Pipeline")
    parser.add_argument("--json_path", type=str, required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--wav_folder", type=str, required=True, help="音频文件所在的文件夹路径")
    parser.add_argument("--output_mfa_file", type=str, required=True, help="MFA 对齐结果保存的 JSONL 文件路径")
    parser.add_argument("--mfa_model_path", type=str, required=True, help="MFA 预训练模型根目录路径")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"], help="语言类型")
    parser.add_argument("--batch_size", type=int, default=500, help="每批处理多少条数据 (推荐 200-500)")
    args = parser.parse_args()

    lan = args.language.lower()
    
    # 1. 加载原始数据
    raw_entries = []
    with open(args.json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_entries.append(json.loads(line))

    # 2. 检查断点续传进度
    processed_utts = load_processed_utts(args.output_mfa_file)
    print(f"📊 统计: 已处理 {len(processed_utts)} 条，总计 {len(raw_entries)} 条，剩余 {len(raw_entries) - len(processed_utts)} 条待处理。")

    # 3. 准备待处理任务列表
    alignment_tasks = []
    for item in tqdm(raw_entries, desc="文本正则化", miniters=20000):
        if item['key'] in processed_utts:
            continue
            
        target_text = normalize_text(item['txt'], lan)
        alignment_tasks.append({
            "utt": item['key'],
            "audio_path": item['audio'],
            "text": item['txt'],
            "target_text": target_text
        })

    if not alignment_tasks:
        print("🎉 所有数据已处理完毕！")
        exit(0)

    # 4. 配置 MFA 字典与模型路径
    if lan == "en":
        dict_name = "english_mfa"
        model_name = "english_mfa"
    else:  # zh
        dict_name = "mandarin_china_mfa"
        model_name = "mandarin_mfa"
        
    print(f"✅ 配置完成，使用字典: {dict_name}，声学模型: {model_name}")

    dict_path = f'{args.mfa_model_path}/pretrained_models/dictionary/{dict_name}.dict'
    model_path = f'{args.mfa_model_path}/pretrained_models/acoustic/{model_name}.zip'

    # 5. 实例化并运行 Pipeline
    pipeline = MFAPipeline(
        dictionary_path=dict_path, 
        acoustic_model_path=model_path, 
        batch_size=args.batch_size, 
        num_jobs=16
    )
    
    import time # 导入 time 用于主循环计时
    pipeline.align_dataset(alignment_tasks, args.output_mfa_file, args.wav_folder)
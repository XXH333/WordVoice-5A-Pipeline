#!/bin/bash

# 下载mfa的字典和模型
export MFA_ROOT_DIR="./checkpoints/mfa"
# 下载英文和中文的 发音字典 (dictionary)
echo "Download mfa dictionary..."
mfa model download dictionary english_mfa
mfa model download dictionary mandarin_china_mfa
# 下载英文和中文的 声学模型 (acoustic)
echo "Download mfa acoustic models..."
mfa model download acoustic english_mfa
mfa model download acoustic mandarin_mfa

# 下载Qwen3FA
export HF_ENDPOINT=https://hf-mirror.com
echo "Download qwen3fa..."
python download_qwen3fa.py

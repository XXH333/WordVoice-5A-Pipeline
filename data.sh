#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=".:$PYTHONPATH"
export PKUSEG_HOME=data_tools/spacy

start_stage=3
end_stage=3

json_folder=./test_demo/json_files
wav_root=./test_demo/audio_files
output_folder=./annotation_files
mkdir -p "$output_folder"

# 英文示例
# json_file=en_test.jsonl
# lan=en
# 中文示例
json_file=zh_test.jsonl
lan=zh

json_name=$(basename "$json_file" .jsonl)

if [ $start_stage -le 1 ] && [ $end_stage -ge 1 ]; then
    echo "Stage 1: 获取 $lan MFA 对齐结果"
    json_path=${json_folder}/${json_file}

    echo "🚀 处理 ${json_path}"
    python data_tools/get_mfa.py \
        --json_path $json_path \
        --wav_folder $wav_root \
        --output_mfa_file ${output_folder}/${json_name}.mfa.jsonl \
        --language $lan \
        --mfa_model_path ./checkpoints/mfa \
        --batch_size 20
fi

if [ $start_stage -le 2 ] && [ $end_stage -ge 2 ]; then
    echo "Stage 2: 获取 $lan Qwen 一致性对齐结果"
    echo "🚀 处理 ${json_name}.mfa.jsonl"

    python data_tools/get_qwen_mfa.py \
        --json_path ${output_folder}/${json_name}.mfa.jsonl \
        --wav_folder $wav_root \
        --output_qwen_file ${output_folder}/${json_name}.mfa.qwen.jsonl \
        --qwen_path ./checkpoints/Qwen3FA \
        --language $lan \
        --batch_size 16
fi

# if [ $start_stage -le 2 ] && [ $end_stage -ge 2 ]; then
#     echo "Stage 1-2: [便捷方案] 单一模型提取 $lan 字级时间戳 (仅使用 Qwen3FA + 响度优化)"
#     json_path=${json_folder}/${json_file}

#     echo "🚀 处理 ${json_path}"
#     python data_tools/get_qwen_only.py \
#         --json_path $json_path \
#         --wav_folder $wav_root \
#         --output_file ${output_folder}/${json_name}.only.qwen.jsonl \
#         --qwen_path ./checkpoints/Qwen3FA \
#         --language $lan
# fi

if [ $start_stage -le 3 ] && [ $end_stage -ge 3 ]; then
    echo "Stage 3: 获取 $lan 字级标注结果：音高、能量、声学边界、声调"
    echo "🚀 处理 ${json_name}.mfa.qwen.jsonl"

    python data_tools/get_dsp.py \
        --align_json_path ${output_folder}/${json_name}.mfa.qwen.jsonl \
        --wav_folder $wav_root \
        --output_dsp_file ${output_folder}/${json_name}.mfa.qwen.dsp.jsonl \
        --language $lan \
        --workers 32 \
        --high_quality
fi

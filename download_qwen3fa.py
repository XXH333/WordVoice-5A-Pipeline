from huggingface_hub import snapshot_download

repo_id="Qwen/Qwen3-ForcedAligner-0.6B" # 模型仓库ID
local_path = './checkpoints/Qwen3FA' # 本地下载路径
snapshot_download(
    repo_id=repo_id, 
    local_dir=local_path,
    allow_patterns=["*"], # 指定下载的具体文件夹
    max_workers=2 # 限制并发
)

print(f"模型 '{repo_id}' 已成功下载到 '{local_path}' 目录下。")
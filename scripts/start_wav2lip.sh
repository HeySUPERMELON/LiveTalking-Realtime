#!/usr/bin/env bash
# ============================================================
# start_wav2lip.sh - Wav2Lip 数字人启动脚本
#
# 用法：
#   bash scripts/start_wav2lip.sh [avatar_id] [options]
#
# 示例：
#   bash scripts/start_wav2lip.sh wav2lip_avatar1
#   bash scripts/start_wav2lip.sh wav2lip_avatar1 --tts edgetts
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AVATAR_ID="${1:-wav2lip_avatar1}"
EXTRA_ARGS="${@:2}"

echo "=================================================="
echo " Wav2Lip 数字人启动"
echo "  Avatar: $AVATAR_ID"
echo "  项目目录: $PROJECT_DIR"
echo "=================================================="

cd "$PROJECT_DIR"

# ──── 环境检查 ────────────────────────────────────────────────
CONDA_ENV="nerfstream"
CONDA_PYTHON="/opt/anaconda3/envs/$CONDA_ENV/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
  echo "✗ 找不到 conda 环境: $CONDA_ENV ($CONDA_PYTHON)"
  exit 1
fi

echo ""
echo "[检查] Python 环境..."
$CONDA_PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}')"
$CONDA_PYTHON -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')" 2>/dev/null || true

# ──── Avatar 检查 ─────────────────────────────────────────────
AVATAR_RUNTIME_PATH="data/avatars/$AVATAR_ID"
if [ ! -d "$AVATAR_RUNTIME_PATH" ]; then
  echo ""
  echo "❌ Avatar 不存在: $AVATAR_RUNTIME_PATH"
  echo "   请先运行: bash scripts/gen_avatar_wav2lip.sh <视频路径> $AVATAR_ID"
  exit 1
fi
echo ""
echo "[检查] Avatar $AVATAR_ID 存在 ✅"

# ──── 模型检查 ────────────────────────────────────────────────
if [ ! -f "models/wav2lip.pth" ]; then
  echo "❌ 模型文件不存在: models/wav2lip.pth"
  echo "   请先运行: bash scripts/download_model_wav2lip.sh"
  exit 1
fi
echo "[检查] 模型文件 ✅"

# ──── 启动参数 ─────────────────────────────────────────────────
# Wav2Lip 推理较轻量，batch_size 可以更大
BATCH_SIZE=16
if $CONDA_PYTHON -c "import torch; m=torch.cuda.get_device_properties(0).total_memory//1024//1024//1024; exit(0 if m>=20 else 1)" 2>/dev/null; then
  BATCH_SIZE=24  # 20GB+ GPU
  echo "[自动检测] 大显存 GPU，batch_size=24"
elif $CONDA_PYTHON -c "import torch; exit(0 if not torch.cuda.is_available() else 1)" 2>/dev/null; then
  BATCH_SIZE=4   # MPS/CPU
  echo "[自动检测] MPS/CPU 模式，batch_size=4"
else
  echo "[自动检测] 标准显存 GPU，batch_size=16"
fi

echo ""
echo "=================================================="
echo " 启动配置："
echo "  模型: wav2lip"
echo "  Avatar: $AVATAR_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  传输模式: webrtc"
echo "  监听端口: 8010"
echo "=================================================="
echo ""
echo " 访问地址："
echo "  http://localhost:8010/dashboard.html"
echo ""

# ──── 启动服务 ────────────────────────────────────────────────
PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" $CONDA_PYTHON app.py \
  --model wav2lip \
  --avatar_id "$AVATAR_ID" \
  --batch_size "$BATCH_SIZE" \
  --fps 50 \
  --tts edgetts \
  --llm_type ai_agent \
  --transport webrtc \
  --listenport 8010 \
  $EXTRA_ARGS

#!/usr/bin/env bash
# ============================================================
# start_wav2lip384.sh - Wav2Lip 384 高分辨率数字人启动脚本
#
# 用法：
#   bash scripts/start_wav2lip384.sh [avatar_id] [options]
#
# 示例：
#   bash scripts/start_wav2lip384.sh wav2lip384_avatar1
#   bash scripts/start_wav2lip384.sh wav2lip384_avatar1 --tts edgetts
#
# 说明：
#   面部分辨率 384x384，推理速度比标准 wav2lip 略慢
#   需要 8GB+ 显存，推荐 4090 (24GB)
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AVATAR_ID="${1:-wav2lip384_avatar1}"
EXTRA_ARGS="${@:2}"

echo "=================================================="
echo " Wav2Lip384 高分辨率数字人启动"
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
  echo "   请先运行: bash scripts/gen_avatar_wav2lip384.sh <视频路径> $AVATAR_ID"
  exit 1
fi
echo ""
echo "[检查] Avatar $AVATAR_ID 存在 ✅"

# ──── 模型检查 ────────────────────────────────────────────────
if [ ! -f "models/wav2lip384.pth" ]; then
  echo "❌ 模型文件不存在: models/wav2lip384.pth"
  echo "   请先运行: bash scripts/download_model_wav2lip384.sh"
  exit 1
fi
echo "[检查] 模型文件 ✅"

# ──── 启动参数 ─────────────────────────────────────────────────
# Wav2Lip384 分辨率更高，显存占用更大，batch_size 适当降低
BATCH_SIZE=8
GPU_MEM=$($CONDA_PYTHON -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024)
else:
    print(0)
" 2>/dev/null || echo "0")

if [ "$GPU_MEM" -ge 24 ]; then
  BATCH_SIZE=16  # 24GB GPU (4090)
  echo "[自动检测] 24GB+ GPU，batch_size=16"
elif [ "$GPU_MEM" -ge 16 ]; then
  BATCH_SIZE=8   # 16GB GPU
  echo "[自动检测] 16GB GPU，batch_size=8"
elif [ "$GPU_MEM" -ge 8 ]; then
  BATCH_SIZE=4   # 8GB GPU
  echo "[自动检测] 8GB GPU，batch_size=4"
else
  BATCH_SIZE=2   # MPS/CPU
  echo "[自动检测] MPS/CPU 模式，batch_size=2"
fi

echo ""
echo "=================================================="
echo " 启动配置："
echo "  模型: wav2lip384 (384x384 高分辨率)"
echo "  Avatar: $AVATAR_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  传输模式: webrtc"
echo "  监听端口: 8010"
echo "=================================================="
echo ""
echo " 访问地址："
echo "  http://localhost:8010/dashboard.html"
echo ""
echo " ⚠ 注意：384 分辨率推理比 96 慢约 4-8 倍"
echo "   推荐使用 24GB 显存 GPU (RTX 4090)"
echo ""

# ──── 启动服务 ────────────────────────────────────────────────
PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" $CONDA_PYTHON app.py \
  --model wav2lip384 \
  --avatar_id "$AVATAR_ID" \
  --batch_size "$BATCH_SIZE" \
  --fps 50 \
  --tts edgetts \
  --llm_type ai_agent \
  --transport webrtc \
  --listenport 8010 \
  $EXTRA_ARGS

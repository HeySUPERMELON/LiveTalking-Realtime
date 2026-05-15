#!/usr/bin/env bash
# ============================================================
# start_local.sh - 本机（Mac M4 / MPS）测试启动脚本
#
# 用法：
#   bash scripts/start_local.sh [avatar_id]
#
# 示例：
#   bash scripts/start_local.sh avator_1
# ============================================================

set -e

AVATAR_ID="${1:-avator_1}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo " LiveTalking 本机测试启动"
echo "  设备: Apple M4 MPS"
echo "  Avatar: $AVATAR_ID"
echo "  项目目录: $PROJECT_DIR"
echo "=================================================="

# ──── 检查必要模型文件 ────────────────────────────────────────
echo ""
echo "[检查] 模型文件..."

MISSING=false
check() {
  if [ ! -e "$1" ]; then
    echo "  ❌ 缺少: $1"
    MISSING=true
  else
    echo "  ✅ $1"
  fi
}

check "models/musetalkV15/unet.pth"
check "models/musetalkV15/musetalk.json"
check "models/sd-vae"
check "models/whisper"
check "models/dwpose/dw-ll_ucoco_384.pth"
check "models/face-parse-bisent/79999_iter.pth"

if [ "$MISSING" = true ]; then
  echo ""
  echo "❌ 缺少模型文件，请先运行："
  echo "   bash scripts/download_models.sh"
  exit 1
fi

# ──── 检查 Avatar ─────────────────────────────────────────────
echo ""
echo "[检查] Avatar 数据..."
if [ ! -d "data/avatars/$AVATAR_ID" ]; then
  echo "  ❌ Avatar 不存在: data/avatars/$AVATAR_ID"
  echo ""
  echo "  请先生成 Avatar（需要准备一段人物说话视频）："
  echo "  bash scripts/gen_avatar_v15.sh /path/to/video.mp4 $AVATAR_ID"
  exit 1
fi
echo "  ✅ data/avatars/$AVATAR_ID"

# ──── 使用 conda nerfstream 环境 ──────────────────────────────
CONDA_ENV="nerfstream"
CONDA_PYTHON="/opt/anaconda3/envs/$CONDA_ENV/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
  echo "❌ 找不到 conda 环境: $CONDA_ENV ($CONDA_PYTHON)"
  exit 1
fi

echo "  Python: $CONDA_PYTHON"

# ──── 启动服务 ────────────────────────────────────────────────
echo ""
echo "[启动] 数字人服务（MPS 模式，batch_size=4）..."
echo ""

PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" "$CONDA_PYTHON" app.py \
  --model musetalk \
  --avatar_id "$AVATAR_ID" \
  --batch_size 4 \
  --tts edgetts \
  --REF_FILE "zh-CN-YunxiaNeural" \
  --llm_type ai_agent \
  --transport rtcpush \
  --listenport 8010

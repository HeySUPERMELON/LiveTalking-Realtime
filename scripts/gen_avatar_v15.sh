#!/usr/bin/env bash
# ============================================================
# gen_avatar_v15.sh - Avatar 生成脚本（MuseTalk v1.5）
# 带完整进度条，适合在 MPS/CPU 模式下观察进度
#
# 用法：
#   bash scripts/gen_avatar_v15.sh <视频/图片路径> [avatar_id]
#
# 示例：
#   bash scripts/gen_avatar_v15.sh video2.mp4 exhibition_avatar1
# ============================================================

set -e

# ──── 参数处理 ────────────────────────────────────────────────
if [ -z "$1" ]; then
  echo "用法: $0 <视频或图片路径> [avatar_id]"
  echo "示例: $0 /path/to/speaker.mp4 exhibition_avatar1"
  exit 1
fi

SOURCE_FILE="$1"
AVATAR_ID="${2:-exhibition_avatar1}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=================================================="
echo " Avatar 生成（MuseTalk v1.5）"
echo "  源文件:   $SOURCE_FILE"
echo "  Avatar ID: $AVATAR_ID"
echo "  项目目录:  $PROJECT_DIR"
echo "=================================================="

cd "$PROJECT_DIR"

# ──── 检查模型文件 ────────────────────────────────────────────
echo ""
echo "[检查] 检查模型文件..."

check_model() {
  if [ ! -f "$1" ]; then
    echo "  ✗ 缺少模型: $1"
    return 1
  fi
  echo "  ✓ $(du -sh "$1" | cut -f1) $1"
  return 0
}

MODELS_OK=true
check_model "models/musetalkV15/unet.pth" || MODELS_OK=false
check_model "models/musetalkV15/musetalk.json" || MODELS_OK=false
check_model "models/sd-vae/config.json" || MODELS_OK=false
check_model "models/sd-vae/diffusion_pytorch_model.bin" || MODELS_OK=false
check_model "models/whisper/tiny.pt" || MODELS_OK=false

if [ "$MODELS_OK" = false ]; then
  echo ""
  echo "✗ 缺少必要模型文件！请检查 models/ 目录。"
  exit 1
fi

# ──── 检查输入文件 ────────────────────────────────────────────
echo ""
echo "[检查] 输入文件..."
if [ ! -f "$SOURCE_FILE" ]; then
  echo "✗ 输入文件不存在: $SOURCE_FILE"
  exit 1
fi
echo "  ✓ $SOURCE_FILE"

# ──── 环境设置 ────────────────────────────────────────────────
CONDA_ENV="nerfstream"
CONDA_PYTHON="/opt/anaconda3/envs/$CONDA_ENV/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
  echo "✗ 找不到 conda 环境: $CONDA_ENV ($CONDA_PYTHON)"
  exit 1
fi

# ──── 检测 GPU ───────────────────────────────────────────────
GPU_ID=-1
if "$CONDA_PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  GPU_ID=0
  GPU_LABEL="CUDA"
else
  GPU_LABEL="MPS/CPU"
fi
echo "  Python:  $CONDA_PYTHON"
echo "  设备:    $GPU_LABEL"

# ──── 清理旧的输出 ────────────────────────────────────────────
AVATAR_PATH="musetalk/data/avatars/$AVATAR_ID"
if [ -d "$AVATAR_PATH" ]; then
  echo ""
  echo "[清理] 删除旧的 Avatar 数据: $AVATAR_PATH"
  rm -rf "$AVATAR_PATH"
fi

# ──── 运行生成脚本 ────────────────────────────────────────────
echo ""
echo "=================================================="
echo " 开始生成 Avatar ..."
echo " (268 帧约需 5-15 分钟，取决于设备性能)"
echo "=================================================="

PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" "$CONDA_PYTHON" \
  musetalk/genavatar_with_progress.py \
  --file "$SOURCE_FILE" \
  --avatar_id "$AVATAR_ID" \
  --gpu_id "$GPU_ID" \
  --version v15 \
  --bbox_shift 0 \
  --extra_margin 10 \
  --parsing_mode jaw \
  --left_cheek_width 90 \
  --right_cheek_width 90

# ──── 验证输出 ────────────────────────────────────────────────
echo ""
echo "[验证] 检查输出文件..."

REQUIRED_FILES=(
  "$AVATAR_PATH/latents.pt"
  "$AVATAR_PATH/coords.pkl"
  "$AVATAR_PATH/mask_coords.pkl"
  "$AVATAR_PATH/avator_info.json"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "  ✓ $f"
  else
    echo "  ✗ 缺少: $f"
    ALL_OK=false
  fi
done

FRAME_COUNT=$(ls "$AVATAR_PATH/full_imgs/"*.png 2>/dev/null | wc -l | tr -d ' ')
MASK_COUNT=$(ls "$AVATAR_PATH/mask/"*.png 2>/dev/null | wc -l | tr -d ' ')
echo "  背景帧: ${FRAME_COUNT} 张"
echo "  Mask帧: ${MASK_COUNT} 张"

echo ""
if [ "$ALL_OK" = true ] && [ "$FRAME_COUNT" -gt 0 ]; then
  # ──── 拷贝到运行时路径 ───────────────────────────────────────
  RUNTIME_AVATAR_PATH="data/avatars/$AVATAR_ID"
  mkdir -p "data/avatars"
  echo "[部署] 拷贝到运行时路径: $RUNTIME_AVATAR_PATH"
  cp -R "$AVATAR_PATH" "$RUNTIME_AVATAR_PATH"
  echo "  ✓ 拷贝完成"

  echo ""
  echo "=================================================="
  echo " ✓ Avatar 生成成功！"
  echo "  Avatar ID: $AVATAR_ID"
  echo "  生成路径:  $AVATAR_PATH"
  echo "  运行路径:  $RUNTIME_AVATAR_PATH"
  echo ""
  echo " 下一步：运行展厅数字人"
  echo "  bash scripts/start_exhibition.sh $AVATAR_ID"
  echo "=================================================="
else
  echo "✗ Avatar 生成不完整，请检查上方错误信息。"
  exit 1
fi

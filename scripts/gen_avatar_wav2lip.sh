#!/usr/bin/env bash
# ============================================================
# gen_avatar_wav2lip.sh - 生成 Wav2Lip 数字人 Avatar
#
# 用法：
#   bash scripts/gen_avatar_wav2lip.sh <视频/图片路径> [avatar_id] [img_size]
#
# 示例：
#   bash scripts/gen_avatar_wav2lip.sh video.mp4 wav2lip_avatar1
#   bash scripts/gen_avatar_wav2lip.sh video.mp4 wav2lip_avatar1 96
# ============================================================

set -e

# ──── 参数处理 ────────────────────────────────────────────────
if [ -z "$1" ]; then
  echo "用法: $0 <视频或图片路径> [avatar_id] [img_size]"
  echo "示例: $0 /path/to/speaker.mp4 wav2lip_avatar1 96"
  echo ""
  echo "参数说明："
  echo "  img_size: 面部裁剪分辨率（默认96，可选96/192/256）"
  exit 1
fi

SOURCE_FILE="$1"
AVATAR_ID="${2:-wav2lip_avatar1}"
IMG_SIZE="${3:-96}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=================================================="
echo " Avatar 生成（Wav2Lip）"
echo "  源文件:    $SOURCE_FILE"
echo "  Avatar ID: $AVATAR_ID"
echo "  面部分辨率: ${IMG_SIZE}x${IMG_SIZE}"
echo "  项目目录:  $PROJECT_DIR"
echo "=================================================="

cd "$PROJECT_DIR"

# ──── 检查模型文件 ────────────────────────────────────────────
echo ""
echo "[检查] 检查模型文件..."

if [ ! -f "models/wav2lip.pth" ]; then
  echo "✗ 缺少模型: models/wav2lip.pth"
  echo "  请先运行: bash scripts/download_model_wav2lip.sh"
  exit 1
fi
echo "  ✓ $(du -sh models/wav2lip.pth | cut -f1) models/wav2lip.pth"

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

# ──── 清理旧的输出 ────────────────────────────────────────────
AVATAR_PATH="data/avatars/$AVATAR_ID"
if [ -d "$AVATAR_PATH" ]; then
  echo ""
  echo "[清理] 删除旧的 Avatar 数据: $AVATAR_PATH"
  rm -rf "$AVATAR_PATH"
fi

# ──── 运行生成脚本 ────────────────────────────────────────────
echo ""
echo "=================================================="
echo " 开始生成 Wav2Lip Avatar ..."
echo "=================================================="

PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" "$CONDA_PYTHON" \
  wav2lip/genavatar.py \
  --avatar_id "$AVATAR_ID" \
  --video_path "$SOURCE_FILE" \
  --img_size "$IMG_SIZE" \
  --pads 0 10 0 0

# ──── 验证输出 ────────────────────────────────────────────────
echo ""
echo "[验证] 检查输出文件..."

AVATAR_OUT_PATH="results/avatars/$AVATAR_ID"
if [ ! -d "$AVATAR_OUT_PATH" ]; then
  echo "✗ Avatar 生成失败，输出目录不存在: $AVATAR_OUT_PATH"
  exit 1
fi

FRAME_COUNT=$(ls "$AVATAR_OUT_PATH/full_imgs/"*.png 2>/dev/null | wc -l | tr -d ' ')
FACE_COUNT=$(ls "$AVATAR_OUT_PATH/face_imgs/"*.png 2>/dev/null | wc -l | tr -d ' ')
echo "  背景帧: ${FRAME_COUNT} 张"
echo "  面部帧: ${FACE_COUNT} 张"

if [ "$FRAME_COUNT" -gt 0 ] && [ "$FACE_COUNT" -gt 0 ]; then
  # 拷贝到运行时路径
  echo ""
  echo "[部署] 拷贝到运行时路径: $AVATAR_PATH"
  mkdir -p "data/avatars"
  cp -R "$AVATAR_OUT_PATH" "$AVATAR_PATH"
  echo "  ✓ 拷贝完成"

  echo ""
  echo "=================================================="
  echo " ✓ Wav2Lip Avatar 生成成功！"
  echo "  Avatar ID: $AVATAR_ID"
  echo "  面部分辨率: ${IMG_SIZE}x${IMG_SIZE}"
  echo ""
  echo " 下一步：启动 Wav2Lip 数字人"
  echo "  bash scripts/start_wav2lip.sh $AVATAR_ID"
  echo "=================================================="
else
  echo "✗ Avatar 生成不完整，请检查上方错误信息。"
  exit 1
fi

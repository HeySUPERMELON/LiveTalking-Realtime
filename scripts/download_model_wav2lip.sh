#!/usr/bin/env bash
# ============================================================
# download_model_wav2lip.sh - 下载 Wav2Lip 模型文件
#
# 用法：
#   bash scripts/download_model_wav2lip.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo " 下载 Wav2Lip 模型文件"
echo " 项目目录: $PROJECT_DIR"
echo "=================================================="

# ──── 创建目录 ────────────────────────────────────────────────
mkdir -p models

# ──── 下载 wav2lip.pth（带 SAM 注意力机制的 Wav2Lip 模型）────
# 这个模型来自 LiveTalking 项目，包含 SAM 空间注意力模块
# 与 wav2lip/models/wav2lip.py 架构匹配
WAV2LIP_MODEL="models/wav2lip.pth"

if [ -f "$WAV2LIP_MODEL" ]; then
  SIZE=$(du -sh "$WAV2LIP_MODEL" | cut -f1)
  echo "  ✓ 已存在: $WAV2LIP_MODEL ($SIZE)"
else
  echo ""
  echo "[下载] wav2lip.pth ..."
  echo "  来源: LiveTalking 项目预训练权重"
  echo ""

  # 方法1: 从 HuggingFace 下载（如果可用）
  if command -v wget &>/dev/null; then
    wget -c "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth" \
      -O "$WAV2LIP_MODEL" 2>/dev/null || true
  elif command -v curl &>/dev/null; then
    curl -L -C - "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth" \
      -o "$WAV2LIP_MODEL" 2>/dev/null || true
  fi

  # 如果下载失败，给出手动下载指引
  if [ ! -f "$WAV2LIP_MODEL" ] || [ ! -s "$WAV2LIP_MODEL" ]; then
    echo ""
    echo "⚠ 自动下载失败，请手动下载："
    echo ""
    echo "  方式1 - HuggingFace:"
    echo "    wget -c https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth -O models/wav2lip.pth"
    echo ""
    echo "  方式2 - 从原项目获取:"
    echo "    git clone https://github.com/lipku/LiveTalking.git /tmp/livetalking_ref"
    echo "    cp /tmp/livetalking_ref/models/wav2lip.pth models/"
    echo ""
    echo "  下载后放到: $PROJECT_DIR/models/wav2lip.pth"
    exit 1
  fi

  SIZE=$(du -sh "$WAV2LIP_MODEL" | cut -f1)
  echo "  ✓ 下载完成: $WAV2LIP_MODEL ($SIZE)"
fi

# ──── 下载人脸检测模型 ─────────────────────────────────────────
SFD_MODEL="models/face_detection/sfd/s3fd-619a316812.pth"
if [ -f "$SFD_MODEL" ]; then
  echo "  ✓ 已存在: $SFD_MODEL"
else
  echo ""
  echo "[下载] SFD 人脸检测模型 ..."
  mkdir -p models/face_detection/sfd

  if command -v wget &>/dev/null; then
    wget -c "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
      -O "$SFD_MODEL" 2>/dev/null || true
  elif command -v curl &>/dev/null; then
    curl -L -C - "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
      -o "$SFD_MODEL" 2>/dev/null || true
  fi

  if [ ! -f "$SFD_MODEL" ] || [ ! -s "$SFD_MODEL" ]; then
    echo "  ⚠ SFD 模型下载失败，但不影响 wav2lip 主体功能"
    echo "    如需要可手动下载: https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
  else
    echo "  ✓ 下载完成: $SFD_MODEL"
  fi
fi

echo ""
echo "=================================================="
echo " ✓ Wav2Lip 模型下载完成"
echo ""
echo " 下一步：生成数字人 Avatar"
echo "  bash scripts/gen_avatar_wav2lip.sh <视频路径> [avatar_id]"
echo "=================================================="

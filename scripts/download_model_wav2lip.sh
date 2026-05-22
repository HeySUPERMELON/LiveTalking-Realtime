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
WAV2LIP_MODEL="models/wav2lip.pth"

echo ""
echo "[下载] wav2lip.pth ..."

# 从魔搭 (ModelScope) 下载
if command -v python3 &>/dev/null || command -v python &>/dev/null; then
  echo "  从魔搭 (ModelScope) Phoenix505026/wav2lip 下载..."
  PYTHON_CMD=$(command -v python3 || command -v python)
  $PYTHON_CMD -c "
import os, shutil
from modelscope import snapshot_download
model_dir = snapshot_download('Phoenix505026/wav2lip', cache_dir='./models/wav2lip_cache')
for root, dirs, files in os.walk(model_dir):
    for f in files:
        if f.endswith('.pth'):
            src = os.path.join(root, f)
            shutil.copy2(src, '$WAV2LIP_MODEL')
            print(f'OK: {f}')
            break
" 2>&1
fi

SIZE=$(du -sh "$WAV2LIP_MODEL" | cut -f1)
echo "  ✓ 下载完成: $WAV2LIP_MODEL ($SIZE)"

# ──── 下载人脸检测模型 ─────────────────────────────────────────
SFD_MODEL="models/face_detection/sfd/s3fd-619a316812.pth"
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
else
  echo "  ✓ 下载完成: $SFD_MODEL"
fi

echo ""
echo "=================================================="
echo " ✓ Wav2Lip 模型下载完成"
echo ""
echo " 下一步：生成数字人 Avatar"
echo "  bash scripts/gen_avatar_wav2lip.sh <视频路径> [avatar_id]"
echo "=================================================="

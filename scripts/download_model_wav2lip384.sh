#!/usr/bin/env bash
# ============================================================
# download_model_wav2lip384.sh - 下载 Wav2Lip 384 高分辨率模型
#
# 用法：
#   bash scripts/download_model_wav2lip384.sh
#
# 说明：
#   wav2lip384 使用 wav2lip_v2 架构（无 SAM 空间注意力）
#   支持任意面部分辨率，默认 384x384
#   模型权重来自原始 Wav2Lip + GAN 微调版本
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo " 下载 Wav2Lip384 模型文件"
echo " 项目目录: $PROJECT_DIR"
echo "=================================================="

mkdir -p models

# ──── 下载 wav2lip384.pth ─────────────────────────────────────
# wav2lip384 使用 wav2lip_v2.py 模型（无 SAM），支持 384x384 分辨率
# 原始 Wav2Lip GAN 预训练权重可直接用于此架构
WAV2LIP384_MODEL="models/wav2lip384.pth"

if [ -f "$WAV2LIP384_MODEL" ]; then
  SIZE=$(du -sh "$WAV2LIP384_MODEL" | cut -f1)
  echo "  ✓ 已存在: $WAV2LIP384_MODEL ($SIZE)"
else
  echo ""
  echo "[下载] wav2lip384.pth ..."
  echo "  来源: Wav2Lip GAN 预训练权重（兼容 wav2lip_v2 架构）"
  echo ""

  # 从魔搭 (ModelScope) 下载
  DOWNLOADED=false
  if command -v python3 &>/dev/null || command -v python &>/dev/null; then
    echo "  尝试从魔搭 (ModelScope) qiaoan/Wav2Lip384 下载..."
    PYTHON_CMD=$(command -v python3 || command -v python)
    $PYTHON_CMD -c "
import os, shutil
try:
    from modelscope import snapshot_download
    model_dir = snapshot_download('qiaoan/Wav2Lip384', cache_dir='./models/wav2lip384_cache')
    # 查找 wav2lip384.pth
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith('.pth'):
                src = os.path.join(root, f)
                shutil.copy2(src, '$WAV2LIP384_MODEL')
                print(f'OK: {f}')
                break
except Exception as e:
    print(f'FAIL: {e}')
" 2>&1 | grep -q "^OK" && DOWNLOADED=true || true
  fi

  # 方法2: 从 HuggingFace 下载（如果魔搭下载失败）
  if [ "$DOWNLOADED" = false ]; then
    echo "  尝试从 HuggingFace 下载 wav2lip_gan.pth (rudrabha/Wav2Lip)..."
    if command -v wget &>/dev/null; then
      wget -c "https://huggingface.co/rudrabha/wav2lip/resolve/main/wav2lip_gan.pth" \
        -O "$WAV2LIP384_MODEL" 2>&1 && DOWNLOADED=true || true
    fi

    if [ "$DOWNLOADED" = false ] && command -v curl &>/dev/null; then
      curl -L -C - "https://huggingface.co/rudrabha/wav2lip/resolve/main/wav2lip_gan.pth" \
        -o "$WAV2LIP384_MODEL" 2>&1 && DOWNLOADED=true || true
    fi
  fi

  # 验证下载结果
  if [ ! -f "$WAV2LIP384_MODEL" ] || [ ! -s "$WAV2LIP384_MODEL" ]; then
    echo ""
    echo "⚠ 自动下载失败，请手动获取 wav2lip384 模型权重："
    echo ""
    echo "  ⚠ 注意：wav2lip384 使用 wav2lip_v2 架构（无 SAM），"
    echo "  与标准 wav2lip.pth（有 SAM）的权重不兼容！"
    echo ""
    echo "  获取方式（魔搭 ModelScope）："
    echo "    pip install modelscope"
    echo "    python -c \"from modelscope import snapshot_download; snapshot_download('qiaoan/Wav2Lip384', cache_dir='./models/wav2lip384_cache')\""
    echo "    然后将下载的 wav2lip384.pth 复制到 models/ 目录"
    echo ""
    echo "  下载后放到: $PROJECT_DIR/models/wav2lip384.pth"
    rm -f "$WAV2LIP384_MODEL" 2>/dev/null
    exit 1
  fi

  SIZE=$(du -sh "$WAV2LIP384_MODEL" | cut -f1)
  echo "  ✓ 下载完成: $WAV2LIP384_MODEL ($SIZE)"
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
    echo "  ⚠ SFD 模型下载失败"
  else
    echo "  ✓ 下载完成: $SFD_MODEL"
  fi
fi

echo ""
echo "=================================================="
echo " ✓ Wav2Lip384 模型下载完成"
echo ""
echo " 下一步：生成 384 分辨率数字人 Avatar"
echo "  bash scripts/gen_avatar_wav2lip384.sh <视频路径> [avatar_id]"
echo "=================================================="

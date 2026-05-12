#!/usr/bin/env bash
# ============================================================
# download_models.sh - MuseTalk v1.5 所需模型一键下载
#
# 优先使用魔搭 ModelScope（国内直连），v1.5 权重走 HuggingFace 镜像
#
# 用法：
#   bash scripts/download_models.sh
#
# 国内用户可选（HuggingFace 镜像，默认已设）：
#   export HF_ENDPOINT=https://hf-mirror.com
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 国内 HuggingFace 镜像（默认开启）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=================================================="
echo " MuseTalk v1.5 模型下载"
echo " 项目目录: $PROJECT_DIR"
echo " HuggingFace 镜像: $HF_ENDPOINT"
echo "=================================================="
echo ""

# ──── 检查依赖 ─────────────────────────────────────────────
pip install huggingface_hub modelscope -q 2>/dev/null

# ══════════════════════════════════════════════════════════
# 以下模型从魔搭 ModelScope 下载（国内直连，无需代理）
# 仓库：geekane/musetalk
# ══════════════════════════════════════════════════════════

# ──── 1. SD-VAE（魔搭）──────────────────────────────────────
echo "[1/5] 下载 SD-VAE（从魔搭，约 330MB）..."
mkdir -p models/sd-vae
python3 - <<'EOF'
import os
from modelscope.hub.file_download import model_file_download

dest = "models/sd-vae/diffusion_pytorch_model.bin"
if os.path.exists(dest):
    print("  ✅ 已存在: models/sd-vae")
else:
    print("  ⬇️  从魔搭下载 sd-vae-ft-mse ...")
    model_file_download(
        model_id="geekane/musetalk",
        file_path="sd-vae-ft-mse/diffusion_pytorch_model.bin",
        local_dir="models/sd-vae-tmp",
    )
    # 魔搭下载到 sd-vae-tmp/sd-vae-ft-mse/xxx，需要移动
    import shutil
    src = "models/sd-vae-tmp/sd-vae-ft-mse"
    if os.path.exists(src):
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join("models/sd-vae", f))
        shutil.rmtree("models/sd-vae-tmp")
    print("  ✅ models/sd-vae")
EOF

# ──── 2. Whisper（魔搭）─────────────────────────────────────
echo ""
echo "[2/5] 下载 Whisper（从魔搭，约 72MB）..."
mkdir -p models/whisper
python3 - <<'EOF'
import os
from modelscope.hub.file_download import model_file_download
import shutil

dest = "models/whisper/tiny.pt"
if os.path.exists(dest):
    print("  ✅ 已存在: models/whisper")
else:
    print("  ⬇️  从魔搭下载 whisper-tiny ...")
    model_file_download(
        model_id="geekane/musetalk",
        file_path="whisper/tiny.pt",
        local_dir="models/whisper-tmp",
    )
    src = "models/whisper-tmp/whisper"
    if os.path.exists(src):
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join("models/whisper", f))
        shutil.rmtree("models/whisper-tmp")
    print("  ✅ models/whisper")
EOF

# ──── 3. DWPose（魔搭）─────────────────────────────────────
echo ""
echo "[3/5] 下载 DWPose（从魔搭，约 388MB）..."
mkdir -p models/dwpose
python3 - <<'EOF'
import os
from modelscope.hub.file_download import model_file_download
import shutil

dest = "models/dwpose/dw-ll_ucoco_384.pth"
if os.path.exists(dest):
    print("  ✅ 已存在: models/dwpose/dw-ll_ucoco_384.pth")
else:
    print("  ⬇️  从魔搭下载 DWPose ...")
    model_file_download(
        model_id="geekane/musetalk",
        file_path="dwpose/dw-ll_ucoco_384.pth",
        local_dir="models/dwpose-tmp",
    )
    src = "models/dwpose-tmp/dwpose"
    if os.path.exists(src):
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join("models/dwpose", f))
        shutil.rmtree("models/dwpose-tmp")
    print("  ✅ models/dwpose/dw-ll_ucoco_384.pth")
EOF

# ──── 4. face-parse-bisent（魔搭）──────────────────────────
echo ""
echo "[4/5] 下载 face-parse-bisent（从魔搭，约 95MB）..."
mkdir -p models/face-parse-bisent
python3 - <<'EOF'
import os
from modelscope.hub.file_download import model_file_download
import shutil

downloads = [
    ("face-parse-bisent/79999_iter.pth", "models/face-parse-bisent/79999_iter.pth"),
    ("face-parse-bisent/resnet18-5c106cde.pth", "models/face-parse-bisent/resnet18-5c106cde.pth"),
]
all_exist = all(os.path.exists(d) for _, d in downloads)
if all_exist:
    print("  ✅ 已存在: models/face-parse-bisent")
else:
    print("  ⬇️  从魔搭下载 face-parse-bisent ...")
    tmp = "models/face-parse-bisent-tmp"
    for src_path, dest_path in downloads:
        if os.path.exists(dest_path):
            continue
        model_file_download(
            model_id="geekane/musetalk",
            file_path=src_path,
            local_dir=tmp,
        )
        src_dir = os.path.join(tmp, "face-parse-bisent")
        if os.path.isdir(src_dir):
            for f in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, f), os.path.join("models/face-parse-bisent", f))
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    print("  ✅ models/face-parse-bisent")
EOF

# ══════════════════════════════════════════════════════════
# 以下模型从 HuggingFace 下载（通过镜像 hf-mirror.com）
# 魔搭上没有 v1.5 的 unet.pth
# ══════════════════════════════════════════════════════════

# ──── 5. MuseTalk v1.5 权重（HuggingFace 镜像）─────────────
echo ""
echo "[5/5] 下载 MuseTalk v1.5 权重 unet.pth（从 HuggingFace 镜像，约 500MB）..."
mkdir -p models/musetalkV15
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os

files = ["musetalkV15/unet.pth", "musetalkV15/musetalk.json"]
all_exist = True
for f in files:
    dest = os.path.join("models", f)
    if os.path.exists(dest):
        print(f"  ✅ 已存在: {dest}")
        continue
    all_exist = False
    print(f"  ⬇️  下载: {f}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    hf_hub_download(
        repo_id="TMElyralab/MuseTalk",
        filename=f,
        local_dir="models",
    )
    print(f"  ✅ {dest}")

if all_exist:
    print("  ✅ 所有 v1.5 文件已存在")
EOF

# ──── 检查完整性 ───────────────────────────────────────────
echo ""
echo "=================================================="
echo " 检查模型完整性..."
echo "=================================================="
python3 - <<'EOF'
import os

REQUIRED = [
    ("models/musetalkV15/unet.pth",       "MuseTalk v1.5 核心权重",  True),
    ("models/musetalkV15/musetalk.json",   "v1.5 模型配置",           True),
    ("models/sd-vae/config.json",          "SD-VAE 配置",             False),
    ("models/whisper/tiny.pt",             "Whisper 特征提取",        False),
    ("models/dwpose/dw-ll_ucoco_384.pth",  "DWPose 人脸关键点",       False),
    ("models/face-parse-bisent/79999_iter.pth",    "人脸分割",       False),
    ("models/face-parse-bisent/resnet18-5c106cde.pth", "ResNet18",   False),
]

all_ok = True
for path, desc, critical in REQUIRED:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {path:55s} ({size_mb:.1f}MB) {desc}")
    else:
        tag = "❌ 关键" if critical else "❌ 缺少"
        print(f"  {tag} {path:55s} {desc}")
        all_ok = False

print()
if all_ok:
    print("🎉 所有模型下载完毕！")
    print("   下一步：生成 Avatar")
    print("   bash scripts/gen_avatar_v15.sh /path/to/video.mp4 my_avatar")
else:
    print("⚠️  部分文件缺失，请检查上方提示。")
EOF

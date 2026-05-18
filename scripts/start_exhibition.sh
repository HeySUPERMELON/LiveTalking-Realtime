#!/usr/bin/env bash
# ============================================================
# start_exhibition.sh - 展厅数字人一键启动脚本
# 用法：
#   bash scripts/start_exhibition.sh [avatar_id] [options]
#
# 示例：
#   bash scripts/start_exhibition.sh exhibition_avatar1
#   bash scripts/start_exhibition.sh exhibition_avatar1 --tts cosyvoice
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AVATAR_ID="${1:-exhibition_avatar1}"
EXTRA_ARGS="${@:2}"  # 其余参数透传

echo "=================================================="
echo " 展厅数字人启动（MuseTalk v1.5 + 智能体工作流）"
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
$CONDA_PYTHON -c "import torch; print(f'  CUDA 可用: {torch.cuda.is_available()}')"
$CONDA_PYTHON -c "import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ('MPS' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'CPU'); print(f'  设备: {gpu}')" 2>/dev/null || true
echo "  Python: $CONDA_PYTHON"

# ──── Avatar 检查 ─────────────────────────────────────────────
# 运行时期望 avatar 在 ./data/avatars/ 下
AVATAR_RUNTIME_PATH="data/avatars/$AVATAR_ID"
if [ ! -d "$AVATAR_RUNTIME_PATH" ]; then
  echo ""
  echo "❌ Avatar 不存在: $AVATAR_RUNTIME_PATH"
  echo "   请先运行: bash scripts/gen_avatar_v15.sh <视频路径> $AVATAR_ID"
  exit 1
fi
echo ""
echo "[检查] Avatar $AVATAR_ID 存在 ✅"

# ──── 知识库初始化 ────────────────────────────────────────────
KB_PATH="data/knowledge_base"
if [ ! -d "$KB_PATH" ]; then
  echo ""
  echo "[初始化] 创建知识库目录..."
  mkdir -p "$KB_PATH"
  
  # 如果有示例知识库则复制
  if [ -f "docs/sample_knowledge.json" ]; then
    cp docs/sample_knowledge.json "$KB_PATH/knowledge.json"
    echo "  已加载示例知识库: docs/sample_knowledge.json"
  else
    # 创建最小化示例知识库
    cat > "$KB_PATH/knowledge.json" << 'EOF'
[
  {
    "id": "welcome",
    "content": "欢迎来到展厅！本展厅展示了丰富的文化艺术藏品，请跟随讲解员了解每件展品的历史背景。",
    "keywords": ["欢迎", "展厅", "介绍"]
  },
  {
    "id": "ticket",
    "content": "票价：成人票80元，学生票40元（凭证），1.2米以下儿童免费。支持微信、支付宝付款。",
    "keywords": ["票", "价格", "购票", "优惠", "儿童", "学生"]
  },
  {
    "id": "hours",
    "content": "开放时间：周二至周日 9:00-17:00，周一闭馆（法定节假日除外）。",
    "keywords": ["时间", "开放", "闭馆", "营业"]
  }
]
EOF
    echo "  已创建默认知识库"
  fi
fi

# ──── 启动参数 ─────────────────────────────────────────────────
# 根据设备自动选择 batch_size
BATCH_SIZE=16  # 默认（12GB GPU）
if $CONDA_PYTHON -c "import torch; m=torch.cuda.get_device_properties(0).total_memory//1024//1024//1024; exit(0 if m>=24 else 1)" 2>/dev/null; then
  BATCH_SIZE=24  # 24GB+ GPU (RTX 4090)
  echo "[自动检测] 大显存 GPU (4090)，batch_size=24"
elif $CONDA_PYTHON -c "import torch; m=torch.cuda.get_device_properties(0).total_memory//1024//1024//1024; exit(0 if m>=20 else 1)" 2>/dev/null; then
  BATCH_SIZE=20  # 20GB+ GPU (RTX 3090)
  echo "[自动检测] 大显存 GPU，batch_size=20"
elif $CONDA_PYTHON -c "import torch; exit(0 if not torch.cuda.is_available() else 1)" 2>/dev/null; then
  BATCH_SIZE=4  # MPS / CPU — 降低 batch_size 避免 OOM
  echo "[自动检测] MPS/CPU 模式，batch_size=4"
else
  echo "[自动检测] 标准显存 GPU，batch_size=16"
fi

echo ""
echo "=================================================="
echo " 启动配置："
echo "  模型: musetalk"
echo "  Avatar: $AVATAR_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  传输模式: webrtc（无需 SRS，浏览器直连）"
echo "  LLM 类型: ai_agent（展厅智能体）"
echo "  监听端口: 8010"
echo "=================================================="
echo ""
echo " 访问地址："
echo "  WebRTC 前端: http://localhost:8010/dashboard.html"
echo "  交互页面:   http://localhost:8010/webrtcapi.html"
echo ""
echo "  提示: webrtc 模式不需要 SRS 服务器，浏览器直接连接即可。"
echo ""

# ──── 启动服务 ────────────────────────────────────────────────
PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" $CONDA_PYTHON app.py \
  --model musetalk \
  --avatar_id "$AVATAR_ID" \
  --batch_size "$BATCH_SIZE" \
  --fps 50 \
  --tts edgetts \
  --llm_type ai_agent \
  --transport webrtc \
  --listenport 8010 \
  $EXTRA_ARGS

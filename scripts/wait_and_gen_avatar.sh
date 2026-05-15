#!/usr/bin/env bash
# 等待 s3fd 模型下载完成，然后自动运行 Avatar 生成
set -e

TARGET="/Users/xuguan/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"
EXPECTED_SIZE=85700000  # ~85.7MB

echo "等待 s3fd 模型下载完成..."
echo "目标: $TARGET (约 85.7MB)"

while true; do
  if [ ! -f "$TARGET" ]; then
    echo "  文件不存在，等待中..."
    sleep 30
    continue
  fi

  SIZE=$(wc -c < "$TARGET" 2>/dev/null || echo 0)
  PCT=$((SIZE * 100 / EXPECTED_SIZE))

  if [ "$SIZE" -ge "$((EXPECTED_SIZE - 1000))" ]; then
    echo "  下载完成! ($SIZE bytes)"
    break
  fi

  echo "  已下载: $(echo "scale=1; $SIZE/1048576" | bc)MB / 85.7MB ($PCT%)"
  sleep 30
done

echo ""
echo "s3fd 下载完毕，开始生成 Avatar..."

PROJECT_DIR="/Users/xuguan/Codes/LiveTalking-Realtime"
cd "$PROJECT_DIR"

PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" /opt/anaconda3/envs/nerfstream/bin/python musetalk/genavatar.py \
  --file video2.mp4 \
  --avatar_id avator_1 \
  --gpu_id -1 \
  --version v15 \
  --bbox_shift 0 \
  --extra_margin 10 \
  --parsing_mode jaw \
  --left_cheek_width 90 \
  --right_cheek_width 90

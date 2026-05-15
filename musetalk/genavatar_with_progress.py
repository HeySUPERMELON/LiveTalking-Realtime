#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genavatar_with_progress.py - Avatar 生成脚本（带详细进度条）

基于 musetalk/genavatar.py，增加了：
  - 分阶段进度条（模型加载 / 视频抽取 / 关键点检测 / VAE 编码 / Mask 生成）
  - 每帧处理耗时显示
  - MPS 兼容性处理
  - 异常捕获与友好提示
"""

import argparse
import glob
import json
import os
import pickle
import platform
import shutil
import sys
import time

import cv2
import numpy as np
import torch


# 数字人视频水印配置
WATERMARK_TEXT    = ""
WATERMARK_POS     = (10, 20)
WATERMARK_FONT    = cv2.FONT_HERSHEY_SIMPLEX
WATERMARK_SCALE   = 0.3
WATERMARK_COLOR   = (128, 128, 128)
WATERMARK_THICK   = 1

# ──── 工具函数 ──────────────────────────────────────────────────

def is_video_file(file_path):
    video_exts = ['.mp4', '.mkv', '.flv', '.avi', '.mov']
    return os.path.splitext(file_path)[1].lower() in video_exts


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """从视频中抽帧"""
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while count <= cut_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if WATERMARK_TEXT:
            cv2.putText(frame, WATERMARK_TEXT, WATERMARK_POS,
                         WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_COLOR, WATERMARK_THICK)
        cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
        count += 1
    cap.release()
    return count


def detect_device(gpu_id):
    """检测计算设备"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ──── 带进度条的关键输出函数 ─────────────────────────────────────

def log_stage(stage_name, total=None):
    """打印阶段标题"""
    bar = "─" * 60
    print(f"\n┌{bar}┐")
    if total:
        print(f"│  {stage_name}  (共 {total} 项)")
    else:
        print(f"│  {stage_name}")
    print(f"└{bar}┘")


def log_ok(msg="OK"):
    print(f"  ✓ {msg}")


def log_fail(msg):
    print(f"  ✗ {msg}")


# ──── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avatar 生成（带进度条）")
    parser.add_argument("--file", type=str, required=True, help="视频或图片路径")
    parser.add_argument("--avatar_id", type=str, default="exhibition_avatar1")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", default="jaw")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root: LiveTalking-Realtime/  (current_dir 的上一级)
    project_root = os.path.dirname(current_dir)
    save_path = os.path.join(current_dir, f'./data/avatars/{args.avatar_id}')

    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    # ═══════════════════════════════════════════════════════════
    # Stage 1: 加载模型
    # ═══════════════════════════════════════════════════════════
    device = detect_device(args.gpu_id)
    log_stage(f"Stage 1/6: 检测设备 — {device}")

    log_stage("Stage 1/6: 加载模型")

    # 1a. 面部关键点模型
    t0 = time.time()
    _IS_LINUX = (platform.system() == "Linux")

    if _IS_LINUX:
        # --- Linux: 使用 insightface ---
        print("  [1/4] 加载 insightface (Linux) ...", end=" ", flush=True)
        import insightface
        from insightface.app import FaceAnalysis
        pose_app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        pose_app.prepare(
            ctx_id=0 if device.type == 'cuda' else -1,
            det_size=(640, 640)
        )
        # 定义一个统一的获取 68 点的闭包
        def _get_face_landmarks_68(img_bgr):
            faces = pose_app.get(img_bgr)
            if not faces:
                return None
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return face.landmark_2d_106[:68].astype(np.int32)
    else:
        # --- Windows/macOS: 使用 DWPose (mmpose) ---
        print("  [1/4] 加载 DWPose (mmpose) ...", end=" ", flush=True)
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import merge_data_samples

        config_file = os.path.join(current_dir, 'utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py')
        checkpoint_file = os.path.join(project_root, 'models/dwpose/dw-ll_ucoco_384.pth')
        pose_model = init_model(config_file, checkpoint_file, device=device)

        def _get_face_landmarks_68(img_bgr):
            results = inference_topdown(pose_model, img_bgr)
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            return keypoints[0][23:91].astype(np.int32)

    print(f"done ({time.time()-t0:.1f}s)")

    # 1b. Face Detection (S3FD)
    t0 = time.time()
    print("  [2/4] 加载 S3FD 人脸检测 ...", end=" ", flush=True)
    sys.path.insert(0, os.path.join(current_dir, 'utils'))
    from face_detection import FaceAlignment, LandmarksType
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=str(device))
    print(f"done ({time.time()-t0:.1f}s)")

    # 1c. VAE + UNet
    t0 = time.time()
    print("  [3/4] 加载 VAE + UNet ...", end=" ", flush=True)
    sys.path.insert(0, current_dir)
    sys.path.insert(0, project_root)  # 确保 "musetalk.utils" 可导入
    from musetalk.utils.utils import load_all_model

    vae, unet, pe = load_all_model(
        unet_model_path=os.path.join(project_root, "models", "musetalkV15", "unet.pth"),
        vae_type="sd-vae",
        unet_config=os.path.join(project_root, "models", "musetalkV15", "musetalk.json"),
        device=device,
    )
    # MPS 不支持 half，只用 float32
    if device.type == "cuda":
        vae.vae = vae.vae.half().to(device)
    else:
        vae.vae = vae.vae.float().to(device)
    print(f"done ({time.time()-t0:.1f}s)")

    # 1d. Face Parsing
    t0 = time.time()
    print("  [4/4] 加载 FaceParsing ...", end=" ", flush=True)
    sys.path.insert(0, os.path.join(current_dir, 'utils'))
    from face_parsing import FaceParsing

    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    print(f"done ({time.time()-t0:.1f}s)")
    log_ok(f"所有模型加载完毕 (设备: {device})")

    # ═══════════════════════════════════════════════════════════
    # Stage 2: 视频抽帧
    # ═══════════════════════════════════════════════════════════
    save_full_path = os.path.join(save_path, 'full_imgs')
    mask_out_path = os.path.join(save_path, 'mask')
    create_dir(save_path)
    create_dir(save_full_path)
    create_dir(mask_out_path)

    log_stage("Stage 2/6: 视频抽帧")
    if os.path.isfile(args.file):
        if is_video_file(args.file):
            t0 = time.time()
            frame_count = video2imgs(args.file, save_full_path, ext='png')
            print(f"  抽取 {frame_count} 帧 ({time.time()-t0:.1f}s)")
        else:
            shutil.copyfile(args.file, f"{save_full_path}/{os.path.basename(args.file)}")
            frame_count = 1
            print(f"  复制图片 1 张")
    else:
        files = sorted(os.listdir(args.file))
        files = [f for f in files if f.split(".")[-1] == "png"]
        for filename in files:
            shutil.copyfile(f"{args.file}/{filename}", f"{save_full_path}/{filename}")
        frame_count = len(files)
        print(f"  复制图片 {frame_count} 张")

    input_img_list = sorted(glob.glob(os.path.join(save_full_path, '*.[jpJP][pnPN]*[gG]')))
    total_frames = len(input_img_list)
    log_ok(f"共 {total_frames} 帧")

    # 保存 avatar 元信息
    with open(os.path.join(save_path, 'avator_info.json'), "w") as f:
        json.dump({
            "avatar_id": args.avatar_id,
            "video_path": args.file,
            "bbox_shift": args.bbox_shift
        }, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Stage 3: 关键点检测 + 人脸检测
    # ═══════════════════════════════════════════════════════════
    log_stage(f"Stage 3/6: 关键点 + 人脸检测", total_frames)

    print("  读取图片...", end=" ", flush=True)
    t0 = time.time()
    frames = []
    for img_path in input_img_list:
        frame = cv2.imread(img_path)
        frames.append(frame)
    print(f"done ({time.time()-t0:.1f}s)")

    coord_list = []
    average_range_minus = []
    average_range_plus = []
    error_count = 0

    t_start = time.time()
    for i, frame in enumerate(frames):
        # 面部关键点检测（insightface 或 DWPose，由 _get_face_landmarks_68 统一封装）
        face_land_mark = _get_face_landmarks_68(frame)

        if face_land_mark is None:
            coord_list.append(coord_placeholder)
            error_count += 1
        else:
            # 人脸检测 (S3FD) — 仍用于兜底 bbox
            bbox_list = fa.get_detections_for_batch(frame[np.newaxis, ...])
            f = bbox_list[0]

            half_face_coord = face_land_mark[29]
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)

            if args.bbox_shift != 0:
                half_face_coord[1] = args.bbox_shift + half_face_coord[1]

            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = max(0, half_face_coord[1] - half_face_dist)

            f_landmark = (
                np.min(face_land_mark[:, 0]), int(upper_bond),
                np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1])
            )
            x1, y1, x2, y2 = f_landmark
            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                if f is not None:
                    coord_list.append(f)
                else:
                    coord_list.append(coord_placeholder)
                    error_count += 1
            else:
                coord_list.append(f_landmark)

        # 进度条
        elapsed = time.time() - t_start
        avg_speed = elapsed / (i + 1)
        eta = avg_speed * (total_frames - i - 1)
        bar_len = 40
        filled = int(bar_len * (i + 1) / total_frames)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = (i + 1) / total_frames * 100
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {i+1}/{total_frames}  "
                         f"速度: {avg_speed:.2f}s/帧  剩余: {eta:.0f}s  "
                         f"错误: {error_count}")
        sys.stdout.flush()

    print()
    t_stage = time.time() - t_start

    if average_range_minus and average_range_plus:
        avg_minus = int(sum(average_range_minus) / len(average_range_minus))
        avg_plus = int(sum(average_range_plus) / len(average_range_plus))
        print(f"  bbox_shift 调整范围: [-{avg_minus} ~ {avg_plus}]")
    log_ok(f"检测完成 ({t_stage:.1f}s, 错误帧: {error_count})")

    # ═══════════════════════════════════════════════════════════
    # Stage 4: VAE 编码 (最耗时)
    # ═══════════════════════════════════════════════════════════
    valid_count = sum(1 for c in coord_list if c != coord_placeholder)
    log_stage(f"Stage 4/6: VAE 编码 ({device})", valid_count)

    input_latent_list = []
    t_start = time.time()

    for idx, (bbox, frame) in enumerate(zip(coord_list, frames)):
        if bbox == coord_placeholder:
            continue

        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]

        crop_frame = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized)
        input_latent_list.append(latents)

        # 进度条
        done = len(input_latent_list)
        elapsed = time.time() - t_start
        avg_speed = elapsed / done
        eta = avg_speed * (valid_count - done)
        bar_len = 40
        filled = int(bar_len * done / valid_count)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = done / valid_count * 100
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {done}/{valid_count}  "
                         f"速度: {avg_speed:.2f}s/帧  剩余: {eta:.0f}s")
        sys.stdout.flush()

    print()
    t_stage = time.time() - t_start
    log_ok(f"VAE 编码完成 ({t_stage:.1f}s, {len(input_latent_list)} 个 latent)")

    # ═══════════════════════════════════════════════════════════
    # Stage 5: Mask 生成 + 文件保存
    # ═══════════════════════════════════════════════════════════
    log_stage(f"Stage 5/6: Mask 生成 + 保存", total_frames)

    from musetalk.utils.blending import get_image_prepare_material

    mask_coords_list = []
    t_start = time.time()

    for i, frame in enumerate(frames):
        # 保存背景帧
        cv2.imwrite(f"{save_full_path}/{str(i).zfill(8)}.png", frame)

        x1, y1, x2, y2 = coord_list[i]
        mode = args.parsing_mode if args.version == "v15" else "raw"

        mask, crop_box = get_image_prepare_material(
            frame, [x1, y1, x2, y2], fp=fp, mode=mode
        )
        cv2.imwrite(f"{mask_out_path}/{str(i).zfill(8)}.png", mask)
        mask_coords_list.append(crop_box)

        # 进度条
        elapsed = time.time() - t_start
        avg_speed = elapsed / (i + 1)
        eta = avg_speed * (total_frames - i - 1)
        bar_len = 40
        filled = int(bar_len * (i + 1) / total_frames)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = (i + 1) / total_frames * 100
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {i+1}/{total_frames}  "
                         f"速度: {avg_speed:.2f}s/帧  剩余: {eta:.0f}s")
        sys.stdout.flush()

    print()
    t_stage = time.time() - t_start
    log_ok(f"Mask 生成完成 ({t_stage:.1f}s)")

    # ═══════════════════════════════════════════════════════════
    # Stage 6: 保存 pkl/pt 文件
    # ═══════════════════════════════════════════════════════════
    log_stage("Stage 6/6: 保存数据文件")

    mask_coords_path = os.path.join(save_path, 'mask_coords.pkl')
    coords_path = os.path.join(save_path, 'coords.pkl')
    latents_out_path = os.path.join(save_path, 'latents.pt')

    with open(mask_coords_path, 'wb') as f:
        pickle.dump(mask_coords_list, f)
    log_ok(f"mask_coords.pkl ({os.path.getsize(mask_coords_path)/1024:.0f} KB)")

    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
    log_ok(f"coords.pkl ({os.path.getsize(coords_path)/1024:.0f} KB)")

    torch.save(input_latent_list, latents_out_path)
    log_ok(f"latents.pt ({os.path.getsize(latents_out_path)/1024/1024:.0f} MB)")

    # ═══════════════════════════════════════════════════════════
    # 最终统计
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 62)
    print(f"  Avatar 生成完成!")
    print(f"  ID:       {args.avatar_id}")
    print(f"  帧数:     {total_frames}")
    print(f"  有效帧:   {valid_count}")
    print(f"  Latent:   {len(input_latent_list)} 个")
    print(f"  输出路径: {save_path}")
    print("=" * 62)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 生成失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

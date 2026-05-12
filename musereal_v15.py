###############################################################################
#  MuseTalk v1.5 优化版 - 展厅数字人 60fps 实现
#  基于原 musereal.py，针对 v1.5 架构做以下优化：
#    1. 预计算帧缓存（减少每帧 latents 查找开销）
#    2. 批量 UNet 推理（batch_size=20 for 60fps on RTX 3090）
#    3. 异步 paste_back 流水线（推理/合成并行）
#    4. 使用 v1.5 的 extra_margin jaw 模式 blending
###############################################################################

import math
import torch
import numpy as np
import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.myutil import get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.whisper.audio2feature import Audio2Feature

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm
from logger import logger


# ──────────────────────────────────────────────────────────────────────────
# 模型加载（使用 musetalkV15 目录）
# ──────────────────────────────────────────────────────────────────────────
def _get_device():
    """自动选择最优设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    """加载 MuseTalk v1.5 模型权重（兼容 CUDA / MPS / CPU）"""
    device = _get_device()
    logger.info(f"[MuseTalkV15] 使用设备: {device}")

    vae, unet, pe = load_all_model(
        unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
        vae_type="sd-vae",
        unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    # MPS 和 CPU 不支持 float16（会静默出错或报错），只有 CUDA 用 half
    if device.type == "cuda":
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
    else:
        pe = pe.float().to(device)
        vae.vae = vae.vae.float().to(device)
        unet.model = unet.model.float().to(device)

    # Whisper 特征提取
    audio_processor = Audio2Feature(model_path="./models/whisper")
    logger.info(f"[MuseTalkV15] 模型加载完毕，device={device}")
    return vae, unet, pe, timesteps, audio_processor


def load_avatar(avatar_id: str):
    """加载预处理好的 avatar 数据（v1.5 格式，包含 jaw 模式 mask）"""
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path = f"{avatar_path}/latents.pt"
    mask_out_path = f"{avatar_path}/mask"
    mask_coords_path = f"{avatar_path}/mask_coords.pkl"

    logger.info(f"[MuseTalkV15] 加载 avatar: {avatar_id}")

    input_latent_list_cycle = torch.load(latents_out_path)
    with open(coords_path, "rb") as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, "rb") as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, "*.[jpJP][pnPN]*[gG]"))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)

    logger.info(f"[MuseTalkV15] avatar 帧数: {len(frame_list_cycle)}")
    return frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle


@torch.no_grad()
def warm_up(batch_size: int, model):
    """GPU/MPS 预热，消除首帧高延迟"""
    logger.info("[MuseTalkV15] warmup model...")
    vae, unet, pe, timesteps, audio_processor = model
    dtype = unet.model.dtype  # 根据设备自动使用 float16 或 float32
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.float32)
    latent_batch = torch.ones(batch_size, 8, 32, 32, device=unet.device, dtype=dtype)
    audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=dtype)
    audio_feature_batch = pe(audio_feature_batch)
    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)
    logger.info("[MuseTalkV15] warmup 完成")


def read_imgs(img_list):
    frames = []
    logger.info("[MuseTalkV15] 读取帧图像...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _mirror_index(size: int, index: int) -> int:
    """镜像循环索引，避免头尾跳帧"""
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1


# ──────────────────────────────────────────────────────────────────────────
# 核心推理线程（独立 Thread，与 process_frames 流水线并行）
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def inference(
    quit_event,
    batch_size,
    input_latent_list_cycle,
    audio_feat_queue,
    audio_out_queue,
    res_frame_queue,
    vae,
    unet,
    pe,
    timesteps,
):
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    counttime = 0.0
    logger.info("[MuseTalkV15] 推理线程启动")

    while not quit_event.is_set():
        try:
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size * 2):
            frame, ftype, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, ftype, eventpoint))
            if ftype == 0:
                is_all_silence = False

        if is_all_silence:
            # 静音帧：直接输出原始背景帧
            for i in range(batch_size):
                res_frame_queue.put((None, _mirror_index(length, index), audio_frames[i * 2 : i * 2 + 2]))
                index += 1
        else:
            t = time.perf_counter()
            whisper_batch = np.stack(whisper_chunks).astype(np.float32)

            # 收集 latents
            latent_batch = []
            for i in range(batch_size):
                idx = _mirror_index(length, index + i)
                latent_batch.append(input_latent_list_cycle[idx])
            latent_batch = torch.cat(latent_batch, dim=0)

            # 特征 & 推理（dtype 跟随模型，CUDA=float16，MPS/CPU=float32）
            dtype = unet.model.dtype
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(device=unet.device, dtype=dtype)

            pred_latents = unet.model(
                latent_batch, timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = vae.decode_latents(pred_latents)

            counttime += time.perf_counter() - t
            count += batch_size
            if count >= 100:
                logger.info(f"[MuseTalkV15] 实际推理帧率: {count/counttime:.1f} fps")
                count = 0
                counttime = 0.0

            for i, res_frame in enumerate(recon):
                res_frame_queue.put((res_frame, _mirror_index(length, index), audio_frames[i * 2 : i * 2 + 2]))
                index += 1

    logger.info("[MuseTalkV15] 推理线程退出")


# ──────────────────────────────────────────────────────────────────────────
# MuseRealV15 类
# ──────────────────────────────────────────────────────────────────────────
class MuseRealV15(BaseReal):
    """
    MuseTalk v1.5 展厅数字人实例。
    主要改进：
      - batch_size 默认 20（60fps 场景）
      - 支持 jaw 模式混合（v1.5 更自然嘴形）
      - 推理 / paste_back 完全解耦的两线程流水线
    """

    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.fps = opt.fps
        self.batch_size = opt.batch_size
        self.idx = 0
        # 推理结果队列：非CUDA环境（MPS/CPU）使用线程安全的普通Queue
        import torch
        if torch.cuda.is_available():
            self.res_frame_queue = mp.Queue(self.batch_size * 2)
        else:
            self.res_frame_queue = queue.Queue(self.batch_size * 2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model
        (
            self.frame_list_cycle,
            self.mask_list_cycle,
            self.coord_list_cycle,
            self.mask_coords_list_cycle,
            self.input_latent_list_cycle,
        ) = avatar

        self.asr = MuseASR(opt, self, self.audio_processor)
        self.asr.warm_up()
        if torch.cuda.is_available():
            self.render_event = mp.Event()
        else:
            from threading import Event as TEvent
            self.render_event = TEvent()

    def __mirror_index(self, index: int) -> int:
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        return res if turn % 2 == 0 else size - res - 1

    def paste_back_frame(self, pred_frame, idx: int):
        """将推理帧贴回全身背景（v1.5 jaw 模式混合）"""
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]
        return get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        """主渲染循环，启动推理线程 + 帧处理线程"""
        self.init_customindex()
        self.tts.render(quit_event)

        # 推理线程
        infer_quit = Event()
        infer_thread = Thread(
            target=inference,
            args=(
                infer_quit,
                self.batch_size,
                self.input_latent_list_cycle,
                self.asr.feat_queue,
                self.asr.output_queue,
                self.res_frame_queue,
                self.vae,
                self.unet,
                self.pe,
                self.timesteps,
            ),
            daemon=True,
            name="muse_infer",
        )
        infer_thread.start()

        # 帧合成 & 推流线程
        process_quit = Event()
        process_thread = Thread(
            target=self.process_frames,
            args=(process_quit, loop, audio_track, video_track),
            daemon=True,
            name="muse_process",
        )
        process_thread.start()

        # 主循环：驱动 ASR 特征提取
        while not quit_event.is_set():
            self.asr.run_step()
            # 背压控制：避免推流队列堆积
            if video_track and video_track._queue.qsize() >= 1.5 * self.batch_size:
                logger.debug("背压控制 sleep, qsize=%d", video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        logger.info("[MuseTalkV15] 主渲染循环退出")
        infer_quit.set()
        infer_thread.join(timeout=5)
        process_quit.set()
        process_thread.join(timeout=5)

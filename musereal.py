###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

#from .utils import *
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

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
#from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.myutil import get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.whisper.audio2feature import Audio2Feature

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm
from logger import logger

def _get_device():
    """自动选择最优设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model():
    # load model weights
    device = _get_device()
    logger.info(f"[MuseReal] 使用设备: {device}")
    vae, unet, pe = load_all_model()
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
    # Initialize audio processor and Whisper model
    audio_processor = Audio2Feature(model_path="./models/whisper")
    logger.info(f"[MuseReal] 模型加载完毕，device={device}")
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path = f"{avatar_path}/latents.pt"
    mask_out_path = f"{avatar_path}/mask"
    mask_coords_path = f"{avatar_path}/mask_coords.pkl"

    logger.info(f"[MuseReal] 加载 avatar: {avatar_id}")
    input_latent_list_cycle = torch.load(latents_out_path, weights_only=False)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    logger.info(f"[MuseReal] avatar 帧数: {len(frame_list_cycle)}")
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size, model):
    # 预热函数
    logger.info('[MuseReal] warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    dtype = unet.model.dtype  # 根据设备自动使用 float16 或 float32
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.float32)
    latent_batch = torch.ones(batch_size, 8, 32, 32, device=unet.device, dtype=dtype)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=dtype)
    audio_feature_batch = pe(audio_feature_batch)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)
    logger.info('[MuseReal] warmup 完成')

def read_imgs(img_list):
    frames = []
    logger.info('[MuseReal] reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def _mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1

@torch.no_grad()
def inference(quit_event,batch_size,input_latent_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,
              vae, unet, pe,timesteps):

    length = len(input_latent_list_cycle)
    index = 0
    count=0
    counttime=0
    logger.info('[MuseReal] 推理线程启动')
    while not quit_event.is_set():
        try:
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type==0:
                is_all_silence=False
        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,_mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            t=time.perf_counter()
            whisper_batch = np.stack(whisper_chunks).astype(np.float32)
            latent_batch = []
            for i in range(batch_size):
                idx = _mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)

            # dtype 跟随模型，CUDA=float16，MPS/CPU=float32
            dtype = unet.model.dtype
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(device=unet.device, dtype=dtype)

            pred_latents = unet.model(latent_batch,
                                        timesteps,
                                        encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)

            counttime += (time.perf_counter() - t)
            count += batch_size
            if count>=100:
                logger.info(f"[MuseReal] 实际推理帧率: {count/counttime:.1f} fps")
                count=0
                counttime=0
            for i,res_frame in enumerate(recon):
                res_frame_queue.put((res_frame,_mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
    logger.info('[MuseReal] 推理线程退出')

class MuseReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        self.fps = opt.fps # 20 ms per frame

        self.batch_size = opt.batch_size
        self.idx = 0
        # 非CUDA环境（MPS/CPU）使用线程安全的普通Queue
        if torch.cuda.is_available():
            self.res_frame_queue = mp.Queue(self.batch_size*2)
        else:
            self.res_frame_queue = queue.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model
        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar

        self.asr = MuseASR(opt,self,self.audio_processor)
        self.asr.warm_up()

        if torch.cuda.is_available():
            self.render_event = mp.Event()
        else:
            from threading import Event as TEvent
            self.render_event = TEvent()


    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]

        combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
        return combine_frame

    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        self.init_customindex()
        self.tts.render(quit_event)

        infer_quit_event = Event()
        infer_thread = Thread(target=inference, args=(infer_quit_event,self.batch_size,self.input_latent_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.vae, self.unet, self.pe,self.timesteps),
                              daemon=True, name="muse_infer")
        infer_thread.start()

        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,loop,audio_track,video_track),
                                daemon=True, name="muse_process")
        process_thread.start()

        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()
            if video_track and video_track._queue.qsize()>=1.5*self.opt.batch_size:
                logger.debug('[MuseReal] 背压控制 sleep, qsize=%d',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)

        logger.info('[MuseReal] 主渲染循环退出')
        infer_quit_event.set()
        infer_thread.join(timeout=5)
        process_quit_event.set()
        process_thread.join(timeout=5)

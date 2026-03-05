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

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS,IndexTTS2,AzureTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# from https://github.com/Rudrabha/Wav2Lip
import audio
mel_step_size = 16

def _load(checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model_path):
    model = Wav2Lip()
    logger.info("Loading checkpoint from: {}".format(model_path))
    checkpoint = _load(model_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.cuda()
    return model.eval()

def img2tensor(img,device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    img = torch.tensor(img,device=device).permute(2,0,1) # HWC->CHW
    img = img.unsqueeze(0)  # 1CHW
    return img

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps  # 320 samples per chunk (20ms * 16000 / 1000)

        self.fps = opt.fps
        self.curr_state=0
        self.custom_index=0
        self.custom_index_cycle=None
        self.custom_opt = {}
        self.__loadcustom()

    def put_msg_txt(self,msg,datainfo:dict={}):
        logger.info(f"Putting message text: {msg[:50]}...")
        self.tts.put_msg_txt(msg,datainfo)
    
    def put_audio_frame(self,audio_chunk,datainfo:dict={}): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,datainfo)

    def put_audio_file(self,filebyte,datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk],datainfo)
            streamlen -= self.chunk
            idx += self.chunk
    
    def flush_talk(self):
        self.tts.flush_talk()

    def is_speaking(self)->bool:
        return self.tts.is_speaking()
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'create_bytes_stream sample rate {sample_rate}, shape {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'multichannel audio, only use first channel')
            stream = stream[:, 0]  

        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'resample {sample_rate} to {self.sample_rate}')
            stream = resampy.resample(x=stream, orig_sr=sample_rate, target_sr=self.sample_rate)

        return stream

    def __loadcustom(self):
        for item in self.opt.customopt:
            print(item)
            opt_custom = load_custom(item['customvideo_img'],item['customvideo_imgnum'])
            self.custom_opt[item['audiotype']]=opt_custom
            if item['audiotype']=='start':
                self.custom_index_cycle=list(range(item['customvideo_imgnum']))
            logger.info(item['audiotype'])

    def set_custom_state(self,audiotype:str,reinit=False):
        if audiotype=="normal":
            self.curr_state=0
        else:
            self.curr_state=1
            if reinit:
                self.custom_index=-1
            self.custom_index_cycle=self.custom_opt[audiotype]['custom_index_cycle']

    def notify(self,eventpoint):
        pass

    def start_recording(self):
        """开始录制视频和音频"""
        try:
            import datetime
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_video_file = f"record_{timestamp}.mp4"
            self.record_audio_file = f"record_{timestamp}.wav"
            logger.info(f"开始录制到文件: {self.record_video_file}")
            
            # 初始化录制状态
            self.is_recording = True
            self.record_start_time = time.time()
            
            # 初始化视频录制管道
            if hasattr(self, 'width') and self.width > 0 and hasattr(self, 'height') and self.height > 0:
                self._init_video_pipe(self.width, self.height)
            else:
                logger.warning("视频尺寸未初始化，将在第一帧到达时初始化录制")
                self._record_video_pipe = None
                self._record_audio_pipe = None
            
            logger.info("录制已开始")
        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """停止录制"""
        try:
            self.is_recording = False
            
            # 关闭视频管道
            if hasattr(self, '_record_video_pipe') and self._record_video_pipe is not None:
                try:
                    self._record_video_pipe.stdin.close()
                    self._record_video_pipe.wait()
                except Exception as e:
                    logger.error(f"关闭视频管道失败: {e}")
                finally:
                    self._record_video_pipe = None
            
            # 关闭音频管道
            if hasattr(self, '_record_audio_pipe') and self._record_audio_pipe is not None:
                try:
                    self._record_audio_pipe.stdin.close()
                    self._record_audio_pipe.wait()
                except Exception as e:
                    logger.error(f"关闭音频管道失败: {e}")
                finally:
                    self._record_audio_pipe = None
            
            if hasattr(self, 'record_video_file'):
                logger.info(f"录制已停止，文件保存为: {self.record_video_file}")
        except Exception as e:
            logger.error(f"停止录制失败: {e}")

    def _init_video_pipe(self, width, height):
        """初始化视频录制管道"""
        try:
            # 确保尺寸是偶数（H.264编码要求）
            width = width if width % 2 == 0 else width + 1
            height = height if height % 2 == 0 else height + 1
            
            # 使用FFmpeg录制视频
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # 覆盖已存在的文件
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',  # 视频尺寸
                '-r', str(self.fps),  # 帧率
                '-i', '-',  # 从stdin读取
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                self.record_video_file
            ]
            
            self._record_video_pipe = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"视频录制管道已初始化: {width}x{height} @ {self.fps}fps")
        except Exception as e:
            logger.error(f"初始化视频录制管道失败: {e}")
            self._record_video_pipe = None

    def _init_audio_pipe(self):
        """初始化音频录制管道"""
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 's16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-i', '-',
                '-acodec', 'pcm_s16le',
                self.record_audio_file
            ]
            
            self._record_audio_pipe = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("音频录制管道已初始化")
        except Exception as e:
            logger.error(f"初始化音频录制管道失败: {e}")
            self._record_audio_pipe = None

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

import time
import logging
import numpy as np

import queue
from queue import Queue
#import multiprocessing as mp
from baseasr import BaseASR
from musetalk.whisper.audio2feature import Audio2Feature

logger = logging.getLogger(__name__)

# 音频上下文窗口大小（单位：whisper 帧），越大嘴形越准确但推理越重
# 默认 [2,2]=200ms，提升到 [3,3]=300ms 以改善嘴形精度
AUDIO_FEAT_LENGTH = [3, 3]


class MuseASR(BaseASR):
    def __init__(self, opt, parent,audio_processor:Audio2Feature):
        super().__init__(opt,parent)
        self.audio_processor = audio_processor
        logger.info(f"[MuseASR] audio_feat_length={AUDIO_FEAT_LENGTH} ({(AUDIO_FEAT_LENGTH[0]+AUDIO_FEAT_LENGTH[1]+1)*2*10}ms 上下文窗口)")

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        for _ in range(self.batch_size*2):
            audio_frame,type,eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type,eventpoint))

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        inputs = np.concatenate(self.frames) # [N * chunk]
        whisper_feature = self.audio_processor.audio2feat(inputs)
        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_feature,
            fps=self.fps/2,
            batch_size=self.batch_size,
            start=self.stride_left_size/2,
            audio_feat_length=AUDIO_FEAT_LENGTH,
        )
        #print(f"whisper_chunks len:{len(whisper_chunks)},self.audio_feats len:{len(self.audio_feats)},self.output_queue len:{self.output_queue.qsize()}")
        #self.audio_feats = self.audio_feats[-(self.stride_left_size + self.stride_right_size):]
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

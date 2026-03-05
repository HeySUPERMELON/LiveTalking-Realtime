import os
import numpy as np
import queue
from queue import Queue
from threading import Thread, Event
import time
from io import BytesIO
import soundfile as sf
import resampy

from enum import Enum
from logger import logger

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    def __init__(self, opt, parent):
        self.opt = opt
        self.parent = parent

        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps  # 320 samples per chunk (20ms * 16000 / 1000)

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,datainfo:dict={}): 
        logger.info(f"TTS put_msg_txt called with msg: {msg[:50]}..., queue size before: {self.msgqueue.qsize()}")
        if len(msg)>0:
            self.msgqueue.put((msg,datainfo))
            logger.info(f"TTS message added to queue, queue size after: {self.msgqueue.qsize()}")
        else:
            logger.warning("TTS received empty message, ignoring")

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        logger.info('ttsreal process_tts thread started')
        while not quit_event.is_set():
            try:
                logger.debug(f'ttsreal waiting for message, queue size: {self.msgqueue.qsize()}')
                msg:tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                logger.info(f'ttsreal got message: {msg[0][:50]}...')
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self,msg:tuple[str, dict]):
        pass
    
    def is_speaking(self)->bool:
        return self.state==State.RUNNING

class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        from edge_tts import Communicate

        text, datainfo = msg
        logger.info(f'EdgeTTS processing text: {text[:50]}...')
        
        # 使用Edge TTS生成音频
        communicate = Communicate(text, voice=self.opt.REF_FILE)
        audio_data = b''
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        # 将音频数据转换为numpy数组
        audio_stream = BytesIO(audio_data)
        audio_np, sample_rate = sf.read(audio_stream, dtype='float32')
        
        logger.info(f'EdgeTTS generated audio, shape: {audio_np.shape}, sample_rate: {sample_rate}')
        
        # 重采样到16000Hz
        if sample_rate != self.sample_rate:
            audio_np = resampy.resample(audio_np, sample_rate, self.sample_rate)
            logger.info(f'EdgeTTS resampled audio to {self.sample_rate}Hz')
        
        # 将音频分割成chunk并传递给parent
        idx = 0
        while idx < len(audio_np):
            chunk = audio_np[idx:idx+self.chunk]
            if len(chunk) < self.chunk:
                # 补齐最后一个chunk
                chunk = np.pad(chunk, (0, self.chunk - len(chunk)), 'constant')
            self.parent.put_audio_frame((chunk * 32767).astype(np.int16).tobytes())
            idx += self.chunk
            time.sleep(0.02)  # 20ms
        
        logger.info(f'EdgeTTS finished processing text: {text[:50]}...')

class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        import requests

        text, datainfo = msg
        logger.info(f'SovitsTTS processing text: {text[:50]}...')
        
        # 调用Sovits TTS服务
        url = self.opt.TTS_SERVER
        params = {
            "text": text,
            "text_language": "zh"
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            # 读取音频数据
            audio_stream = BytesIO(response.content)
            audio_np, sample_rate = sf.read(audio_stream, dtype='float32')
            
            logger.info(f'SovitsTTS generated audio, shape: {audio_np.shape}, sample_rate: {sample_rate}')
            
            # 重采样到16000Hz
            if sample_rate != self.sample_rate:
                audio_np = resampy.resample(audio_np, sample_rate, self.sample_rate)
                logger.info(f'SovitsTTS resampled audio to {self.sample_rate}Hz')
            
            # 将音频分割成chunk并传递给parent
            idx = 0
            while idx < len(audio_np):
                chunk = audio_np[idx:idx+self.chunk]
                if len(chunk) < self.chunk:
                    chunk = np.pad(chunk, (0, self.chunk - len(chunk)), 'constant')
                self.parent.put_audio_frame((chunk * 32767).astype(np.int16).tobytes())
                idx += self.chunk
                time.sleep(0.02)  # 20ms
            
            logger.info(f'SovitsTTS finished processing text: {text[:50]}...')
        except Exception as e:
            logger.error(f'SovitsTTS error: {e}')

class XTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        import requests

        text, datainfo = msg
        logger.info(f'XTTS processing text: {text[:50]}...')
        
        # 调用XTTS服务
        url = self.opt.TTS_SERVER
        data = {
            "text": text,
            "language": "zh",
            "speaker_wav": self.opt.REF_FILE,
            "gpt_cond_len": 30
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            # 读取音频数据
            audio_stream = BytesIO(response.content)
            audio_np, sample_rate = sf.read(audio_stream, dtype='float32')
            
            logger.info(f'XTTS generated audio, shape: {audio_np.shape}, sample_rate: {sample_rate}')
            
            # 重采样到16000Hz
            if sample_rate != self.sample_rate:
                audio_np = resampy.resample(audio_np, sample_rate, self.sample_rate)
                logger.info(f'XTTS resampled audio to {self.sample_rate}Hz')
            
            # 将音频分割成chunk并传递给parent
            idx = 0
            while idx < len(audio_np):
                chunk = audio_np[idx:idx+self.chunk]
                if len(chunk) < self.chunk:
                    chunk = np.pad(chunk, (0, self.chunk - len(chunk)), 'constant')
                self.parent.put_audio_frame((chunk * 32767).astype(np.int16).tobytes())
                idx += self.chunk
                time.sleep(0.02)  # 20ms
            
            logger.info(f'XTTS finished processing text: {text[:50]}...')
        except Exception as e:
            logger.error(f'XTTS error: {e}')

class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        import requests

        text, datainfo = msg
        logger.info(f'CosyVoiceTTS processing text: {text[:50]}...')
        
        # 调用CosyVoice服务
        url = self.opt.TTS_SERVER
        data = {
            "text": text,
            "spk_id": self.opt.REF_FILE
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            # 读取音频数据
            audio_stream = BytesIO(response.content)
            audio_np, sample_rate = sf.read(audio_stream, dtype='float32')
            
            logger.info(f'CosyVoiceTTS generated audio, shape: {audio_np.shape}, sample_rate: {sample_rate}')
            
            # 重采样到16000Hz
            if sample_rate != self.sample_rate:
                audio_np = resampy.resample(audio_np, sample_rate, self.sample_rate)
                logger.info(f'CosyVoiceTTS resampled audio to {self.sample_rate}Hz')
            
            # 将音频分割成chunk并传递给parent
            idx = 0
            while idx < len(audio_np):
                chunk = audio_np[idx:idx+self.chunk]
                if len(chunk) < self.chunk:
                    chunk = np.pad(chunk, (0, self.chunk - len(chunk)), 'constant')
                self.parent.put_audio_frame((chunk * 32767).astype(np.int16).tobytes())
                idx += self.chunk
                time.sleep(0.02)  # 20ms
            
            logger.info(f'CosyVoiceTTS finished processing text: {text[:50]}...')
        except Exception as e:
            logger.error(f'CosyVoiceTTS error: {e}')

class FishTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        import requests

        text, datainfo = msg
        logger.info(f'FishTTS processing text: {text[:50]}...')
        
        # 调用Fish TTS服务
        url = self.opt.TTS_SERVER
        data = {
            "text": text,
            "reference_id": self.opt.REF_FILE
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            # 读取音频数据
            audio_stream = BytesIO(response.content)
            audio_np, sample_rate = sf.read(audio_stream, dtype='float32')
            
            logger.info(f'FishTTS generated audio, shape: {audio_np.shape}, sample_rate: {sample_rate}')
            
            # 重采样到16000Hz
            if sample_rate != self.sample_rate:
                audio_np = resampy.resample(audio_np, sample_rate, self.sample_rate)
                logger.info(f'FishTTS resampled audio to {self.sample_rate}Hz')
            
            # 将音频分割成chunk并传递给parent
            idx = 0
            while idx < len(audio_np):
                chunk = audio_np[idx:idx+self.chunk]
                if len(chunk) < self.chunk:
                    chunk = np.pad(chunk, (0, self.chunk - len(chunk)), 'constant')
                self.parent.put_audio_frame((chunk * 32767).astype(np.int16).tobytes())
                idx += self.chunk
                time.sleep(0.02)  # 20ms
            
            logger.info(f'FishTTS finished processing text: {text[:50]}...')
        except Exception as e:
            logger.error(f'FishTTS error: {e}')

class TencentTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        # 腾讯云TTS实现
        logger.info(f'TencentTTS processing text: {msg[0][:50]}...')
        # TODO: 实现腾讯云TTS
        logger.warning('TencentTTS not implemented yet')

class DoubaoTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        # 豆包TTS实现
        logger.info(f'DoubaoTTS processing text: {msg[0][:50]}...')
        # TODO: 实现豆包TTS
        logger.warning('DoubaoTTS not implemented yet')

class IndexTTS2(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        # IndexTTS2实现
        logger.info(f'IndexTTS2 processing text: {msg[0][:50]}...')
        # TODO: 实现IndexTTS2
        logger.warning('IndexTTS2 not implemented yet')

class AzureTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        # Azure TTS实现
        logger.info(f'AzureTTS processing text: {msg[0][:50]}...')
        # TODO: 实现Azure TTS
        logger.warning('AzureTTS not implemented yet')

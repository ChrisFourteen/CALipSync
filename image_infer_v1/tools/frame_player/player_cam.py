import cv2
from queue import Queue, Empty
import threading
import numpy as np
import pyaudio
from time import sleep, time
import pyvirtualcam  # 添加这个导入

# 固定参数
FPS = 25
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paFloat32
FRAME_TIME = 1.0 / FPS

class StreamPlayer:
    def __init__(self, window_width, window_height):  # 移除 window_width 和 window_height 参数
        self.running = True
        self.frame_queue = Queue()
        self.width = 1080  # 固定宽度
        self.height = 1920  # 固定高度
        
        # 初始化虚拟摄像头
        self.virtual_cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=FPS)
        
        # 设置可配置的采样率，默认为32kHz
        self.SAMPLE_RATE = 32000
        
        # 音频帧大小（根据采样率和FPS动态计算）
        self.samples_per_frame = self.SAMPLE_RATE // FPS
        
        # 音频播放状态
        self.audio_frame_count = 0
        self.last_audio_time = 0
        
        # 音频播放锁
        self.audio_lock = threading.Lock()
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        self.audio_stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        try:
            if not self.running or self.frame_queue.empty():
                return (np.zeros(frame_count, dtype=np.float32), pyaudio.paContinue)
            
            with self.audio_lock:
                frame_info = self.frame_queue.get_nowait()
                self.audio_frame_count += 1
                
                # 处理视频帧
                frame = frame_info['frame']
                # 确保帧大小正确
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                # 确保是RGB格式
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame_info.get('is_bgr', True):  # 假设输入是BGR格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 发送到虚拟摄像头
                self.virtual_cam.send(frame)
                
                return (frame_info['audio'], pyaudio.paContinue)
                
        except Empty:
            return (np.zeros(frame_count, dtype=np.float32), pyaudio.paContinue)

    def start_playing(self):
        self.running = True
        self.audio_frame_count = 0
        self.last_audio_time = time()
        self._init_audio_stream()
        self.audio_stream.start_stream()

    def stop_playing(self):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        self.p.terminate()
        self.virtual_cam.close()  # 关闭虚拟摄像头
        self.clear_queues()

    def _init_audio_stream(self):
        """初始化音频流"""
        if self.audio_stream is not None:
            return
            
        self.audio_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.SAMPLE_RATE,  # 使用类变量
            output=True,
            stream_callback=self._audio_callback,
            frames_per_buffer=self.samples_per_frame
        )

    def clear_queues(self):
        """清空所有队列"""
        while not self.frame_queue.empty():
            self.frame_queue.get()

    def upload_frame(self, frame_info):
        """上传单个帧到队列"""
        self.frame_queue.put(frame_info)

    def get_queue_length(self):
        """获取当前队列长度"""
        return self.frame_queue.qsize()

    def get_queue_item_at_index(self, index):
        """获取指定索引的队列项"""
        try:
            return list(self.frame_queue.queue)[index]
        except IndexError:
            return None

    def get_last_queue_item(self):
        """获取最后一个队列项"""
        try:
            return list(self.frame_queue.queue)[-1]
        except IndexError:
            return None
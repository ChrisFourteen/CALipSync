import cv2
from queue import Queue, Empty
import threading
import numpy as np
import pyaudio
from time import sleep, time
import win32gui
import win32con

# 固定参数
FPS = 25
SAMPLE_RATE = 16000  # 修改为高采样率32kHz
CHANNELS = 1
FORMAT = pyaudio.paFloat32
FRAME_TIME = 1.0 / FPS

class CVPlayer:
    def __init__(self, width, height):
        self.original_width = width
        self.original_height = height
        self.WIDTH = width
        self.HEIGHT = height
        self.aspect_ratio = width / height
        self.frame_queue = Queue()
        self.is_playing = False
        self._render_thread = None
        self.window_name = "Video Player"
        
    def calculate_new_size(self, width, height):
        """计算保持纵横比的新尺寸"""
        current_ratio = width / height
        if current_ratio > self.aspect_ratio:
            # 以高度为基准
            new_width = int(height * self.aspect_ratio)
            new_height = height
        else:
            # 以宽度为基准
            new_width = width
            new_height = int(width / self.aspect_ratio)
        return new_width, new_height

    def remove_window_buttons(self):
        """移除窗口的最小化、最大化和关闭按钮"""
        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd:
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            style = style & ~win32con.WS_SYSMENU  # 移除系统菜单（包含关闭按钮）
            win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)

    def render_loop(self):
        # 创建窗口并设置为可调整大小
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.WIDTH, self.HEIGHT)
        
        # 移除窗口按钮
        self.remove_window_buttons()
        
        def on_window_resize(width, height):
            """窗口大小变化回调"""
            if width > 0 and height > 0:  # 避免无效尺寸
                new_width, new_height = self.calculate_new_size(width, height)
                cv2.resizeWindow(self.window_name, new_width, new_height)
                self.WIDTH, self.HEIGHT = new_width, new_height
        
        # 设置窗口回调
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        
        while self.is_playing:
            try:
                frame = self.frame_queue.get(timeout=0.01)
                
                # 获取当前窗口大小
                current_width = cv2.getWindowImageRect(self.window_name)[2]
                current_height = cv2.getWindowImageRect(self.window_name)[3]
                
                # 如果窗口大小发生变化，进行等比例调整
                if current_width != self.WIDTH or current_height != self.HEIGHT:
                    on_window_resize(current_width, current_height)
                
                # 缩放图像以适应窗口
                if self.WIDTH != self.original_width or self.HEIGHT != self.original_height:
                    frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT), 
                                     interpolation=cv2.INTER_LINEAR)
                
                # 显示帧
                cv2.imshow(self.window_name, frame)
                
                # 等待1ms，不检测ESC键
                cv2.waitKey(1)
                    
            except Empty:
                sleep(0.001)
            except Exception as e:
                print(f"Error in render loop: {e}")
                break
        
        cv2.destroyWindow(self.window_name)

    def start_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self._render_thread = threading.Thread(target=self.render_loop)
            self._render_thread.start()

    def add_frame(self, frame):
        if self.is_playing:
            self.frame_queue.put(frame)

    def stop_playback(self):
        self.is_playing = False
        if self._render_thread:
            self._render_thread.join()
            self._render_thread = None

class StreamPlayer:
    def __init__(self, window_width, window_height):
        self.running = True
        self.frame_queue = Queue()
        self.width = window_width
        self.height = window_height
        
        # 初始化OpenCV播放器
        self.player = CVPlayer(self.width, self.height)
        
        # 设置可配置的采样率，默认为32kHz
        self.SAMPLE_RATE = 32000  # 修改为高采样率
        
        # 音频帧大小（根据采样率和FPS动态计算）
        self.samples_per_frame = self.SAMPLE_RATE // FPS
        
        # 音频播放状态
        self.audio_frame_count = 0  # 已播放的音频帧数
        self.last_audio_time = 0    # 上一帧的音频时间
        
        # 音频播放锁
        self.audio_lock = threading.Lock()
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        self.audio_stream = None  # 推迟初始化，直到start_playing

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

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调，同时控制视频播放"""
        try:
            if not self.running or self.frame_queue.empty():
                return (np.zeros(frame_count, dtype=np.float32), pyaudio.paContinue)
            
            # 获取当前帧
            with self.audio_lock:
                frame_info = self.frame_queue.get_nowait()
                self.audio_frame_count += 1
                
                # 发送视频帧到OpenCV播放器
                self.player.add_frame(frame_info['frame'])
                
                # 返回音频数据
                return (frame_info['audio'], pyaudio.paContinue)
                
        except Empty:
            return (np.zeros(frame_count, dtype=np.float32), pyaudio.paContinue)

    def start_playing(self):
        self.running = True
        self.audio_frame_count = 0
        self.last_audio_time = time()
        self.player.start_playback()
        self._init_audio_stream()  # 确保音频流已初始化
        self.audio_stream.start_stream()

    def stop_playing(self):
        self.running = False
        self.player.stop_playback()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        self.p.terminate()
        self.clear_queues()

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
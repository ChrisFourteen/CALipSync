import subprocess
import threading
import time
import numpy as np
import win32pipe
import win32file
import queue
from typing import Dict, Any

class RTMPStreamer:
    def __init__(self, rtmp_url: str, width: int, height: int):
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.ffmpeg_process = None
        self.running = False
        
        self.video_pipe_name = r'\\.\pipe\video_pipe'
        self.audio_pipe_name = r'\\.\pipe\audio_pipe'
        self.video_handle = None
        self.audio_handle = None
        
        self.video_queue = queue.Queue(maxsize=100)
        self.audio_queue = queue.Queue(maxsize=100)

    def _create_pipe(self, pipe_name):
        return win32pipe.CreateNamedPipe(
            pipe_name,
            win32pipe.PIPE_ACCESS_OUTBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1, 65536, 65536,
            0,
            None
        )

    def _video_worker(self):
        win32pipe.ConnectNamedPipe(self.video_handle, None)
        print("视频管道已连接")
        
        while self.running:
            try:
                frame = self.video_queue.get(timeout=1)
                win32file.WriteFile(self.video_handle, frame.tobytes())
            except queue.Empty:
                continue

    def _audio_worker(self):
        win32pipe.ConnectNamedPipe(self.audio_handle, None)
        print("音频管道已连接")
        
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1)
                audio_data = (audio * 32767).astype(np.int16)
                win32file.WriteFile(self.audio_handle, audio_data.tobytes())
            except queue.Empty:
                continue

    def start(self):
        print("创建管道...")
        self.video_handle = self._create_pipe(self.video_pipe_name)
        self.audio_handle = self._create_pipe(self.audio_pipe_name)
        
        print("启动FFmpeg进程...")
        command = [
            'ffmpeg',
            '-loglevel', 'warning',  # 使用warning可以看到可能的错误信息
            '-y',
            '-hwaccel', 'cuda',     # 添加: 启用CUDA硬件加速
            '-hwaccel_output_format', 'cuda',  # 添加: 保持数据在GPU内存中
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', '25',
            '-i', self.video_pipe_name,
            '-f', 's16le',
            '-ar', '16000',
            '-ac', '1',
            '-i', self.audio_pipe_name,
            '-c:v', 'h264_nvenc',   # 修改: 使用NVENC编码器
            '-preset', 'p4',        # 修改: NVENC特有的预设
            '-c:a', 'aac',
            '-f', 'flv',
            self.rtmp_url
        ]
        
        # 修改: 重定向FFmpeg输出
        self.ffmpeg_process = subprocess.Popen(
            command, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        self.running = True
        
        print("启动工作线程...")
        threading.Thread(target=self._video_worker, daemon=True).start()
        threading.Thread(target=self._audio_worker, daemon=True).start()
        
        print("RTMP推流器已启动")

    def push_frame(self, frame_data: Dict[str, np.ndarray]):
        if not self.running:
            return
        try:
            self.video_queue.put(frame_data['frame'], timeout=1)
            self.audio_queue.put(frame_data['audio'], timeout=1)
        except queue.Full:
            print("队列已满，丢弃帧")

    def stop(self):
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
        
        if self.video_handle:
            win32file.CloseHandle(self.video_handle)
        if self.audio_handle:
            win32file.CloseHandle(self.audio_handle)

class StreamPlayer:
    #def __init__(self, width: int, height: int, push_url:str):
    def __init__(self, width: int, height: int):
        push_url = 'rtmp://192.168.31.172:1935/live/test'
        self.streamer = RTMPStreamer(push_url, width, height)
        self.frame_queue = queue.Queue()
        self.running = False
        self.play_thread = None
        self.FPS = 25
        self.FRAME_TIME = 1.0 / self.FPS

    def start_playing(self):
        if self.running:
            return
        
        self.running = True
        self.streamer.start()
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.play_thread.start()
        print("播放器已启动")

    def stop_playing(self):
        self.running = False
        if self.play_thread:
            self.play_thread.join()
        self.streamer.stop()
        print("播放器已停止")

    def get_queue_length(self) -> int:
        return self.frame_queue.qsize()

    def get_last_queue_item(self) -> Dict[str, Any]:
        if self.frame_queue.empty():
            return None
        return list(self.frame_queue.queue)[-1]

    def upload_frame(self, frame_info: Dict[str, Any]):
        self.frame_queue.put(frame_info)

    def _play_worker(self):
        start_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                frame_info = self.frame_queue.get(timeout=1)
                
                target_time = start_time + frame_count * self.FRAME_TIME
                wait_time = target_time - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)
                
                self.streamer.push_frame(frame_info)
                frame_count += 1
                
            except queue.Empty:
                start_time = time.time()
                frame_count = 0
                continue

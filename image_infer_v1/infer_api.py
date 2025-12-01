import os
import threading
from queue import Queue
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from bin.image_infer_v1.utils.hubert_extractor import HubertExtractor
from bin.image_infer_v1.tools.frame_synthesizer.infer_api import FrameSynthesizer
#from bin.image_infer_v1.tools.frame_player.player_main import StreamPlayer
from bin.image_infer_v1.tools.frame_player.player_cam import StreamPlayer
import librosa
import time

@dataclass
class AudioBatch:
    """单帧音频数据结构"""
    hubert_feature: np.ndarray    # 单帧hubert特征
    audio_chunk: np.ndarray       # 低采样率的音频数据 (16kHz, 用于特征提取)
    original_audio_chunk: np.ndarray  # 高采样率的音频数据 (32kHz, 用于播放)
    audio_path: str               # 音频文件路径
    is_sync_frame: bool = True    # 是否生成口型同步帧，默认为True

class VideoStreamManager:
    def __init__(self, 
                 data_dir: str,
                 unet_checkpoint: str,
                 hubert_path: str,
                 window_width: int,
                 window_height: int, 
                 buffer_threshold: int,
                 #push_url: str,
                 device: str = "cuda:0",
                 batch_size: int = 8,
                 synthesis_window_seconds: float = 1.0,
                 output_sample_rate: int = 24000,
                 image_output_mode: str = "window",
                 is_interrupt: bool = False):  # 添加是否打断参数
        """初始化视频流管理器"""
        # 初始化合成器
        self.synthesizer = FrameSynthesizer(
            unet_checkpoint=unet_checkpoint,
            data_dir=data_dir,
            device=device,
            batch_size=batch_size
        )
        
        # 初始化Hubert特征提取器
        self.hubert_extractor = HubertExtractor(hubert_path, device)
        
        # 初始化音频队列
        self.audio_queues = {
            'normal': Queue(),
            'danmu': Queue()
        }
        
        # 基础音频参数
        self.feature_sample_rate = 16000  # Hubert特征提取采样率
        self.output_sample_rate = output_sample_rate  # 输出/播放采样率
        self.fps = 25
        self.samples_per_frame = self.feature_sample_rate // self.fps  # 特征提取的每帧样本数
        self.output_samples_per_frame = self.output_sample_rate // self.fps  # 输出的每帧样本数
        
        # 合成窗口参数 (基于fps和指定的窗口时长)
        self.synthesis_frame_count = int(self.fps * synthesis_window_seconds)
        
        # 队列优先级设置
        self.queue_priority = ['danmu', 'normal']
        
        # 窗口参数
        self.window_width = window_width
        self.window_height = window_height
        self.buffer_threshold = buffer_threshold
        
        # 创建临时目录
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 生成静默音频
        silence_duration = self.buffer_threshold / self.fps
        self.silence_audio_path = self.generate_silence_audio(silence_duration)
        
        # 初始化播放器
        self.player = None
        
        # 保存输出模式
        self.image_output_mode = image_output_mode
        
        # 保存是否打断设置
        self.is_interrupt = is_interrupt
        
        #self.push_url = push_url
        
    def init_player(self):
        """初始化本地播放器"""
        if self.image_output_mode == "window":
            # 导入窗口播放器
            from bin.image_infer_v1.tools.frame_player.player_main import StreamPlayer
            self.player = StreamPlayer(self.window_width, self.window_height)
        else:  # 默认为cam模式
            # 导入摄像头播放器  
            from bin.image_infer_v1.tools.frame_player.player_cam import StreamPlayer
            self.player = StreamPlayer(self.window_width, self.window_height)
        
        # 设置播放器的采样率
        self.player.SAMPLE_RATE = self.output_sample_rate
        self.player.samples_per_frame = self.output_samples_per_frame
        
    def generate_silence_audio(self, duration_seconds: float) -> str:
        """生成指定时长的静默音频"""
        # 生成高采样率的静默音频文件
        samples = int(duration_seconds * self.output_sample_rate)
        silence = np.zeros(samples, dtype=np.float32)
        silence_path = os.path.join(self.temp_dir, f"silence_{duration_seconds:.1f}s.wav")
        sf.write(silence_path, silence, self.output_sample_rate)
        return silence_path
        
    def add_audio_to_queue(self, audio_path: str, queue_name='normal', is_sync_frame=True, use_silent_features: bool = False):
        """
        处理音频文件并将其添加到指定优先级队列
        Args:
            audio_path: 音频文件路径
            queue_name: 队列名称
            is_sync_frame: 是否需要生成口型同步帧，默认为True
            use_silent_features: 如果为True, 则使用静默特征，实现助播效果
        """
        if queue_name not in self.audio_queues:
            raise ValueError(f"Unknown queue name: {queue_name}")
            
        # 加载原始音频（保持32kHz采样率）
        original_audio, original_sr = librosa.load(audio_path, sr=self.output_sample_rate, mono=True)
        
        # 加载并重采样音频用于特征提取（16kHz）
        feature_audio, _ = librosa.load(audio_path, sr=self.feature_sample_rate, mono=True)
        
        # 确保音频格式正确
        if original_audio.dtype != np.float32:
            original_audio = original_audio.astype(np.float32)
        if feature_audio.dtype != np.float32:
            feature_audio = feature_audio.astype(np.float32)
                
        # 提取Hubert特征
        hubert_features = []
        if use_silent_features:
            print(f"使用静默特征处理助播音频: {os.path.basename(audio_path)}")
            # 1. 计算真实音频的帧数
            num_frames = len(feature_audio) // self.samples_per_frame
            # 2. 提取一段静默特征
            silent_hubert_features_full = self.hubert_extractor.extract_from_file(self.silence_audio_path)
            
            if len(silent_hubert_features_full) == 0:
                # 如果静默特征提取失败，使用零向量作为后备
                hubert_features = [np.zeros(256, dtype=np.float32)] * num_frames
            else:
                # 使用第一个静默特征帧来填充整个长度
                silent_feature_frame = silent_hubert_features_full[0]
                hubert_features = [silent_feature_frame] * num_frames
        else:
            hubert_features = self.hubert_extractor.extract_from_file(audio_path)
        
        # 生成音频块并将单帧数据加入队列
        for i in range(len(hubert_features)):
            # 创建16kHz音频块 (特征提取用)
            audio_start = i * self.samples_per_frame
            if audio_start + self.samples_per_frame > len(feature_audio):
                audio_chunk = np.pad(feature_audio[audio_start:], 
                                    (0, self.samples_per_frame - (len(feature_audio) - audio_start)))
            else:
                audio_chunk = feature_audio[audio_start:audio_start + self.samples_per_frame]
            
            # 创建32kHz音频块 (播放用)
            original_start = i * self.output_samples_per_frame
            if original_start + self.output_samples_per_frame > len(original_audio):
                original_chunk = np.pad(original_audio[original_start:], 
                                       (0, self.output_samples_per_frame - (len(original_audio) - original_start)))
            else:
                original_chunk = original_audio[original_start:original_start + self.output_samples_per_frame]
            
            # 创建并加入队列
            batch = AudioBatch(
                hubert_feature=hubert_features[i],
                audio_chunk=audio_chunk,
                original_audio_chunk=original_chunk,
                audio_path=audio_path,
                is_sync_frame=is_sync_frame
            )
            self.audio_queues[queue_name].put(batch)
        
    def start_playing(self):
        """启动本地播放和处理线程"""
        if self.player is None:
            self.init_player()
        threading.Thread(target=self.player.start_playing).start()
        threading.Thread(target=self.process_and_manage_audio).start()
        
    def stop_playing(self):
        """停止播放"""
        if self.player:
            self.player.stop_playing()
            self.player = None
        for queue in self.audio_queues.values():
            queue.queue.clear()
            
    def process_and_manage_audio(self):
        """处理音频队列并生成帧"""
        if self.is_interrupt:
            # 使用当前文件中的打断实现
            print("使用打断模式")
            self._process_and_manage_audio_interrupt()
        else:
            # 使用copy文件中的不打断实现
            print("使用不打断模式")
            self._process_and_manage_audio_no_interrupt()

    def _process_and_manage_audio_interrupt(self):
        """打断式处理音频队列并生成帧"""
        last_frame_index = 0
        
        while self.player and self.player.running:
            try:
                queue_length = self.player.get_queue_length()
                
                if queue_length < self.buffer_threshold:
                    # 获取最后处理的帧索引
                    if queue_length > 0:
                        last_frame_index = self.player.get_last_queue_item()['index'] + 1
                    
                    # 从优先级队列中获取音频批次
                    batch_frames = []
                    selected_queue = None
                    is_sync_frame = True  # 默认需要同步
                    
                    # 按优先级选择队列
                    for queue_name in self.queue_priority:
                        if not self.audio_queues[queue_name].empty():
                            selected_queue = queue_name
                            break
                    
                    # 从选定队列收集帧
                    if selected_queue is not None:
                        queue = self.audio_queues[selected_queue]
                        
                        # 收集足够的帧以形成批次
                        max_frames_to_collect = min(self.synthesis_frame_count, queue.qsize())
                        
                        if max_frames_to_collect > 0:
                            # 直接收集可用帧，不再检查音频路径
                            for _ in range(max_frames_to_collect):
                                if queue.empty():
                                    break
                                
                                # 收集这一帧
                                next_batch = queue.get()
                                batch_frames.append(next_batch)
                                # 使用最后一帧的同步标志
                                is_sync_frame = next_batch.is_sync_frame
                    
                    # 如果没有足够的数据，使用静默音频
                    if not batch_frames:
                        # 处理静默音频
                        original_audio, _ = librosa.load(self.silence_audio_path, sr=self.output_sample_rate, mono=True)
                        feature_audio, _ = librosa.load(self.silence_audio_path, sr=self.feature_sample_rate, mono=True)
                        hubert_features = self.hubert_extractor.extract_from_file(self.silence_audio_path)
                        
                        # 创建静默音频批次
                        for i in range(min(len(hubert_features), self.synthesis_frame_count)):
                            # 低采样率音频块
                            audio_start = i * self.samples_per_frame
                            if audio_start + self.samples_per_frame > len(feature_audio):
                                audio_chunk = np.pad(feature_audio[audio_start:], 
                                                   (0, self.samples_per_frame - (len(feature_audio) - audio_start)))
                            else:
                                audio_chunk = feature_audio[audio_start:audio_start + self.samples_per_frame]
                            
                            # 高采样率音频块
                            original_start = i * self.output_samples_per_frame
                            if original_start + self.output_samples_per_frame > len(original_audio):
                                original_chunk = np.pad(original_audio[original_start:], 
                                                      (0, self.output_samples_per_frame - (len(original_audio) - original_start)))
                            else:
                                original_chunk = original_audio[original_start:original_start + self.output_samples_per_frame]
                            
                            batch_frames.append(AudioBatch(
                                hubert_feature=hubert_features[i],
                                audio_chunk=audio_chunk,
                                original_audio_chunk=original_chunk,
                                audio_path=self.silence_audio_path,
                                is_sync_frame=True  # 静默音频默认使用同步
                            ))
                    
                    # 准备特征数据
                    hubert_features = np.stack([frame.hubert_feature for frame in batch_frames])
                    
                    # 生成合成帧，传递同步标志
                    frame_generator = self.synthesizer.iterate_synthesized_frames(
                        features=hubert_features,
                        start_frame_idx=last_frame_index,
                        is_generate_sync_frame=is_sync_frame  # 传递同步标志
                    )
                    
                    # 处理生成的帧
                    for i, frame_info in enumerate(frame_generator):
                        if i < len(batch_frames):
                            # 使用高采样率音频进行播放
                            frame_info['audio'] = batch_frames[i].original_audio_chunk
                            self.player.upload_frame(frame_info)
                
                threading.Event().wait(0.001)
            except Exception as e:
                # 添加关键错误捕获，输出详细错误信息但继续循环
                print(f"视频帧处理发生错误: {e}")
                import traceback
                print(traceback.format_exc())
                # 短暂暂停避免疯狂打印错误
                time.sleep(1)

    def _process_and_manage_audio_no_interrupt(self):
        """不打断式处理音频队列并生成帧"""
        last_frame_index = 0
        current_audio_path = None
        
        while self.player and self.player.running:
            try:
                queue_length = self.player.get_queue_length()
                
                if queue_length < self.buffer_threshold:
                    # 获取最后处理的帧索引
                    if queue_length > 0:
                        last_frame_index = self.player.get_last_queue_item()['index'] + 1
                    
                    # 从优先级队列中获取音频批次
                    batch_frames = []
                    selected_queue = None
                    is_sync_frame = True  # 默认需要同步
                    
                    # 检查我们是否需要继续处理当前音频
                    if current_audio_path is not None:
                        # 查找包含当前音频的队列
                        for queue_name, queue in self.audio_queues.items():
                            if not queue.empty():
                                # 检查队列中是否还有当前音频的帧
                                for item in list(queue.queue):
                                    if item.audio_path == current_audio_path:
                                        selected_queue = queue_name
                                        is_sync_frame = item.is_sync_frame  # 保存同步标志
                                        break
                                if selected_queue:
                                    break
                        
                        # 如果没有找到当前音频的帧，说明已经处理完毕
                        if not selected_queue:
                            current_audio_path = None
                    
                    # 只有在没有当前处理的音频时，才按优先级选择队列
                    if current_audio_path is None:
           
                        for queue_name in self.queue_priority:
                            if not self.audio_queues[queue_name].empty():
                                selected_queue = queue_name
                                break
                    
                    # 从选定队列收集帧
                    if selected_queue is not None:
                        queue = self.audio_queues[selected_queue]
                        
                        # 收集足够的帧以形成批次，确保来自同一音频
                        max_frames_to_collect = min(self.synthesis_frame_count, queue.qsize())
                        
                        if max_frames_to_collect > 0:
                            # 获取第一帧以确定音频路径和同步标志
                            first_batch = queue.queue[0]
                            audio_path = first_batch.audio_path
                            is_sync_frame = first_batch.is_sync_frame  # 获取同步标志
                            current_audio_path = audio_path
                            
                            # 收集同一音频路径的所有可用帧，但不超过最大批次大小
                            frames_collected = 0
                            
                            while frames_collected < max_frames_to_collect:
                                if queue.empty():
                                    break
                                    
                                # 查看下一个批次但不移除
                                next_batch = queue.queue[0]
                                
                                # 如果不是同一个音频，停止收集
                                if next_batch.audio_path != audio_path:
                                    break
                                    
                                # 收集这一帧
                                batch_frames.append(queue.get())
                                frames_collected += 1
                    
                    # 如果没有足够的数据，使用静默音频
                    if not batch_frames:
                        # 处理静默音频
                        original_audio, _ = librosa.load(self.silence_audio_path, sr=self.output_sample_rate, mono=True)
                        feature_audio, _ = librosa.load(self.silence_audio_path, sr=self.feature_sample_rate, mono=True)
                        hubert_features = self.hubert_extractor.extract_from_file(self.silence_audio_path)
                        
                        # 创建静默音频批次
                        for i in range(min(len(hubert_features), self.synthesis_frame_count)):
                            # 低采样率音频块
                            audio_start = i * self.samples_per_frame
                            if audio_start + self.samples_per_frame > len(feature_audio):
                                audio_chunk = np.pad(feature_audio[audio_start:], 
                                                   (0, self.samples_per_frame - (len(feature_audio) - audio_start)))
                            else:
                                audio_chunk = feature_audio[audio_start:audio_start + self.samples_per_frame]
                            
                            # 高采样率音频块
                            original_start = i * self.output_samples_per_frame
                            if original_start + self.output_samples_per_frame > len(original_audio):
                                original_chunk = np.pad(original_audio[original_start:], 
                                                      (0, self.output_samples_per_frame - (len(original_audio) - original_start)))
                            else:
                                original_chunk = original_audio[original_start:original_start + self.output_samples_per_frame]
                            
                            batch_frames.append(AudioBatch(
                                hubert_feature=hubert_features[i],
                                audio_chunk=audio_chunk,
                                original_audio_chunk=original_chunk,
                                audio_path=self.silence_audio_path,
                                is_sync_frame=True  # 静默音频默认使用同步
                            ))
                        
                        current_audio_path = self.silence_audio_path
                    
                    # 准备特征数据
                    hubert_features = np.stack([frame.hubert_feature for frame in batch_frames])
                    
                    # 生成合成帧，传递同步标志
                    frame_generator = self.synthesizer.iterate_synthesized_frames(
                        features=hubert_features,
                        start_frame_idx=last_frame_index,
                        is_generate_sync_frame=is_sync_frame  # 传递同步标志
                    )
                    
                    # 处理生成的帧
                    for i, frame_info in enumerate(frame_generator):
                        if i < len(batch_frames):
                            # 使用高采样率音频进行播放
                            frame_info['audio'] = batch_frames[i].original_audio_chunk
                            self.player.upload_frame(frame_info)
                
                threading.Event().wait(0.001)
            except Exception as e:
                # 添加关键错误捕获，输出详细错误信息但继续循环
                print(f"视频帧处理发生错误: {e}")
                import traceback
                print(traceback.format_exc())
                # 短暂暂停避免疯狂打印错误
                time.sleep(1)
                
    def get_queue_lengths(self):
        """获取各个队列中的音频片段数量"""
        return {name: queue.qsize() for name, queue in self.audio_queues.items()}
    
    def process_single_file(self, audio_path: str, output_path: str):
        """
        处理单个音频文件并生成短视频
        Args:
            audio_path: 输入音频文件路径
            output_path: 输出视频路径
        Returns:
            输出视频路径
        """
        import cv2
        import time
        
        print(f"开始处理音频文件: {audio_path}")
        start_time = time.time()
        
        # 1. 提取Hubert特征
        print("正在提取音频特征...")
        hubert_features = self.hubert_extractor.extract_from_file(audio_path)
        print(f"提取完成，共 {len(hubert_features)} 帧")
        
        # 2. 使用合成器直接生成所有帧
        print("开始生成视频帧...")
        all_frames = []
        
        # 直接调用合成器的迭代方法，传入所有特征
        frame_generator = self.synthesizer.iterate_synthesized_frames(
            features=hubert_features,
            start_frame_idx=0,
            is_generate_sync_frame=True
        )
        
        # 收集所有生成的帧
        frame_count = 0
        for frame_info in frame_generator:
            all_frames.append(frame_info['frame'])
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count} 帧")
        
        print(f"共生成 {len(all_frames)} 帧")
        
        # 3. 保存视频文件
        print("所有帧处理完毕，开始保存视频...")
        if len(all_frames) > 0:
            # 获取第一帧的尺寸
            height, width = all_frames[0].shape[:2]
            fps = self.fps
            
            # 创建临时视频
            temp_video_path = output_path.replace('.mp4', '_temp.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            # 写入所有帧
            for frame in all_frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            # 合并音频
            print("合并音频与视频...")
            command = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{output_path}"'
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # 删除临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            print(f"视频生成完成，总耗时: {time.time() - start_time:.2f}秒")
            print(f"输出文件: {output_path}")
            return output_path
        else:
            raise ValueError("未能生成任何视频帧")
    

import os
import threading
from queue import Queue
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from image_infer_v1.utils.hubert_extractor import HubertExtractor
from image_infer_v1.tools.frame_synthesizer.infer_api import FrameSynthesizer
import librosa
import time

class VideoStreamManager:
    def __init__(self, 
                 data_dir: str,
                 unet_checkpoint: str,
                 hubert_path: str,
                 device: str = "cuda:0",
                 batch_size: int = 8,
                 output_sample_rate: int = 24000,):  # 添加是否打断参数
        """初始化视频流管理器"""
        # 初始化合成器
        print(data_dir)
        self.synthesizer = FrameSynthesizer(
            unet_checkpoint=unet_checkpoint,
            data_dir=data_dir,
            device=device,
            batch_size=batch_size
        )
        
        # 初始化Hubert特征提取器
        self.hubert_extractor = HubertExtractor(hubert_path, device)
        # 基础音频参数
        self.feature_sample_rate = 16000  # Hubert特征提取采样率
        self.output_sample_rate = output_sample_rate  # 输出/播放采样率
        self.fps = 25
    
        
        # 创建临时目录
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
         

        
    
    
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
    


if __name__ == "__main__":
    data_dir="E:/unetsync/train_res/example5/infer_data"
    unet_checkpoint="E:/unetsync/train_res/example5/weights/checkpoint_epoch_20.pth"
    hubert_path="E:/unetsync/pretrained_models/hubert_checkpoints"
    video_streamer = VideoStreamManager(
        data_dir=data_dir,
        unet_checkpoint=unet_checkpoint,
        hubert_path=hubert_path,
    )
    video_streamer.process_single_file(audio_path="E:/unetsync/eng_test3.wav", output_path="E:/unetsync/result_en7.mp4")
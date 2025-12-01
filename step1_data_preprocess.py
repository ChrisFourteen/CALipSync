import os
import cv2
import torch
import numpy as np
import av
from concurrent.futures import ThreadPoolExecutor
import subprocess
import librosa
import json
from utils.lip_detector.lip_detector import LipDetector
from utils.hubert_extractor import HubertExtractor
from typing import Optional, List, Tuple

class VideoPreprocessor:
    def __init__(self, weight_base_dir: str, hubert_path: str):
        self.weight_base_dir = weight_base_dir
        self.hubert_extractor = HubertExtractor(hubert_path)
        self.lip_detector = LipDetector(weight_base_dir)
        self.batch_size = 64  # 处理批次大小
        self.frame_buffer_size = 64  # 视频帧缓冲区大小
        self.device = torch.device("cuda:0")

    def frame_generator(self, video_path: str, target_fps: int = 25):
        """
        生成器函数，处理视频帧
        返回: frames tensor [N, H, W, C]
        """
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        # 优化视频解码
        video_stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO
        video_stream.codec_context.thread_count = 32
        
        frames_buffer = []
        
        for frame in container.decode(video=0):
            try:
                img_array = frame.to_ndarray(format='rgb24')
                if img_array.shape[0] > 0 and img_array.shape[1] > 0:
                    img = torch.from_numpy(img_array)
                    frames_buffer.append(img)
                    
                    if len(frames_buffer) >= self.frame_buffer_size:
                        frames_batch = torch.stack(frames_buffer).to(self.device)
                        yield frames_batch
                        frames_buffer = []
                        
            except Exception as e:
                print(f"Warning: Error processing frame: {e}")
        
        if frames_buffer:
            frames_batch = torch.stack(frames_buffer).to(self.device)
            yield frames_batch

        container.close()

    def process_batch(self, frames: torch.Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        处理一批帧，返回关键点和裁剪后的人脸图像
        返回: [(landmarks, face_crop), ...]
        """
        # 将frames转换为CPU上的numpy数组供检测器使用
        frames_cpu = frames.cpu().numpy().astype(np.uint8)
        
        # 确保每个帧都是有效的
        valid_frames = []
        for frame in frames_cpu:
            if frame.shape[0] > 0 and frame.shape[1] > 0 and frame.shape[2] == 3:
                valid_frames.append(frame)
            else:
                print(f"Warning: Invalid frame shape: {frame.shape}")
        
        if not valid_frames:
            return [(None, None)] * len(frames)
            
        landmarks_list = self.lip_detector.detect_landmarks(valid_frames)
        
        results = []
        for frame, landmarks in zip(frames_cpu, landmarks_list):
            if landmarks is not None and len(landmarks) > 0:
                # 提取人脸区域
                xmin = landmarks[0][1][0]
                ymin = landmarks[0][52][1]
                xmax = landmarks[0][31][0]
                width = xmax - xmin
                ymax = ymin + width
                
                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (168, 168), cv2.INTER_AREA)
                    face_crop = face_crop[4:164, 4:164].copy()
                    results.append((landmarks[0], face_crop))
                else:
                    results.append((None, None))
            else:
                results.append((None, None))
        
        return results

    def save_results(self, results: List[Tuple[np.ndarray, np.ndarray]], 
                    frame_indices: List[int], output_dir: str):
        """并行保存处理结果"""
        def save_item(args):
            idx, (landmarks, face_crop) = args
            if landmarks is not None and face_crop is not None:
                # 保存关键点
                landmark_path = os.path.join(output_dir, 'landmarks', f'{idx}.lms')
                np.savetxt(landmark_path, landmarks, fmt='%d')
                
                # 保存人脸图像
                face_path = os.path.join(output_dir, 'faces', f'{idx}.jpg')
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(face_path, face_crop)
        
        with ThreadPoolExecutor(max_workers=64) as executor:
            executor.map(save_item, zip(frame_indices, results))

    def fix_missing_landmarks(self, output_dir: str):
        """补齐缺失的landmarks文件"""
        landmarks_dir = os.path.join(output_dir, 'landmarks')
        all_files = os.listdir(os.path.join(output_dir, 'full_body_img'))
        max_frame = max([int(f.split('.')[0]) for f in all_files])
        
        # 检查每一帧
        for i in range(max_frame + 1):
            lms_path = os.path.join(landmarks_dir, f'{i}.lms')
            if not os.path.exists(lms_path):
                print(f"发现缺失的landmarks文件: {i}.lms")
                
                # 向前和向后查找最近的存在的landmarks文件
                prev_idx = i - 1
                next_idx = i + 1
                reference_path = None
                
                while prev_idx >= 0 or next_idx <= max_frame:
                    # 优先使用前一帧
                    if prev_idx >= 0:
                        prev_path = os.path.join(landmarks_dir, f'{prev_idx}.lms')
                        if os.path.exists(prev_path):
                            reference_path = prev_path
                            break
                    
                    # 其次使用后一帧
                    if next_idx <= max_frame:
                        next_path = os.path.join(landmarks_dir, f'{next_idx}.lms')
                        if os.path.exists(next_path):
                            reference_path = next_path
                            break
                    
                    prev_idx -= 1
                    next_idx += 1
                
                if reference_path:
                    print(f"使用参考文件 {reference_path} 补齐缺失文件")
                    import shutil
                    shutil.copy2(reference_path, lms_path)
                else:
                    print(f"警告: 无法找到合适的参考文件来补齐 {i}.lms")

    def process_video(self, video_path: str, output_dir: str):
        """优化后的主处理流程"""
        os.makedirs(output_dir, exist_ok=True)
        for dir_path in ['landmarks', 'faces', 'full_body_img']:
            os.makedirs(os.path.join(output_dir, dir_path), exist_ok=True)
        
        print("\n开始处理视频...")
        
        # 1. 处理音频
        print("├── 处理音频...")
        audio_path = os.path.join(output_dir, 'aud.wav')
        subprocess.run([
            'ffmpeg','-y', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', 
            audio_path
        ])

        # 加载音频数据并检查音量
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # 检查音频音量
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        db = 20 * np.log10(mean_rms) if mean_rms > 0 else -100
        
        if db < -150:
            raise Exception("音量太小，未识别到说话人")

        # 提取特征
        hubert_features = self.hubert_extractor.extract(audio)
        np.save(os.path.join(output_dir, 'aud_hu.npy'), hubert_features)
        
        # 2. 分批处理视频帧
        print("├── 开始分批处理视频帧...")
        import time
        start_time = time.time()
        total_frames_processed = 0
        
        # 使用生成器逐批次处理帧
        for batch_idx, frames_batch in enumerate(self.frame_generator(video_path)):
            batch_start_time = time.time()
            print(f"├── 处理第 {batch_idx + 1} 批次 ({len(frames_batch)} 帧)...")
            
            # 保存完整帧
            start_idx = total_frames_processed
            batch_indices = list(range(start_idx, start_idx + len(frames_batch)))
            
            def save_full_frame(args):
                idx, frame = args
                frame_cpu = frame.cpu().numpy()
                frame_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, 'full_body_img', f'{idx}.jpg')
                cv2.imwrite(output_path, frame_bgr)
            
            print("   ├── 保存完整帧...")
            with ThreadPoolExecutor(max_workers=64) as executor:
                executor.map(save_full_frame, zip(batch_indices, frames_batch))
            
            # 在批次内继续使用小批次进行处理
            print("   ├── 进行人脸检测和关键点提取...")
            for i in range(0, len(frames_batch), self.batch_size):
                sub_batch = frames_batch[i:i + self.batch_size]
                sub_batch_indices = batch_indices[i:i + self.batch_size]
                
                # 处理小批次
                batch_results = self.process_batch(sub_batch)
                
                # 保存结果
                self.save_results(batch_results, sub_batch_indices, output_dir)
            
            total_frames_processed += len(frames_batch)
            batch_time = time.time() - batch_start_time
            print(f"   └── 批次处理完成，耗时 {batch_time:.2f} 秒")
            
            # 手动清理GPU内存
            del frames_batch
            torch.cuda.empty_cache()
        
        print("\n开始检查并补齐缺失的landmarks文件...")
        self.fix_missing_landmarks(output_dir)
        print("补齐操作完成！")

        process_time = time.time() - start_time
        print(f"└── 处理完成！共处理 {total_frames_processed} 帧，总耗时 {process_time:.2f} 秒")

        return {
            'total_frames': total_frames_processed,
            'process_time': process_time,
            'output_dir': output_dir
        }
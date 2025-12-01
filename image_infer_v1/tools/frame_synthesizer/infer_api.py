import os
import cv2
import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Iterator, Dict
from concurrent.futures import ThreadPoolExecutor
from image_infer_v1.models.unet import Model
import time

class FrameSynthesizer:
    def __init__(self, unet_checkpoint: str, data_dir: str, 
                 device: str = "cuda:0", batch_size: int = 8):
        """
        初始化帧合成器
        Args:
            unet_checkpoint: UNet模型检查点路径
            data_dir: 预处理数据目录
            device: 计算设备
            batch_size: 批处理大小
        """
        self.device = device
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # 初始化数据路径
        print(data_dir)
        self.frames_dir = os.path.join(data_dir, "frames")
        self.positions_dir = os.path.join(data_dir, "positions")
        self.masks_dir = os.path.join(data_dir, "masks")
        
        # 获取总帧数
        self.total_frames = len([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        print("总帧数", self.total_frames)
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=self.batch_size)
        
        # 初始化UNet模型
        self.net = Model(6, "hubert").to(device)
        self.net.load_state_dict(torch.load(unet_checkpoint))
        self.net.eval()

        # 初始化动作泛化相关的状态
        self.current_direction = None  # 当前播放方向
        self.target_frame_count = 0    # 当前方向的目标帧数
        self.processed_frame_count = 0  # 当前方向已处理的帧数
        self.current_frame_position = 0 # 当前物理帧位置
        self.last_logical_index = -1    # 维护连续的逻辑帧序号
    
    def _load_single_frame(self, frame_idx: int) -> tuple:
        """加载单帧数据，包括图像、关键点和遮罩"""
        frame_number = str(frame_idx % self.total_frames).zfill(6)
        
        # 加载帧图像
        frame_path = os.path.join(self.frames_dir, f"{frame_number}.jpg")
        img = cv2.imread(frame_path)
        
        # 加载关键点
        position_path = os.path.join(self.positions_dir, f"{frame_number}.txt")
        lms = np.loadtxt(position_path)
        
        # 加载遮罩
        mask_path = os.path.join(self.masks_dir, f"{frame_number}.jpg")
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = mask.astype(np.float32) / 255.0
        
        return img, lms, mask
    
    def _load_batch_frames(self, frame_indices: list) -> tuple:
        """
        并行加载一批帧数据
        Args:
            frame_indices: 要加载的帧索引列表
        Returns:
            (batch_images, batch_landmarks, batch_masks) 元组
        """
        # 使用进程池并行加载
        futures = []
        for frame_idx in frame_indices:
            futures.append(self.executor.submit(self._load_single_frame, frame_idx))
        
        # 收集结果
        batch_images = []
        batch_landmarks = []
        batch_masks = []
        for future in futures:
            img, lms, mask = future.result()
            batch_images.append(img)
            batch_landmarks.append(lms)
            batch_masks.append(mask)
                
        return batch_images, batch_landmarks, batch_masks

    def _get_audio_features(self, features: np.ndarray, indices: list) -> np.ndarray:
        """获取一批音频特征，确保输出与输入索引数量一致"""
        batch_features = []
        
        # 为每个索引创建对应的特征
        for idx in indices:
            # 默认特征，如果处理失败会使用这个
            default_feature = np.zeros((32, 32, 32), dtype=np.float32)
            feature_added = False
            
            # 尝试处理音频特征
            try:
                left = idx - 8
                right = idx + 8
                pad_left = 0
                pad_right = 0
                
                if left < 0:
                    pad_left = -left
                    left = 0
                if right > features.shape[0]:
                    pad_right = right - features.shape[0]
                    right = features.shape[0]
                
                auds = torch.from_numpy(features[left:right])
                
                if pad_left > 0:
                    auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
                if pad_right > 0:
                    auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
                
                # 检查处理后的数据是否足够reshape为(32,32,32)
                total_elements = auds.numel()
                if total_elements >= 32*32*32:
                    feat = auds.reshape(32, 32, 32).numpy()
                    batch_features.append(feat)
                    feature_added = True
            except Exception as e:
                # 记录错误但继续处理
                pass
            
            # 如果当前索引没有成功添加特征，使用默认特征
            if not feature_added:
                batch_features.append(default_feature)
        
        # 确保结果是一个numpy数组，形状为[len(indices), 32, 32, 32]
        return np.array(batch_features)

    def _generate_frame_sequence(self, needed_frames: int) -> list:
        """
        生成帧序列，实现动作泛化的核心逻辑
        Args:
            needed_frames: 需要生成的帧数量
        Returns:
            包含物理帧索引的列表
        """
        frame_sequence = []
        
        # 如果已经达到目标帧数或没有方向，重新随机
        if self.processed_frame_count >= self.target_frame_count or self.current_direction is None:
            # 重新随机：设定新的目标帧数为总帧数的5%-15%
            self.target_frame_count = self.total_frames * random.randint(5, 15) // 100
            self.current_direction = random.choice([1, -1])
            self.processed_frame_count = 0
            
        # 使用当前方向继续播放
        while len(frame_sequence) < needed_frames:
            # 计算当前方向可用的帧数
            if self.current_direction == 1:
                available_frames = self.total_frames - self.current_frame_position
            else:
                available_frames = self.current_frame_position + 1
                
            seq_length = min(available_frames, needed_frames - len(frame_sequence))
            
            # 在当前方向上收集帧
            for _ in range(seq_length):
                frame_sequence.append(self.current_frame_position)
                self.current_frame_position += self.current_direction
                
                # 处理边界情况
                if self.current_frame_position >= self.total_frames:
                    self.current_frame_position = self.total_frames - 2
                    self.current_direction = -1
                elif self.current_frame_position < 0:
                    self.current_frame_position = 1
                    self.current_direction = 1
                    
        # 更新已处理的帧数
        self.processed_frame_count += len(frame_sequence)
        
        return frame_sequence
    
    def process_batch(self, batch_images: list, batch_landmarks: list, 
                      batch_masks: list, hubert_features: np.ndarray) -> list:
        """处理一批图像"""
        try:
            batch_size = len(batch_images)
            processed_frames = []
            batch_inputs = []
            
            for i in range(batch_size):
                img = batch_images[i].copy()
                lms = batch_landmarks[i]
                mask = batch_masks[i]
                
                # 计算裁剪区域
                xmin = int(lms[1][0])
                ymin = int(lms[52][1])
                xmax = int(lms[31][0])
                width = xmax - xmin
                
                # 计算正方形区域的ymax，确保高宽相等
                target_height = width
                ymax = ymin + target_height
                
                # 安全检查：确保不超出图片边界
                height, img_width = img.shape[:2]
                if ymax > height:
                    # 如果超出底部边界，向上调整
                    diff = ymax - height
                    ymax = height
                    ymin = max(0, ymin - diff)
                
                if ymin < 0:
                    # 如果超出顶部边界，向下调整
                    ymax = min(height, ymax - ymin)
                    ymin = 0
                    
                if xmin < 0:
                    xmin = 0
                if xmax > img_width:
                    xmax = img_width
                
                # 裁剪和调整图像
                crop_img = img[ymin:ymax, xmin:xmax]
                crop_img = cv2.resize(crop_img, (168, 168))
                
                # 准备模型输入
                img_real_ex = crop_img[4:164, 4:164].copy()
                img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 150, 145), (0, 0, 0), -1)
                
                img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0
                img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
                
                img_concat = np.concatenate([img_real_ex, img_masked])
                batch_inputs.append(img_concat)
                
                processed_frames.append({
                    'original_img': img,
                    'landmarks': lms,
                    'crop_params': (ymin, ymax, xmin, xmax, width),
                    'original_crop': crop_img,
                    'mask': mask
                })
            
            # 模型推理
            batch_tensor = torch.from_numpy(np.stack(batch_inputs)).to(self.device)
            hubert_tensor = torch.from_numpy(hubert_features).to(self.device)
            
            with torch.no_grad():
                predictions = self.net(batch_tensor, hubert_tensor)
            
            # 处理预测结果
            results = []
            for i in range(batch_size):
                pred = predictions[i].cpu().numpy().transpose(1, 2, 0) * 255
                pred = np.array(pred, dtype=np.uint8)
                
                frame_data = processed_frames[i]
                img = frame_data['original_img']
                lms = frame_data['landmarks']
                ymin, ymax, xmin, xmax, width = frame_data['crop_params']
                crop_img = frame_data['original_crop'].copy()
                mask = frame_data['mask']  # 获取遮罩
                
                # 更新预测结果
                crop_img[4:164, 4:164] = pred
                crop_img = cv2.resize(crop_img, (width, width))
                
                # 创建面部区域遮罩
                face_mask = np.zeros((ymax-ymin, xmax-xmin), dtype=np.uint8)
                face_points = lms[:33].copy()
                face_points[:, 0] -= xmin
                face_points[:, 1] -= ymin
                
                scale_x = width / (xmax - xmin)
                scale_y = width / (ymax - ymin)
                face_points[:, 0] *= scale_x
                face_points[:, 1] *= scale_y
                face_points = face_points.astype(np.int32)
                
                cv2.fillPoly(face_mask, [face_points], 255)
                
                # 修改点1: 基于面积的等比例外扩
                mask_area = np.sum(face_mask > 0)
                mask_radius = np.sqrt(mask_area / np.pi)  # 估算遮罩等效半径
                expand_pixels = int(mask_radius * 0.15)  # 外扩5%
                # 确保扩展像素至少为1
                expand_pixels = max(1, expand_pixels)
                
                kernel = np.ones((expand_pixels * 2 + 1, expand_pixels * 2 + 1), np.uint8)
                expanded_mask = cv2.dilate(face_mask, kernel, iterations=1)
                
                height, width = face_mask.shape
                border_mask = np.zeros((height + 2, width + 2), np.uint8)
                border_mask[1:-1, 1:-1] = 255
                final_face_mask = cv2.bitwise_and(expanded_mask, border_mask[1:-1, 1:-1])
                
                # 修改点2: 切除顶部15%区域
                #top_region = int(height * 0.15)
                top_region = int(height * 0)
                final_face_mask[:top_region, :] = 0
                
                # 将面部掩码转换为浮点数并扩展为3通道
                final_face_mask_float = final_face_mask / 255.0
                final_face_mask_3ch = np.repeat(final_face_mask_float[..., np.newaxis], 3, axis=2)
                
                # 添加安全检查：如果形状不匹配，直接返回原始图像
                try:
                    # 检查形状是否匹配
                    target_shape = img[ymin:ymax, xmin:xmax].shape
                    if crop_img.shape != target_shape:
                        # 形状不匹配，直接将原始图像添加到结果中
                        results.append(img)
                        continue  # 跳过后续融合步骤
                        
                    # 如果形状匹配，继续执行原来的融合逻辑
                    if mask is not None:
                        # 将遮罩调整为与裁剪区域相同大小
                        resized_mask = cv2.resize(mask, (crop_img.shape[1], crop_img.shape[0]))
                        # 将遮罩转换为3通道
                        resized_mask_3ch = np.repeat(resized_mask[..., np.newaxis], 3, axis=2)
                        
                        # 使用面部掩码和遮罩的组合来决定哪些区域应该保留原始图像
                        inverted_mask = 1.0 - resized_mask_3ch
                        
                        # 结合面部掩码和反转遮罩
                        combined_mask = final_face_mask_3ch * (1.0 - inverted_mask)
                        
                        # 应用组合遮罩
                        result = (crop_img * combined_mask) + (img[ymin:ymax, xmin:xmax] * (1.0 - combined_mask))
                    else:
                        # 如果没有遮罩，只使用面部掩码
                        result = (crop_img * final_face_mask_3ch) + (img[ymin:ymax, xmin:xmax] * (1.0 - final_face_mask_3ch))
                    
                    img[ymin:ymax, xmin:xmax] = result
                    results.append(img)
                except Exception as e:
                    # 任何错误发生都直接使用原始图像
                    results.append(img)
            
            return results
        except Exception as e:
            print(f"视频帧批处理失败: {e}")
            import traceback
            print(traceback.format_exc())
            # 返回原始图像作为后备
            return batch_images

    def iterate_synthesized_frames(self, features: np.ndarray, start_frame_idx: int = 0, is_generate_sync_frame: bool = True) -> Iterator[Dict]:
        """
        生成合成帧迭代器 - 集成了动作泛化机制，使用批处理提升性能
        Args:
            features: 预处理好的hubert特征
            start_frame_idx: 起始帧索引
            is_generate_sync_frame: 是否进行口型同步，默认为True。如果为False，则直接返回原始帧
        Returns:
            包含frame和index的字典迭代器
        """
        # 设置起始逻辑帧序号
        self.last_logical_index = start_frame_idx - 1
        
        # 初始化计时器
        time_stats = {
            'load_frame': 0.0,
            'get_audio': 0.0,
            'process_batch': 0.0
        }
        
        try:
            # 按batch_size处理数据
            total_frames = len(features)
            for batch_start in range(0, total_frames, self.batch_size):
                try:
                    # 确定当前批次大小
                    batch_end = min(batch_start + self.batch_size, total_frames)
                    current_batch_size = batch_end - batch_start
                    
                    # 生成一批物理帧序号
                    frame_sequence = self._generate_frame_sequence(current_batch_size)
                    
                    # 使用优化后的批量加载
                    t_start = time.time()
                    batch_images, batch_landmarks, batch_masks = self._load_batch_frames(frame_sequence)
                    time_stats['load_frame'] += time.time() - t_start
                    
                    # 如果不需要进行口型同步，直接返回原始帧
                    if not is_generate_sync_frame:
                        # 逐帧返回结果，不进行口型同步处理
                        for i, original_image in enumerate(batch_images):
                            self.last_logical_index += 1
                            yield {
                                'frame': original_image,
                                'index': self.last_logical_index,
                                'physical_index': frame_sequence[i]
                            }
                        continue
                    
                    # 以下是需要进行口型同步的情况，保持原有逻辑
                    
                    # 批量获取音频特征
                    t_start = time.time()
                    feature_indices = list(range(batch_start, batch_end))
                    batch_features = self._get_audio_features(features, feature_indices)
                    time_stats['get_audio'] += time.time() - t_start
                    
                    # 批量处理
                    t_start = time.time()
                    processed_frames = self.process_batch(batch_images, batch_landmarks, batch_masks, batch_features)
                    time_stats['process_batch'] += time.time() - t_start
                    
                    # 逐帧返回结果
                    for i, processed_frame in enumerate(processed_frames):
                        self.last_logical_index += 1
                        yield {
                            'frame': processed_frame,
                            'index': self.last_logical_index,
                            'physical_index': frame_sequence[i]
                        }
                except Exception as e:
                    # 捕获批处理内部的异常，但允许迭代器继续处理后续批次
                    print(f"处理视频帧批次时出错: {e}")
                    import traceback
                    print(traceback.format_exc())
                    # 暂停一下避免循环过快
                    time.sleep(0.1)
                    continue
                
        except Exception as e:
            print(f"视频帧生成迭代器发生致命错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 确保在异常情况下也能返回一些东西，避免主线程崩溃
            self.last_logical_index += 1
            yield {'frame': np.zeros((480, 640, 3), dtype=np.uint8), 
                   'index': self.last_logical_index, 
                   'physical_index': 0}
        finally:
            # 计算并打印统计信息
            total_time = sum(time_stats.values())
            if 'total_frames' in locals() and total_time > 0:
                print(f"平均帧生成速度: {total_frames/total_time:.2f} FPS")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
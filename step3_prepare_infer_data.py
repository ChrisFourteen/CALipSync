import os
import cv2
import numpy as np
import subprocess
#import onnxruntime as ort
from utils.lip_detector.lip_detector import LipDetector
#from bin.image_clone_v1.utils.get_srt.slice_origin_audio import process_and_split_audio

class VideoPreprocessor:
    def __init__(self, lip_detect_weight_base_dir, asr_weight_path, xseg_model_path):
        self.lip_detect_weight_base_dir = lip_detect_weight_base_dir
        self.asr_weight_path = asr_weight_path
        self.xseg_model_path = xseg_model_path
        self.lip_detector = LipDetector(lip_detect_weight_base_dir)
        self.batch_size = 32
        
        # 初始化ONNX模型会话
        #self.xseg_session = ort.InferenceSession(xseg_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model_size = (256, 256)  # XSeg模型输入尺寸
        
        # 打印遮罩模型使用的设备类型
        # session_providers = self.xseg_session.get_providers()
        # if 'CUDAExecutionProvider' in session_providers:
        #     print("遮罩模型正在使用GPU进行推理")
        # else:
        #     print("遮罩模型正在使用CPU进行推理")
        
    def generate_mask(self, frame, landmarks):
        """为帧生成面部遮罩"""
        frame_height, frame_width = frame.shape[:2]
        
        # 根据关键点确定下半脸区域
        xmin = landmarks[1][0]
        ymin = landmarks[52][1]
        xmax = landmarks[31][0]
        width = xmax - xmin
        
        # 计算正方形区域的ymax，确保高宽相等
        target_height = width
        ymax = ymin + target_height
        
        # 计算扩展区域
        expand_up_px = int(width)  # 向上扩展100%宽度
        expand_sides_px = int(width)  # 向两侧扩展100%宽度
        expand_down_px = int(width * 0.5)  # 向下扩展50%宽度
        
        # 计算扩展后的边界（确保不超出图像边界）
        new_xmin = max(0, int(xmin - expand_sides_px))
        new_ymin = max(0, int(ymin - expand_up_px))
        new_xmax = min(frame_width, int(xmax + expand_sides_px))
        new_ymax = min(frame_height, int(ymax + expand_down_px))
        
        # 裁剪扩展后的区域
        expanded_image = frame[new_ymin:new_ymax, new_xmin:new_xmax]
        
        # 调整图像大小为模型输入尺寸
        resized_image = cv2.resize(expanded_image, self.model_size)
        
        # 图像预处理
        prepare_image = resized_image.astype(np.float32) / 255.0
        prepare_image = np.expand_dims(prepare_image, axis=0)
        
        # 模型推理
        outputs = self.xseg_session.run(None, {'input': prepare_image})
        mask = outputs[0][0]
        
        # 后处理
        mask = mask.transpose(0, 1, 2)
        mask = mask.clip(0, 1)
        
        # 将mask调整回扩展区域的原始尺寸
        expanded_mask = cv2.resize(mask, (expanded_image.shape[1], expanded_image.shape[0]))
        
        # 计算原始下半脸区域在扩展图像中的位置
        original_x1 = int(xmin - new_xmin)
        original_y1 = int(ymin - new_ymin)
        original_x2 = int(xmax - new_xmin)
        original_y2 = int(ymax - new_ymin)
        
        # 提取原始下半脸区域对应的遮罩部分
        if original_y1 < expanded_mask.shape[0] and original_x1 < expanded_mask.shape[1] and original_y2 <= expanded_mask.shape[0] and original_x2 <= expanded_mask.shape[1]:
            cropped_mask = expanded_mask[original_y1:original_y2, original_x1:original_x2]
            
            # 调整为168x168大小，与lips_jpg一致
            final_mask = cv2.resize(cropped_mask, (168, 168))
            return final_mask
        else:
            # 如果坐标计算有问题，返回空白遮罩
            return np.ones((168, 168), dtype=np.float32)
    
    def process_frames_batch(self, frames, frame_indices, infer_dir):
        # 将OpenCV读取的BGR图像批量转换为RGB，以匹配训练时使用的数据格式
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        landmarks_list = self.lip_detector.detect_landmarks(rgb_frames)
        
        for landmarks, frame_idx, frame in zip(landmarks_list, frame_indices, frames):
            frame_number = str(frame_idx).zfill(6)
            
            if landmarks is not None and len(landmarks) > 0:
                # 1. 保存帧为jpg
                cv2.imwrite(os.path.join(infer_dir, 'frames', f'{frame_number}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 2. 保存关键点为txt
                np.savetxt(os.path.join(infer_dir, 'positions', f'{frame_number}.txt'), landmarks[0])
                
                # 3. 保存唇部图像
                xmin = landmarks[0][1][0]
                ymin = landmarks[0][52][1]
                xmax = landmarks[0][31][0]
                width = xmax - xmin
                
                mouth_crop = frame[int(ymin):int(ymin+width), int(xmin):int(xmax)]
                if mouth_crop.size > 0:
                    crop_img = cv2.resize(mouth_crop, (168, 168))
                    cv2.imwrite(os.path.join(infer_dir, 'lips_jpg', f'{frame_number}.jpg'), crop_img)
                    
                    # # 4. 生成并保存唇部遮罩
                    # mask = self.generate_mask(frame, landmarks[0])
                    # mask_img = (mask * 255).astype(np.uint8)
                    # cv2.imwrite(os.path.join(infer_dir, 'masks', f'{frame_number}.jpg'), mask_img)

    def process_video(self, video_path, output_dir):
        """处理视频并提取音频文本信息"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建srt目录用于存放切分后的音频和文本
        srt_dir = os.path.join(output_dir, 'srt')
        os.makedirs(srt_dir, exist_ok=True)
        
        # 读取第一帧并保存为sample.jpg
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, 'sample.jpg'), first_frame)
        cap.release()
        
        # 创建推理数据目录
        infer_dir = os.path.join(output_dir, 'infer_data')
        for dir_path in ['frames', 'positions', 'lips_jpg', 'masks']:  # 添加masks目录
            os.makedirs(os.path.join(infer_dir, dir_path), exist_ok=True)
        
        # # 提取音频并进行语音识别和切分
        # print("开始提取音频并进行识别...")
        # process_and_split_audio(
        #     audio_path=video_path,  # 直接使用原始视频路径，函数内部会提取音频
        #     local_model_dir=self.asr_weight_path,
        #     output_dir=srt_dir,
        #     min_seconds=5,
        #     max_seconds=12
        # )
        # print(f"音频切分完成，结果保存在: {srt_dir}")

        # 读取并处理视频帧
        cap = cv2.VideoCapture(video_path)
        frames_buffer = []
        frame_indices = []
        frame_idx = 0
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_batches = (total_frames + self.batch_size - 1) // self.batch_size
        current_batch = 1
        
        print(f"开始处理视频帧... 总帧数: {total_frames}, 总批数: {total_batches}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_buffer.append(frame)
            frame_indices.append(frame_idx)
            
            if len(frames_buffer) >= self.batch_size:
                print(f"开始处理第 {current_batch}/{total_batches} 批帧")
                self.process_frames_batch(frames_buffer, frame_indices, infer_dir)
                frames_buffer = []
                frame_indices = []
                current_batch += 1
            
            frame_idx += 1
        
        # 处理剩余帧
        if frames_buffer:
            print(f"开始处理第 {current_batch}/{total_batches} 批帧（最后一批）")
            self.process_frames_batch(frames_buffer, frame_indices, infer_dir)
        
        cap.release()
        print(f"处理完成！共处理 {frame_idx} 帧")
        
        # 返回srt目录路径，方便后续使用
        return {
            "infer_dir": infer_dir,
            "srt_dir": srt_dir,
            "processed_frames": frame_idx
        }
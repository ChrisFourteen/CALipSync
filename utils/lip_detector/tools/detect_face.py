import torch
import numpy as np
from .s3fd.main import S3FD

class S3FDFaceDetector:
    def __init__(self, weight_path, conf_threshold=0.1, nms_threshold=0.5):
        """
        初始化S3FD人脸检测器，与YoloFaceDetector接口保持一致
        
        Args:
            weight_path: S3FD模型权重路径
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值（S3FD内部使用）
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 初始化S3FD检测器
        self.det_net = S3FD(
            weight_path=weight_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 记录上一帧的检测结果
        self.last_detection = None

    def detect(self, images):
        """
        批量检测人脸，与YoloFaceDetector接口保持一致
        
        Args:
            images: 图片列表[img1, img2, ...], 每张图片格式为(H,W,3)的RGB图像
            
        Returns:
            detections: 列表，每个元素为(bboxes, indices)
                bboxes: numpy数组，格式为[x1,y1,w,h]
                indices: 索引列表
        """
        detections = []

        for img in images:
            # 使用S3FD检测人脸
            bboxes = self.det_net.detect_faces(
                img,
                conf_th=self.conf_threshold,
                scales=[0.25]  # 使用单一尺度以提高速度
            )
            
            if len(bboxes) == 0:
                if self.last_detection is None:
                    print("未检测到人脸,且没有历史检测结果")
                    detections.append((np.array([]), []))
                else:
                    print("未检测到人脸,使用上一帧的检测结果")
                    detections.append(self.last_detection)
                continue
            
            # S3FD输出格式为[x1,y1,x2,y2,conf]
            # 转换为与YoloFaceDetector输出格式一致的结果[x1,y1,w,h]
            bboxes_np = np.array([box[:-1] for box in bboxes])  # 去掉置信度
            
            # 转换为[x1,y1,w,h]格式
            converted_bboxes = np.column_stack((
                bboxes_np[:, :2],  # x1,y1
                bboxes_np[:, 2:] - bboxes_np[:, :2]  # w,h (x2-x1, y2-y1)
            ))
            
            indices = list(range(len(bboxes)))
            current_detection = (converted_bboxes, indices)
            
            # 更新最后一次检测结果
            self.last_detection = current_detection
            detections.append(current_detection)

        return detections

    def release(self):
        """释放资源"""
        if hasattr(self, 'det_net'):
            del self.det_net
            torch.cuda.empty_cache()
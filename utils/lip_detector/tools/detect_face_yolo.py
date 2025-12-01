from ultralytics import YOLO
import numpy as np

class YoloFaceDetector:
    def __init__(self, weight_path, conf_threshold=0.1, nms_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.det_net = YOLO(weight_path)
        # 记录上一帧的检测结果
        self.last_detection = None

    def detect(self, images):
        results = self.det_net.predict(images, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                if self.last_detection is None:
                    print("未检测到人脸,且没有历史检测结果")
                    detections.append((np.array([]), []))
                else:
                    print("未检测到人脸,使用上一帧的检测结果")
                    detections.append(self.last_detection)
                continue

            # 转换为与原SCRFD输出格式一致的结果
            bboxes = np.column_stack((
                boxes.xyxy.cpu().numpy()[:, :2],  # x1,y1
                boxes.xyxy.cpu().numpy()[:, 2:] - boxes.xyxy.cpu().numpy()[:, :2]  # w,h
            ))
            
            indices = list(range(len(boxes)))
            current_detection = (bboxes, indices)
            # 更新最后一次检测结果
            self.last_detection = current_detection
            detections.append(current_detection)

        return detections

    def release(self):
        del self.det_net
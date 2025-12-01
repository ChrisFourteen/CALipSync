import os
import cv2
import torch
import numpy as np
from .tools.detect_face_yolo import YoloFaceDetector
#from .tools.detect_face import S3FDFaceDetector
from .tools.pfld_mobileone import PFLD_GhostOne as PFLDInference

class LipDetector:
   def __init__(self, weight_base_dir):
       self.mean_face_path = os.path.join(weight_base_dir, 'mean_face.txt')
       self.yolo_path = os.path.join(weight_base_dir, 'yolov8n-face.pt')
       self.s3fd_path = os.path.join(weight_base_dir, 'sfd_face.pth')
       self.checkpoint_path = os.path.join(weight_base_dir, 'checkpoint_epoch_335.pth.tar')
       
       with open(self.mean_face_path, 'r') as f:
           mean_face = f.read()
       self.mean_face = np.asarray(mean_face.split(' '), dtype=np.float32)
       
       print("正在加载模型...")
       #self.det_net = S3FDFaceDetector(self.s3fd_path)
       self.det_net = YoloFaceDetector(self.yolo_path)
       self.pfld_backbone = PFLDInference().cuda()
       checkpoint = torch.load(self.checkpoint_path)
       self.pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
       self.pfld_backbone.eval()
       print("模型加载完成！")

   def _face_det(self, images):
       """批量人脸检测和区域裁剪
       Args:
           images: 图片数组 [N,H,W,C]
       Returns:
           crops_list: 每张图片检测到的所有人脸crop
           offsets_list: 对应的偏移量
       """
       detections = self.det_net.detect(images)
       crops_list = []
       offsets_list = []
       
       for img, (bboxes, indices) in zip(images, detections):
           crops = []
           offsets = []
           height, width = img.shape[:2]
           
           for i in indices:
               x1, y1 = int(bboxes[i, 0]), int(bboxes[i, 1])
               w = int(bboxes[i, 2])
               h = int(bboxes[i, 3])
               x2, y2 = x1 + w, y1 + h
               
               cx, cy = (x2+x1)//2, (y2+y1)//2
               boxsize = int(max(w,h)*1.05)
               
               size = boxsize
               x1, y1 = cx - size//2, cy - size//2
               x2, y2 = x1 + size, y1 + size
               
               dx = max(0, -x1)
               dy = max(0, -y1)
               x1 = max(0, x1)
               y1 = max(0, y1)
               edx = max(0, x2 - width)
               edy = max(0, y2 - height)
               x2 = min(width, x2)
               y2 = min(height, y2)
               
               cropped = img[y1:y2, x1:x2]
               if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                   cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
                   y1 = y1-dy
                   x1 = x1-dx
               
               crops.append(cropped)
               offsets.append((x1, y1))
               
           crops_list.append(crops)
           offsets_list.append(offsets)
           
       return crops_list, offsets_list

   def detect_landmarks(self, images):
       """批量检测关键点
       Args:
           images: 图片数组
       Returns:
           results: 每张图片的关键点数组列表,无人脸则为None
       """
       crops_list, offsets_list = self._face_det(images)
       results = []
       
       for crops, offsets in zip(crops_list, offsets_list):
           if not crops:
               results.append(None)
               continue
               
           batch_landmarks = []
           for crop, (offset_x, offset_y) in zip(crops, offsets):
               h, w = crop.shape[:2]
               input_img = cv2.resize(crop, (192, 192))
               input_img = np.asarray(input_img, dtype=np.float32) / 255.0
               input_img = input_img.transpose(2, 0, 1)
               input_img = torch.from_numpy(input_img)[None].cuda()
               
               landmarks = self.pfld_backbone(input_img)
               pre_landmark = landmarks[0].cpu().detach().numpy()
               pre_landmark = pre_landmark + self.mean_face
               pre_landmark = pre_landmark.reshape(-1, 2)
               
               pre_landmark[:, 0] *= w
               pre_landmark[:, 1] *= h
               pre_landmark[:, 0] += offset_x
               pre_landmark[:, 1] += offset_y
               pre_landmark = pre_landmark.astype(np.int32)
               
               batch_landmarks.append(pre_landmark)
               
           results.append(batch_landmarks)
           
       return results

if __name__ == "__main__":
   weight_base_dir = r"I:\dh_video_unet\pretrained_models\lip_detect_weights"
   image_dir = r"I:\dh_video_unet\test\characters\test4\full_body_img"
   lm_output_dir = r"I:\dh_video_unet\temp\landmarks"
   face_output_dir = r"I:\dh_video_unet\temp\faces"
   
   detector = LipDetector(weight_base_dir)
   
   # 读取所有图片
   images = []
   image_names = []
   for i in range(1000):
       img_path = os.path.join(image_dir, f'{i}.jpg')
       if os.path.exists(img_path):
           img = cv2.imread(img_path)
           if img is not None:
               images.append(img)
               image_names.append(f'{i}')
   
   # 批量处理
   landmarks_list = detector.detect_landmarks(images)
   
   # 保存关键点和人脸图片
   os.makedirs(lm_output_dir, exist_ok=True)
   os.makedirs(face_output_dir, exist_ok=True)
   
   for img, name, landmarks in zip(images, image_names, landmarks_list):
       if landmarks is not None:
           # 保存关键点
           landmark_path = os.path.join(lm_output_dir, f'{name}.lms')
           np.savetxt(landmark_path, landmarks[0], fmt='%d')  # 只保存第一个检测到的人脸关键点
           
           # 保存人脸图片
           x_min = landmarks[0][:, 0].min()
           x_max = landmarks[0][:, 0].max()
           y_min = landmarks[0][:, 1].min()
           y_max = landmarks[0][:, 1].max()
           
           # 扩大人脸区域
           h, w = img.shape[:2]
           margin = int(min(x_max - x_min, y_max - y_min) * 0.3)
           x_min = max(0, x_min - margin)
           x_max = min(w, x_max + margin)
           y_min = max(0, y_min - margin)
           y_max = min(h, y_max + margin)
           
           face = img[y_min:y_max, x_min:x_max]
           face_path = os.path.join(face_output_dir, f'{name}.jpg')
           cv2.imwrite(face_path, face)
           
           print(f"处理完成: {name}")
   
   
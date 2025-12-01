import os
import cv2
import numpy as np

def crop_to_portrait(frame, target_ratio=9/16):
    """将图像裁切为指定比例的竖屏格式
    
    Args:
        frame: 输入图像
        target_ratio: 目标宽高比(默认9:16)
    
    Returns:
        裁切后的图像
    """
    height, width = frame.shape[:2]
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        # 输入图像太宽，需要在宽度上裁切
        new_width = int(height * target_ratio)
        margin = (width - new_width) // 2
        return frame[:, margin:margin + new_width]
    elif current_ratio < target_ratio:
        # 输入图像太高，需要在高度上裁切
        new_height = int(width / target_ratio)
        margin = (height - new_height) // 2
        return frame[margin:margin + new_height, :]
    else:
        return frame

def process_frame(frame, target_width=720, target_height=1280):
    """处理单帧图像
    
    Args:
        frame: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
    
    Returns:
        处理后的图像
    """
    height, width = frame.shape[:2]
    current_ratio = width / height
    target_ratio = target_width / target_height
    
    # 如果不是9:16，进行裁切
    if abs(current_ratio - target_ratio) > 0.01:  # 添加一些容差
        frame = crop_to_portrait(frame, target_ratio)
    
    # resize到目标尺寸
    return cv2.resize(frame, (target_width, target_height))

def extract_frames(video_path, frames_save_folder):
    """从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        frames_save_folder: 帧保存文件夹
        
    Returns:
        int: 总帧数
    """
    if not os.path.exists(frames_save_folder):
        os.makedirs(frames_save_folder)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame = process_frame(frame)
        
        # 保存处理后的帧
        np.save(os.path.join(frames_save_folder, f"{frame_idx}.npy"), processed_frame)
        
        # 每100帧打印一次进度
        if frame_idx % 100 == 0:
            print(f"已处理: {frame_idx}/{total_frames} 帧")
            
        frame_idx += 1
    
    print(f"完成! 共处理: {frame_idx} 帧")
    cap.release()
    return total_frames

if __name__ == "__main__":
    # 测试代码
    video_path = "test_video.mp4"
    frames_folder = "output_frames"
    total = extract_frames(video_path, frames_folder)
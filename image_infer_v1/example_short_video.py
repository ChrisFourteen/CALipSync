"""
短视频生成示例
使用VideoStreamManager的process_single_file方法生成短视频
"""

from bin.image_infer_v1.infer_api import VideoStreamManager

def generate_short_video_example():
    """
    生成短视频的完整示例
    """
    # 1. 配置参数
    data_dir = "path/to/your/preprocessed/data"  # 推理素材目录（预处理好的数据）
    unet_checkpoint = "path/to/unet/checkpoint.pth"  # UNet模型路径
    hubert_path = "path/to/hubert/model"  # Hubert模型路径
    
    # 音频和输出路径
    audio_path = "path/to/input/audio.wav"  # 输入音频文件
    output_path = "output/short_video.mp4"  # 输出视频路径
    
    # 2. 初始化视频流管理器
    manager = VideoStreamManager(
        data_dir=data_dir,
        unet_checkpoint=unet_checkpoint,
        hubert_path=hubert_path,
        window_width=1280,  # 窗口宽度（对于视频生成不重要）
        window_height=720,   # 窗口高度（对于视频生成不重要）
        buffer_threshold=50, # 缓冲阈值（对于视频生成不重要）
        device="cuda:0",     # 设备：cuda:0 或 cpu
        batch_size=8,        # 批处理大小，根据GPU内存调整
        synthesis_window_seconds=1.0,
        output_sample_rate=24000,
        image_output_mode="window",
        is_interrupt=False
    )
    
    # 3. 生成视频
    try:
        result_path = manager.process_single_file(
            audio_path=audio_path,
            output_path=output_path
        )
        print(f"短视频生成成功！输出路径: {result_path}")
        return result_path
    except Exception as e:
        print(f"生成视频时出错: {e}")
        raise
    finally:
        # 清理资源
        del manager


if __name__ == "__main__":
    # 运行示例
    generate_short_video_example()


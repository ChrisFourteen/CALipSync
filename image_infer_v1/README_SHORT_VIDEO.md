# 短视频生成功能说明

## 概述

本模块基于 `VideoStreamManager` 实现了短视频生成功能。输入音频文件和预处理好的推理素材，即可生成口型同步的短视频。

## 核心功能

### 1. `process_single_file` 方法

这个方法可以从音频文件生成完整的短视频：

```python
def process_single_file(self, audio_path: str, output_path: str) -> str:
    """
    处理单个音频文件并生成短视频
    
    Args:
        audio_path: 输入音频文件路径（.wav格式）
        output_path: 输出视频路径（.mp4格式）
    
    Returns:
        输出视频路径
    """
```

## 工作流程

1. **音频特征提取**
   - 使用 Hubert 模型提取音频特征
   - 特征用于驱动口型同步

2. **视频帧生成**
   - 根据音频特征生成视频帧
   - 支持口型同步（`is_generate_sync_frame=True`）

3. **视频保存**
   - 将生成的帧写入临时视频文件
   - 使用 ffmpeg 合并音频和视频
   - 清理临时文件

## 使用方法

### 基本用法

```python
from bin.image_infer_v1.infer_api import VideoStreamManager

# 初始化管理器
manager = VideoStreamManager(
    data_dir="path/to/preprocessed/data",
    unet_checkpoint="path/to/unet.pth",
    hubert_path="path/to/hubert/model",
    window_width=1280,
    window_height=720,
    buffer_threshold=50,
    device="cuda:0",
    batch_size=8
)

# 生成视频
output = manager.process_single_file(
    audio_path="input/audio.wav",
    output_path="output/video.mp4"
)
```

### 完整示例

参考 `example_short_video.py` 文件。

## 参数说明

### VideoStreamManager 初始化参数

- `data_dir`: 预处理好的推理素材目录（包含 frames/, positions/, masks/ 等子目录）
- `unet_checkpoint`: UNet 模型权重文件路径
- `hubert_path`: Hubert 模型路径
- `window_width`, `window_height`: 窗口尺寸（对于视频生成不重要）
- `buffer_threshold`: 缓冲阈值（对于视频生成不重要）
- `device`: 计算设备，`"cuda:0"` 或 `"cpu"`
- `batch_size`: 批处理大小，根据GPU内存调整（推荐4-16）
- `synthesis_window_seconds`: 合成窗口时长（默认1.0秒）
- `output_sample_rate`: 输出音频采样率（默认24000Hz）
- `is_interrupt`: 是否使用打断模式（对于视频生成不重要）

## 与实时播放的区别

- **实时播放**: 使用队列和播放器，适合实时流媒体场景
- **短视频生成**: 直接处理完整音频，生成视频文件，适合批量生成场景

## 依赖要求

- Python 3.8+
- PyTorch
- OpenCV (cv2)
- librosa
- soundfile
- ffmpeg（需要安装并配置到系统PATH）

## 注意事项

1. 确保 ffmpeg 已正确安装并可在命令行中调用
2. 输入音频应该是 WAV 格式
3. 确保预处理数据的帧数和音频长度匹配
4. 根据GPU内存调整 `batch_size` 参数
5. 生成的视频会保持原始预处理数据的尺寸和帧率（25fps）

## 性能优化建议

1. **批处理大小**: 根据GPU内存调整（RTX 3090 建议 8-16）
2. **设备选择**: 优先使用 CUDA 加速
3. **音频长度**: 较长的音频会需要更多时间，建议分段处理
4. **预处理质量**: 高质量的预处理数据会显著提升输出质量

## 故障排查

### 问题1: ffmpeg 找不到

```
解决方案: 确保 ffmpeg 已安装并在系统PATH中
```

### 问题2: CUDA 内存不足

```
解决方案: 减小 batch_size 参数
```

### 问题3: 生成的视频口型不同步

```
检查项:
1. 音频采样率是否正确（应该是16kHz用于特征提取）
2. 预处理数据的质量
3. UNet模型是否匹配
```

## 相关文件

- `infer_api.py`: 核心实现文件
- `example_short_video.py`: 使用示例
- `tools/frame_synthesizer/infer_api.py`: 帧合成器
- `utils/hubert_extractor.py`: 音频特征提取器


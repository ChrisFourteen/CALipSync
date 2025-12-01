# ATSYNC - 音频驱动数字人面部同步

## 快速开始

### 1. 数据处理
```bash
# 处理视频文件，提取音频特征和面部关键点
python data_process.py /path/to/video.mp4 --asr hubert
```

### 2. 模型训练
```bash
# 训练音频驱动的面部同步模型
python train.py --dataset_dir /path/to/processed/data --save_dir /path/to/save/models --epochs 200 --batchsize 1 --lr 0.001 --asr hubert
```

### 3. 模型推理
```bash
# 使用原始视频和音频文件进行推理（推荐）
python inference.py --input_video your_video.mp4 --input_audio your_audio.wav --model_path trained_model.pth --output_path result.mp4
```

## 完整流程示例

### 步骤1: 准备数据
```bash
# 处理示例视频
python data_process.py data/example1/Video_sample1.mp4 --asr hubert
```

### 步骤2: 训练模型
```bash
# 使用处理后的数据训练模型
python train.py --dataset_dir data/example1 --save_dir train_res --epochs 50 --batchsize 1 --lr 0.001 --asr hubert
```

### 步骤3: 推理生成
```bash
# 使用训练好的模型生成新视频
python inference.py \
    --input_video data/example1/Video_sample1.mp4 \
    --input_audio data/example1/aud.wav \
    --model_path train_res/best_digital_model_0.pth \
    --output_path result.mp4
```

## 主要参数说明

### 数据处理参数
- `--asr`: 音频特征提取方法 (`hubert` 或 `wenet`)

### 训练参数
- `--dataset_dir`: 预处理后的数据目录
- `--save_dir`: 模型保存目录
- `--epochs`: 训练轮数
- `--batchsize`: 批次大小
- `--lr`: 学习率
- `--asr`: 音频特征类型

### 推理参数
- `--input_video`: 输入MP4视频文件
- `--input_audio`: 输入WAV音频文件
- `--model_path`: 训练好的模型路径
- `--output_path`: 输出视频文件路径
- `--asr`: 音频特征提取方法
- `--batch_size`: 推理批次大小（默认32）

## 数据格式

处理后的数据目录结构：
```
data/
├── aud.wav                    # 提取的音频文件
├── aud_hu.npy                # HuBERT音频特征
├── full_body_img/            # 提取的视频帧
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── landmarks/                # 面部关键点
    ├── 0.lms
    ├── 1.lms
    └── ...
```

## 环境要求

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install transformers opencv-python librosa soundfile
pip install numpy scipy tqdm

# 安装FFmpeg（用于视频处理）
# Windows: 下载FFmpeg并添加到PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

## 常见问题

**Q: 训练时显存不足？**
A: 减小batch_size参数

**Q: 推理速度慢？**
A: 增加batch_size参数（在显存允许的情况下）

**Q: 支持哪些视频格式？**
A: 支持MP4格式，会自动转换为25fps

**Q: 支持哪些音频格式？**
A: 支持WAV、MP3等格式，会自动转换为16kHz采样率

## 注意事项

- 确保输入视频包含清晰的人脸
- 音频和视频长度建议匹配
- 训练需要GPU支持
- 推理过程会显示进度条


# 参数量统计   
 - OURS             参数量 19.794M   FLOPS：4.08G
 - Wav2lip          参数量 36.298M   FLOPS: 3.99G
 - MuseTalk         参数量 946.90M   FLOPS: 548.40G
 - LatentSync       参数量 1.36B     FLOPS：6.23T
 - videoRetalking   参数量 325.09M    FLOPS: 174.80G
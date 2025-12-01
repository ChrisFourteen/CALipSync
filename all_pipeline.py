import os
import requests
import traceback
import uuid
import shutil
# from step0_video_normalize import normalize_video
from step1_data_preprocess import VideoPreprocessor as TrainPreprocessor
from step2_train_unet import train_digital_model
from step3_prepare_infer_data import VideoPreprocessor as InferPreprocessor

def clean_intermediate_data(model_dir):
    """
    清理中间数据文件夹
    
    Args:
        model_dir: 模型目录
    """
    # 清理文件夹
    folders_to_clean = ['srt', 'faces', 'full_body_img', 'landmarks']
    for folder in folders_to_clean:
        folder_path = os.path.join(model_dir, folder)
        if os.path.exists(folder_path):
            print(f"清理文件夹: {folder_path}")
            shutil.rmtree(folder_path)
    
    # 清理文件
    files_to_clean = ['aud_hu.npy', 'aud.wav', 'sample.jpg']
    for file in files_to_clean:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"删除文件: {file_path}")
            os.remove(file_path)

def clone_video_local(
    video_path: str,
    output_dir: str,
    lip_detect_weight_base_dir: str,
    asr_weight_path: str,
    xseg_model_path: str,
    hubert_path: str, 
    vgg_path: str,
    unet_base_model: str,
    batch_size: int = 4,
    epochs: int = 5,
    model_name: str = None,
    use_base_model: bool = True
) -> tuple[str, str]:
    """
    使用本地视频进行训练与推理数据准备
    
    Args:
        video_path: 源视频路径
        output_dir: 输出目录
        lip_detect_weight_base_dir: 唇形检测权重目录
        asr_weight_path: ASR模型权重路径
        xseg_model_path: XSeg模型路径
        hubert_path: HuBERT模型路径
        vgg_path: VGG模型路径
        unet_base_model: UNet基础模型路径
        batch_size: 批处理大小
        epochs: 训练轮数
        model_name: 模型名称，如不指定则生成UUID
        use_base_model: 是否使用底模
        
    Returns:
        tuple[str, str]: (权重文件路径, 模型ID)
    """
    try:
        # 生成模型ID
        model_id = model_name if model_name else str(uuid.uuid4())
        model_dir = os.path.join(output_dir, model_id)
        weights_dir = os.path.join(model_dir, "weights")
        
        # 创建目录
        os.makedirs(weights_dir, exist_ok=True)
        
        # 1. 预处理
        # print("开始归一化视频...")
        # normalized_video = normalize_video(video_path)
        # print("归一化完成")
        
        print("开始预处理视频...")
        preprocessor = TrainPreprocessor(lip_detect_weight_base_dir, hubert_path)
        preprocessor.process_video(video_path, model_dir)
        print("预处理完成")
        
        # 2. 训练
        print("开始训练模型...")
        train_digital_model(
            dataset_dir=model_dir,
            save_dir=weights_dir,
            unet_checkpoint=unet_base_model if use_base_model else None,  # 根据是否使用底模决定
            vgg_path=vgg_path,
            batch_size=batch_size,
            epochs=epochs
        )
        print("训练完成")
        
        # 3. 准备推理数据
        print("准备推理数据...")
        infer_processor = InferPreprocessor(lip_detect_weight_base_dir, asr_weight_path, xseg_model_path)
        infer_processor.process_video(video_path, model_dir)
        print("推理数据准备完成")
        
        # 清理中间数据，节省空间
        print("开始清理中间数据...")
        clean_intermediate_data(model_dir)
        print("中间数据清理完成")
        
        final_weight_path = os.path.join(weights_dir, "model_final.pth")
        print(f"处理完成！模型权重已保存至: {final_weight_path}")
        print(f"模型ID: {model_id}")
        
        return final_weight_path, model_id
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        print("完整错误链路:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    
    video_path = "E:/unetsync/data/example5/0007.mp4"
    output_dir = "E:/unetsync/train_res"
    lip_detect_weight_base_dir = "E:/unetsync/pretrained_models/lip_detect_weights"
    asr_weight_path = "E:/unetsync/pretrained_models/asr"
    xseg_model_path = "E:/unetsync/pretrained_models/dfl_xseg.onnx"
    hubert_path = "E:/unetsync/pretrained_models/hubert_checkpoints"
    vgg_path = "E:/unetsync/pretrained_models/vgg/vgg19-dcbb9e9d.pth"
    unet_base_model = "E:/unetsync/pretrained_models/unet/checkpoint_e140.pth"
    batch_size = 4
    epochs = 20
    model_name = "example5"

    clone_video_local(video_path, output_dir, lip_detect_weight_base_dir, asr_weight_path, xseg_model_path, hubert_path, vgg_path, unet_base_model, batch_size, epochs, model_name)
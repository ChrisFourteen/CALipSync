import os
import torch
import soundfile as sf
from pydub import AudioSegment
from funasr import AutoModel

def process_and_split_audio(audio_path, local_model_dir, output_dir, min_seconds=5, max_seconds=12):
    """
    处理音频文件，识别内容，按随机目标长度（5-12秒）累积句子并保留原始间隔切分音频
    
    Args:
        audio_path: 原始音频文件路径
        local_model_dir: 本地模型目录
        output_dir: 输出目录，用于保存切分后的音频和文本
        min_seconds: 每段累积音频的最小长度（秒）
        max_seconds: 每段累积音频的最大长度（秒）
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型
        print(f"正在加载模型...")
        model = AutoModel(
            model=os.path.join(local_model_dir, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            model_revision="v2.0.4",
            vad_model=os.path.join(local_model_dir, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            vad_model_revision="v2.0.4", 
            punc_model=os.path.join(local_model_dir, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
            punc_model_revision="v2.0.4"
        )
        
        # 处理音频
        print(f"处理音频: {audio_path}")
        result = model.generate(input=audio_path, sentence_timestamp=True)
        
        if not result or len(result) == 0:
            print("没有识别到有效内容")
            return
            
        # 加载原始音频
        print("加载原始音频文件...")
        audio = AudioSegment.from_file(audio_path)
        
        # 获取句子级别时间戳信息
        sentences = result[0].get('sentence_info', [])
        
        print(f"识别到 {len(sentences)} 个句子，开始按目标长度累积切分...")
        
        import random
        
        # 累积音频片段直到达到目标长度
        segment_index = 0
        i = 0
        while i < len(sentences):
            # 随机选择本次目标长度（5-12秒）
            target_length_ms = random.randint(min_seconds * 1000, max_seconds * 1000)
            print(f"第 {segment_index} 段目标长度: {target_length_ms/1000:.1f}秒")
            
            # 初始化累积变量
            accumulated_text = ""
            start_sentence_index = i
            segment_start_ms = sentences[i]['start']
            segment_end_ms = segment_start_ms  # 初始结束时间等于开始时间
            
            # 累积句子直到达到目标长度
            while i < len(sentences):
                sentence = sentences[i]
                text = sentence['text']
                current_end_ms = sentence['end']
                
                # 计算当前累积片段长度
                current_duration_ms = current_end_ms - segment_start_ms
                
                # 如果已经达到或超过目标长度，且至少累积了一个句子，则停止累积
                if current_duration_ms >= target_length_ms and i > start_sentence_index:
                    break
                
                # 累积文本
                accumulated_text += text
                
                # 更新片段结束时间
                segment_end_ms = current_end_ms
                
                # 移动到下一个句子
                i += 1
            
            # 切分原始音频，保留原有间隔
            segment = audio[segment_start_ms:segment_end_ms]
            
            # 保存音频文件
            audio_output_path = os.path.join(output_dir, f"{segment_index}.wav")
            segment.export(audio_output_path, format="wav")
            
            # 保存文本文件
            text_output_path = os.path.join(output_dir, f"{segment_index}.txt")
            with open(text_output_path, "w", encoding="utf-8") as f:
                f.write(accumulated_text)
                
            actual_duration = segment_end_ms - segment_start_ms
            print(f"已保存片段 {segment_index}: {segment_start_ms}ms-{segment_end_ms}ms，时长 {actual_duration/1000:.2f}秒")
            print(f"片段文本: {accumulated_text}")
            
            # 下一个片段索引
            segment_index += 1
        
        print(f"处理完成，共生成 {segment_index} 个音频片段，已保存至: {output_dir}")
        
    finally:
        # 清理模型
        if 'model' in locals():
            if hasattr(model, 'model'):
                model.model = None
            if hasattr(model, 'vad_model'):
                model.vad_model = None
            if hasattr(model, 'punc_model'):
                model.punc_model = None
            model = None
            
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # 配置参数
    audio_path = r"I:\dh_live\static_sources\characters\images\model1_zhanzi1\model1_zhanzi1.mp4"
    local_model_dir = r"I:\dh_live_new\static_sources\pretrained_models\asr"
    output_dir = r"I:\dh_live_new\utils\get_srt\results"
    
    # 处理音频（可以自定义最小和最大秒数）
    process_and_split_audio(
        audio_path=audio_path, 
        local_model_dir=local_model_dir, 
        output_dir=output_dir,
        min_seconds=5,  # 最短累积5秒
        max_seconds=12  # 最长累积12秒
    )
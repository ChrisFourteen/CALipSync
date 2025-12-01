import torch
import soundfile as sf
import numpy as np
import os
from transformers import Wav2Vec2Processor, HubertModel

class HubertExtractor:
    def __init__(self, hubert_path, device="cuda:0"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_path, local_files_only=True)
        self.model = HubertModel.from_pretrained(hubert_path, local_files_only=True).to(device)
        
    def convert_to_16k(self, audio_path):
        output_path = audio_path.replace('.wav', '_16k.wav')
        os.system(f'ffmpeg -i {audio_path} -ar 16000 -ac 1 -y {output_path} -loglevel error')
        return output_path

    @torch.no_grad()
    def extract_features(self, speech):
        if speech.ndim == 2:
            speech = speech[:, 0]
            
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        num_iter = input_values.shape[1] // clip_length
        expected_T = (input_values.shape[1] - (kernel - stride)) // stride
        features = []
        
        for i in range(num_iter):
            start_idx = clip_length * i if i > 0 else 0
            end_idx = start_idx + (clip_length - stride + kernel) if i > 0 else clip_length - stride + kernel
            batch = input_values[:, start_idx:end_idx]
            hidden_states = self.model(batch).last_hidden_state
            features.append(hidden_states[0])
            
        if num_iter == 0 or input_values[:, clip_length * num_iter:].shape[1] >= kernel:
            remaining = input_values[:, clip_length * num_iter:]
            if remaining.shape[1] >= kernel:
                hidden_states = self.model(remaining).last_hidden_state
                features.append(hidden_states[0])
                
        features = torch.cat(features, dim=0).cpu()
        
        if features.shape[0] < expected_T:
            features = torch.nn.functional.pad(features, (0, 0, 0, expected_T - features.shape[0]))
        else:
            features = features[:expected_T]
            
        size = list(features.size())
        if size[0] % 2 == 1:
            size[0] -= 1
            features = features[:size[0]]
            
        return features.reshape(-1, 2, 1024)

    def extract_from_file(self, audio_path):
        audio_16k_path = self.convert_to_16k(audio_path)
        audio, _ = sf.read(audio_16k_path)
        features = self.extract_features(audio)
        # os.remove(audio_16k_path)
        return features.detach().numpy()
        
    def __del__(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    HUBERT_PATH = r"I:\unet_sync\pretrianed_models\hubert_checkpoints\models--facebook--hubert-large-ls960-ft\snapshots\ece5fabbf034c1073acae96d5401b25be96709d8"
    AUDIO_PATH = r"I:\dh_video\test\audios\short.wav"
    
    extractor = HubertExtractor(HUBERT_PATH)
    features = extractor.extract_from_file(AUDIO_PATH)
    print(f"Extracted features shape: {features.shape}")
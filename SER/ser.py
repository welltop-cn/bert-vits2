import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import os
import librosa
import numpy as np
import whisper
import logging


emotions_adv = {
    '悲伤': [0.203456, 0.215678, 0.134567],
    '冷漠': [0.102345, 0.309876, 0.204321],
    '放松': [0.150000, 0.608765, 0.706543], # 调整了唤醒值
    '忧郁': [0.401234, 0.302345, 0.201456],
    '冷静': [0.103456, 0.609876, 0.504321],
    '愉悦': [0.405678, 0.708901, 0.807654],
    '自信': [0.507890, 0.804321, 0.709876],
    '满足': [0.306543, 0.709876, 0.806789],
    '悲观': [0.302345, 0.207654, 0.109876],
    '烦恼': [0.607890, 0.203456, 0.102345],
    '淡定': [0.205432, 0.605432, 0.607890], # 调整了愉悦度值
    '焦躁': [0.709876, 0.306543, 0.201234],
    '惊恐': [0.807654, 0.102345, 0.103456],
    '紧张': [0.706543, 0.405678, 0.306543],
    '激动': [0.809876, 0.507890, 0.608765],
    '愤怒': [0.703210, 0.605432, 0.104321], # 调整了支配感值
    '热情': [0.609876, 0.608765, 0.809876],
    '暴躁': [0.708901, 0.207654, 0.205432],
    '兴奋': [0.807654, 0.509876, 0.708901], # 调整了支配感值
    '抱怨': [0.605678, 0.309876, 0.207654],
    '气喘吁吁': [0.801234, 0.304321, 0.208765],
    '哽咽': [0.502345, 0.304321, 0.207654],
    '嚎啕大哭': [0.901234, 0.104321, 0.105678],
    '平静': [0.302145, 0.706789, 0.804321],
    '开心': [0.621345, 0.812678, 0.932145],
    '震惊': [0.783912, 0.300000, 0.543210], # 调整了支配感值
}


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits
    
    
def find_top3_closest_emotions(input_vector, emotions_adv):
    # 计算所有情绪与输入向量的欧氏距离
    distances = [(emotion, np.linalg.norm(np.array(input_vector) - np.array(vector))) 
                 for emotion, vector in emotions_adv.items()]
    # 按照欧氏距离排序
    distances.sort(key=lambda x: x[1])
    # 选择距离最小的前三个情绪
    top3 = distances[:3]
    # 计算相似度（1除以（1加上距离））
    top3_with_similarity = [(emotion, 1 / (1 + distance)) for emotion, distance in top3]
    return top3_with_similarity

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


if __name__ == "__main__":
    
    # 设置基本配置。你可以自定义日志格式和日期格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # load model from hub
    device = 'cuda'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)
    whisper_v3 = whisper.load_model("large-v3")
    logging.info("Models loading succeed.")

    # dummy signal
    sampling_rate = 16000
    signal = np.zeros((1, sampling_rate), dtype=np.float32)

    # set filename
    filename = "/path/to/your/wav"
    srt = whisper_v3.transcribe(filename, language="English")['text']
    y, sr = librosa.load(filename, sr=16000)
    input_vector = process_func(y, sr)[0]
    logging.info(f"{filename} vec={input_vector} srt={srt}")
    top3_with_similarity = find_top3_closest_emotions(input_vector, emotions_adv)
    for emotion, score in top3_with_similarity:
        logging.info(f"{emotion}={score:.6f}")
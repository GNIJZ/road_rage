import numpy as np
import torch
from torch import nn
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from  dataset.clip.voice_data import get_audio
from util import mfcc_eval


class Audio_Tran(nn.Module):
    def __init__(self,audio_feature,hidden_size,num_heads,num_layers,dropout):
        super(Audio_Tran, self).__init__()
        # audio_data的形状为(batch_size,seq_len,feature)
        self.audio_feature = audio_feature
        self.hidden_size = hidden_size
        self.num_heads=num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.audio_feature, nhead=self.num_heads, dropout= self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.fc = nn.Linear(self.audio_feature, self.hidden_size)
        # Dropout 层用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        # 输入 x 的形状是 (batch_size, sequence_length, feature_dim)
        # batch_first 为false时，需要将其转置为 (sequence_length, batch_size, feature_dim) 以适应 Transformer 的输入，需要x = x.permute(1, 0, 2),输出需要x = x.permute(1, 0, 2)
        # batch_first 为true时,直接使用x，输出也不需要改
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x

if __name__ == '__main__':
    path, sr = 'E:/Python_Pro/road_rage/dataset/audio/20240719165745.wav', 16000
    # for i in range(1000):
    #
    #     data=get_audio(path,sr=16000,start_frame=i,end_frame=i+2)
    #     audio_mfcc=mfcc_eval(data,16000,100)
    #
    #     last_audio_mfcc = audio_mfcc
    #
    # # 打印最后一次的 MFCC 矩阵维度
    #     if last_audio_mfcc is not None:
    #         print(f"Last audio MFCC shape: {last_audio_mfcc.shape}")

    data=get_audio(path,sr=16000,start_frame=0,end_frame=2)
    audio_mfcc=mfcc_eval(data,16000,128)
    audio_mfcc = torch.unsqueeze(torch.tensor(audio_mfcc), dim=0)
    print(audio_mfcc.shape)

    tran_model=Audio_Tran(audio_feature=128,hidden_size=256,num_heads=8,num_layers=2,dropout=0.1)

    y=tran_model(audio_mfcc)
    print(y.shape)

import os

import pandas as pd
from PIL import Image

import torch.utils.data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,time,images,voices,images_transformer,voice_transformer):
        self.time = time
        self.images = images
        self.voices = voices
        self.images_transformer =images_transformer
        self.voice_transformer =voice_transformer
    def __getitem__(self,index):
        images=[]



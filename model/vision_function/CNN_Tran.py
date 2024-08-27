import numpy as  np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn


class CNN_Trans(nn.Module):
    def __init__(self,cnn_layer,trans_layer):
        super(CNN_Trans, self)
        self.cnn_layer=cnn_layer
        self.trans_layer=trans_layer
        

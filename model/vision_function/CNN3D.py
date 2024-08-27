import numpy as np
import cv2
import torch
from torch import nn



#VGG18
class CNN3D_VGG(nn.Module):
    def __init__(self,in_channel,time_steps,num_features,args):
        super(CNN3D_VGG, self).__init__()

        assert in_channel == 1, "Error: in_channel should  be 1."
        self.layers=self._make_layers(
            conv_out_channels=[64, 128, 256, 512],
            conv_kernel_size=3,
            conv_padding=1,
            pool_kernel_size=2,
            pool_stride=2,
            in_channels=in_channel
        )
        self.fc = nn.Linear(295936, num_features)

    def _make_layers(self, conv_out_channels, conv_kernel_size, conv_padding, pool_kernel_size, pool_stride,
                     in_channels):
        layers = []
        for x in conv_out_channels:
            layers.append(nn.Conv3d(in_channels, x, kernel_size=(conv_kernel_size, conv_kernel_size, conv_kernel_size),
                                    padding=(conv_padding, conv_padding, conv_padding)))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x
            # Adjust padding to ensure the depth dimension remains the same after pooling
            layers.append(nn.MaxPool3d(kernel_size=(pool_kernel_size, pool_kernel_size, pool_kernel_size),
                                       stride=pool_stride, padding=(1, conv_padding, conv_padding)))
        return nn.Sequential(*layers)
    def forward(self, x):
        x=self.layers(x)
        print(x.shape)
        batch_size, num_channels, depth, height, width = x.size()
        x = x.view(batch_size, -1)  # Flatten the tensor
        print(x.shape)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    in_channel = 1
    time_steps = 4
    num_features = 128
    args = None  # 这里没有使用 args，但可以用于其他配置

    # Instantiate the model
    model = CNN3D_VGG(in_channel, time_steps, num_features, args)

    # Create a sample input tensor
    example_input = torch.randn(8, in_channel, time_steps, 255, 255)  # (batch_size, channels, depth, height, width)

    # Run the model
    output = model(example_input)
    print("Output shape:", output.shape)                              # 应该输出 [8, time_steps, num_features]



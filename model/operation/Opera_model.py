import torch
from torch import nn


class Opera_model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Opera_model, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=50,
                          num_layers=2,
                          batch_first=True
                          )
        self.fc=nn.Linear(in_features=50,out_features=output_size)

    def forward(self,x):
        out, (hn, cn)=self.lstm(x)
        print(out.shape)
        x=self.fc(out[:,-1,:])
        print(x.shape)

        return x
x = torch.randn(32, 10, 3)
model = Opera_model(input_size=10, output_size=1)
output = model(x)

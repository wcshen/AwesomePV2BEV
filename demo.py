import torch
from torch import nn

class MS_CAM(nn.Module):
    
    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)

def main():
    input_tensor = torch.ones(size=(4, 16, 3, 3))
    test_layer = nn.Sequential(
        # nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(
            in_channels=16,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0
        )
    )
    out_tensor = test_layer(input_tensor)
    print(f"in: {input_tensor.shape}, out: {out_tensor.shape}")
    
    
    print(out_tensor[0,2])

if __name__ == '__main__':
    main()
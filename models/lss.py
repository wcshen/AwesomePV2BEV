import torch
from torch import nn

# camera feature: (N,C,H,W)
# camera frustum: (N,C,D,H,W)
# bev: (N,C,H',W')

# base LSS
# voxel_pooling LSS
# bevpool
# bevpoolv2

class LSSBase(nn.Module):
    def __init__(self, cfg=None):
        self.img_backbone = None
    
    def img_backbone(self):
        pass
    
    def predict_depth(self):
        pass
    
    def lift(self):
        pass
    
    def splat(self):
        pass
    
    def forward(self, input_dict):
        # 2d multi-view camera image to multi-view feature
        
        # 2d depth predict
        
        # 2d to 3d frustum
        
        # 3d to bev
        pass
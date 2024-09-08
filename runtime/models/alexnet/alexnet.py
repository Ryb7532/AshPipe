import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2

class alexnet(torch.nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()

    

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        out2 = self.stage2(out1)
        return out2

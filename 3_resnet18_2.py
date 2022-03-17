import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch,out_ch, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, *args, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__() 
        self.conv1= conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2= conv_bn_relu(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        if (stride == 2):
            self.shortcut = conv_bn_relu(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, x):  
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = out + shortcut
        return out 

class ResNet(nn.Module):
    def __init__(self, BasicBlock, num_classes = 1000):
        super(ResNet, self).__init__()
        self.in_ch = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 112x112x64, 1층
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) #56x56x64
        
        self.layer = nn.Sequential(
            self._make_layer(BasicBlock, out_ch=64, stride=1), 
            self._make_layer(BasicBlock, out_ch=64, stride=1),
            self._make_layer(BasicBlock, out_ch=128, stride=2),
            self._make_layer(BasicBlock, out_ch=128, stride=1),
            self._make_layer(BasicBlock, out_ch=256, stride=2),
            self._make_layer(BasicBlock, out_ch=256, stride=1),
            self._make_layer(BasicBlock, out_ch=512, stride=2),
            self._make_layer(BasicBlock, out_ch=512, stride=1)
        )
        
        self.linear = nn.Linear(512, num_classes)# FC, 18층 
            
        
    def _make_layer(self, BasicBlock, out_ch, stride):
        layer = BasicBlock(self.in_ch, out_ch, stride)
        self.in_ch = out_ch
        return layer
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.maxpool(out) 
        print(out.shape)
        out = self.layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        print("[layer FC]:",out.shape)
        return out 
        
        
def ResNet18():
    return ResNet(BasicBlock, num_classes=1000)

if __name__ == '__main__':
    model = ResNet18().cuda()
    input = torch.zeros(1, 3, 224, 224).cuda()
    
    pred = model(input)
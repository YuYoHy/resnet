import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_ch,out_ch, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, *args, **kwargs),
        nn.BatchNorm2d(out_ch),
    )

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__() 
        self.conv1= conv_bn(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2= conv_bn(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        
        if (stride == 2): #BasicBlock의 shortcut 사용 조건 
            self.shortcut = conv_bn(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, x):  
        out = self.conv1(x)
        out = F.relu(out)
        print("    conv1 :",out.shape)
        out = self.conv2(out)
        
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x       
        out = out + shortcut
        out = F.relu(out)
        print("    conv2 :",out.shape)
        return out 

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes = 1000):
        super(ResNet, self).__init__()
        self.in_ch = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 112x112x64, 1층
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) #56x56x64
        
        self.layer = nn.Sequential(
            self._make_layer(num_blocks[0], out_ch=64, stride=1), 
            self._make_layer(num_blocks[1], out_ch=128, stride=2),
            self._make_layer(num_blocks[2], out_ch=256, stride=2),
            self._make_layer(num_blocks[3], out_ch=512, stride=2),
        )
        
        self.linear = nn.Linear(512, num_classes)# FC
            
        
    def _make_layer(self, num_blocks, out_ch, stride):
        layers = []
        layers.append(BasicBlock(self.in_ch, out_ch, stride)) #레이어의 1번째 block (stride == 2)
#         print("[ 0 ]\n",layers[0])
        for i in range (num_blocks-1): #레이어의 나머지 블럭 생성 for문 (stride == 1) 
            self.in_ch = out_ch
            layers.append(BasicBlock(self.in_ch, out_ch, 1))            
#             print("change ch : ", self.in_ch) #채널 검증 
#             print("[",i+1,"]\n",layers[i+1]) #레이어 검증    
        
        return nn.Sequential(*layers)        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        print("7x7 conv:",out.shape)
        out = self.maxpool(out) 
        print("3x3 maxpool:",out.shape)
        out = self.layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        print("Flatten:",out.shape)
        out = self.linear(out)
        print("[layer FC]:",out.shape)
        return out        
        
def ResNet18():
    return ResNet(num_blocks = [2, 2, 2, 2], num_classes=1000)

def ResNet34():
    return ResNet(num_blocks = [3, 4, 6, 3], num_classes=1000)

if __name__ == '__main__':
    model = ResNet34().cuda()
    input = torch.zeros(1, 3, 224, 224).cuda()
    
    pred = model(input)
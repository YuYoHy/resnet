import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet18, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 112x112x64, 1층
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) #56x56x64
        
        self.conv_01 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 2층
        self.bn_01 = nn.BatchNorm2d(64)
        self.conv_02 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 3층
        self.bn_02 = nn.BatchNorm2d(64)       
        
        self.conv_03 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 4층
        self.bn_03 = nn.BatchNorm2d(64)
        self.conv_04 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 5층
        self.bn_04 = nn.BatchNorm2d(64)
        
        self.conv_11 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1, bias = False) #28x28x128, 6층
        self.bn_11 = nn.BatchNorm2d(128)
        self.conv_12 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 7층
        self.bn_12 = nn.BatchNorm2d(128)
                                      
        self.conv_short1_1 = nn.Conv2d(64, 128, kernel_size = 1, stride = (2, 2), bias = False) #stride==2 / ch : 64 -> 128
        self.bn_short1_1 = nn.BatchNorm2d(128)
                                      
        self.conv_13 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 8층
        self.bn_13 = nn.BatchNorm2d(128)
        self.conv_14 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 9층
        self.bn_14 = nn.BatchNorm2d(128)                                                                                                                                      
        self.conv_21 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1, bias = False) #14x14x256, 10층
        self.bn_21 = nn.BatchNorm2d(256)
        self.conv_22 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 11층
        self.bn_22 = nn.BatchNorm2d(256)
        
        self.conv_short2_1 = nn.Conv2d(128, 256, kernel_size = 1, stride = 2, bias = False) #stride==2 / ch : 128 -> 256
        self.bn_short2_1 = nn.BatchNorm2d(256)                             
                                      
        self.conv_23 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 12층
        self.bn_23 = nn.BatchNorm2d(256)
        self.conv_24 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 13층
        self.bn_24 = nn.BatchNorm2d(256) 
        
        self.conv_31 = nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1, bias = False) #7x7x512, 14층
        self.bn_31 = nn.BatchNorm2d(512)
        self.conv_32 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False) #7x7x512, 15층
        self.bn_32 = nn.BatchNorm2d(512)
                                      
        self.conv_short3_1 = nn.Conv2d(256, 512, kernel_size = 1, stride = 2, bias = False) #stride==2 / ch : 256 -> 512
        self.bn_short3_1 = nn.BatchNorm2d(512)
                                      
        self.conv_33 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False) #7x7x512, 16층
        self.bn_33 = nn.BatchNorm2d(512)
        self.conv_34 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False) #7x7x512, 17층
        self.bn_34 = nn.BatchNorm2d(512)
        
        self.linear = nn.Linear(512, num_classes)# FC, 18층 
        
    def forward(self, x):
        print("[input]:",x.shape)
        #224
        out = self.conv1(x)
        out = self.bn(out)
        print("[7x7 conv]:",out.shape)
        out = F.relu(out) 
        #112
        out = self.maxpool(out)
        print("[max pooled] :",out.shape)
        x = out

        #56
        out = self.conv_01(out) #64
        out = self.bn_01(out)
        out = F.relu(out)
        out = self.conv_02(out) #64
        out = self.bn_02(out)                                  
        out = F.relu(out)
        out = self.conv_03(out) #64
        out = self.bn_03(out)
        out = F.relu(out)
        out = self.conv_04(out) #64
        out = self.bn_04(out)                                
        out = F.relu(out)                              
        print("[layer 0]:",out.shape)
        
        #28
        out = self.conv_11(out) #128 / stride == 2
        out = self.bn_11(out)
        out = F.relu(out)
        out = self.conv_12(out) #128
        out = self.bn_12(out)
        shortcut = self.conv_short1_1(x) #채널 변경 구간 shortcut
        shortcut = self.bn_short1_1(shortcut)
        print("shortcut:",shortcut.shape)
        out = out + shortcut
        out = F.relu(out)
        out = self.conv_13(out) #128
        out = self.bn_13(out)
        out = F.relu(out)
        out = self.conv_14(out) #128
        out = self.bn_14(out)
        out = F.relu(out)                              
        print("[layer 1]:",out.shape)
        
        #14
        out = self.conv_21(out) #256 / stride == 2
        out = self.bn_21(out)
        out = F.relu(out)
        out = self.conv_22(out) #256
        out = self.bn_22(out)
        shortcut = self.conv_short2_1(shortcut) #채널 변경 구간 shortcut
        shortcut = self.bn_short2_1(shortcut)
        print("shortcut:",shortcut.shape)
        out = out + shortcut
        out = F.relu(out)
        out = self.conv_23(out) #256
        out = self.bn_23(out)
        out = F.relu(out)
        out = self.conv_24(out) #256
        out = self.bn_24(out)
        out = F.relu(out)                              
        print("[layer 2]:",out.shape)
        
        #7
        out = self.conv_31(out) #512 / stride == 2
        out = self.bn_31(out)
        out = F.relu(out)
        out = self.conv_32(out)
        out = self.bn_32(out)
        shortcut = self.conv_short3_1(shortcut) #채널 변경 구간 shortcut
        shortcut = self.bn_short3_1(shortcut)
        print("shortcut:",shortcut.shape)
        out = out + shortcut
        out = F.relu(out)
        out = self.conv_33(out) #512
        out = self.bn_33(out)
        out = F.relu(out)
        out = self.conv_34(out)
        out = self.bn_34(out)
        out = F.relu(out)                              
        print("[layer 3]:",out.shape)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        print("[layer FC]:",out.shape)
        return out
        
    
if __name__ == '__main__':
    
    model = ResNet18().cuda()
    input = torch.zeros(1, 3, 224, 224).cuda()
    
    pred = model(input)
    
        
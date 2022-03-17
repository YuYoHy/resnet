import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet50, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False) # 112x112x64, 1층
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) #56x56x64
        
        self.conv_001 = nn.Conv2d(64, 64, kernel_size = 1, stride = 1, bias = False) #56x56x64, 2층
        self.bn_001 = nn.BatchNorm2d(64)
        self.conv_002 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 3층
        self.bn_002 = nn.BatchNorm2d(64)
        self.conv_003 = nn.Conv2d(64, 256, kernel_size = 1, stride = 1,bias = False) #56x56x256, 4층
        self.bn_003 = nn.BatchNorm2d(256)
#         shortcut 위치 (입력 64, 출력 256)
        self.conv_short0 = nn.Conv2d(64, 256, kernel_size = 1, stride = 1, bias = False)
        self.bn_short0 = nn.BatchNorm2d(256)

        self.conv_011 = nn.Conv2d(256, 64, kernel_size = 1, stride = 1, bias = False) #56x56x256, 5층
        self.bn_011 = nn.BatchNorm2d(64)
        self.conv_012 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 6층
        self.bn_012 = nn.BatchNorm2d(64)
        self.conv_013 = nn.Conv2d(64, 256, kernel_size = 1, stride = 1,bias = False) #56x56x256, 7층
        self.bn_013 = nn.BatchNorm2d(256)
#         (입력 256, 출력 256)
        
        self.conv_021 = nn.Conv2d(256, 64, kernel_size = 1, stride = 1, bias = False) #56x56x256, 8층
        self.bn_021 = nn.BatchNorm2d(64)
        self.conv_022 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False) #56x56x64, 9층
        self.bn_022 = nn.BatchNorm2d(64)
        self.conv_023 = nn.Conv2d(64, 256, kernel_size = 1, stride = 1,bias = False) #56x56x256, 10층
        self.bn_023 = nn.BatchNorm2d(256)
#         (입력 256, 출력 256)
        
        self.conv_031 = nn.Conv2d(256, 128, kernel_size = 1, stride = 1, bias = False) #56x56x128, 11층
        self.bn_031 = nn.BatchNorm2d(128)
        self.conv_032 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1, bias = False) #28x28x128, 12층
        self.bn_032 = nn.BatchNorm2d(128)
        self.conv_033 = nn.Conv2d(128, 512, kernel_size = 1, stride = 1,bias = False) #28x28x512, 13층
        self.bn_033 = nn.BatchNorm2d(512)
#         shortcut 위치 (입력 256, 출력 512) (stride == 2)
        self.conv_short3 = nn.Conv2d(256, 512, kernel_size = 1, stride = 2, bias = False)
        self.bn_short3 = nn.BatchNorm2d(512)
        
        self.conv_041 = nn.Conv2d(512, 128, kernel_size = 1, stride = 1, bias = False) #28x28x128, 14층
        self.bn_041 = nn.BatchNorm2d(128)
        self.conv_042 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 15층
        self.bn_042 = nn.BatchNorm2d(128)
        self.conv_043 = nn.Conv2d(128, 512, kernel_size = 1, stride = 1,bias = False) #28x28x512, 16층
        self.bn_043 = nn.BatchNorm2d(512)
#         (입력 512, 출력 512)
                                                                           
        self.conv_051 = nn.Conv2d(512, 128, kernel_size = 1, stride = 1, bias = False) #28x28x128, 17층
        self.bn_051 = nn.BatchNorm2d(128)
        self.conv_052 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 18층
        self.bn_052 = nn.BatchNorm2d(128)
        self.conv_053 = nn.Conv2d(128, 512, kernel_size = 1, stride = 1,bias = False) #28x28x512, 19층
        self.bn_053 = nn.BatchNorm2d(512)    
#         (입력 512, 출력 512)
        
        self.conv_061 = nn.Conv2d(512, 128, kernel_size = 1, stride = 1, bias = False) #28x28x128, 20층
        self.bn_061 = nn.BatchNorm2d(128)
        self.conv_062 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False) #28x28x128, 21층
        self.bn_062 = nn.BatchNorm2d(128)
        self.conv_063 = nn.Conv2d(128, 512, kernel_size = 1, stride = 1,bias = False) #28x28x512, 22층
        self.bn_063 = nn.BatchNorm2d(512) 
#         (입력 512, 출력 512)
                                                                         
        self.conv_071 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, bias = False) #28x28x256, 23층
        self.bn_071 = nn.BatchNorm2d(256)
        self.conv_072 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1, bias = False) #14x14x256, 24층
        self.bn_072 = nn.BatchNorm2d(256)
        self.conv_073 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 25층
        self.bn_073 = nn.BatchNorm2d(1024) 
#         shortcut 위치 (입력 512, 출력 1024) (stride == 2)
        self.conv_short7 = nn.Conv2d(512, 1024, kernel_size = 1, stride = 2, bias = False)
        self.bn_short7 = nn.BatchNorm2d(1024)
        
        self.conv_081 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, bias = False) #14x14x256, 26층
        self.bn_081 = nn.BatchNorm2d(256)
        self.conv_082 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 27층
        self.bn_082 = nn.BatchNorm2d(256)
        self.conv_083 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 28층
        self.bn_083 = nn.BatchNorm2d(1024)
#         (입력 1024, 출력 1024)

        self.conv_091 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, bias = False) #14x14x256, 29층
        self.bn_091 = nn.BatchNorm2d(256)
        self.conv_092 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 30층
        self.bn_092 = nn.BatchNorm2d(256)
        self.conv_093 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 31층
        self.bn_093 = nn.BatchNorm2d(1024)
#         (입력 1024, 출력 1024)
     
        self.conv_101 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, bias = False) #14x14x256, 32층
        self.bn_101 = nn.BatchNorm2d(256)
        self.conv_102 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 33층
        self.bn_102 = nn.BatchNorm2d(256)
        self.conv_103 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 34층
        self.bn_103 = nn.BatchNorm2d(1024)
#         (입력 1024, 출력 1024)

        self.conv_111 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, bias = False) #14x14x256, 35층
        self.bn_111 = nn.BatchNorm2d(256)
        self.conv_112 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 36층
        self.bn_112 = nn.BatchNorm2d(256)
        self.conv_113 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 37층
        self.bn_113 = nn.BatchNorm2d(1024)
#         (입력 1024, 출력 1024)
     
        self.conv_121 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, bias = False) #14x14x256, 38층
        self.bn_121 = nn.BatchNorm2d(256)
        self.conv_122 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False) #14x14x256, 39층
        self.bn_122 = nn.BatchNorm2d(256)
        self.conv_123 = nn.Conv2d(256, 1024, kernel_size = 1, stride = 1,bias = False) #14x14x1024, 40층
        self.bn_123 = nn.BatchNorm2d(1024)
#         (입력 1024, 출력 1024)

        self.conv_131 = nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, bias = False) #14x14x512, 41층
        self.bn_131 = nn.BatchNorm2d(512)
        self.conv_132 = nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1, bias = False) #7x7x512, 42층
        self.bn_132 = nn.BatchNorm2d(512)
        self.conv_133 = nn.Conv2d(512, 2048, kernel_size = 1, stride = 1,bias = False) #7x7x2048, 43층
        self.bn_133 = nn.BatchNorm2d(2048)
#         shortcut 위치 (입력 1024, 출력 2048) (stride == 2)
        self.conv_short13 = nn.Conv2d(1024, 2048, kernel_size = 1, stride = 2, bias = False)
        self.bn_short13 = nn.BatchNorm2d(2048)

        self.conv_141 = nn.Conv2d(2048, 512, kernel_size = 1, stride = 1, bias = False) #7x7x512, 44층
        self.bn_141 = nn.BatchNorm2d(512)
        self.conv_142 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False) #7x7x512, 45층
        self.bn_142 = nn.BatchNorm2d(512)
        self.conv_143 = nn.Conv2d(512, 2048, kernel_size = 1, stride = 1,bias = False) #7x7x2048, 46층
        self.bn_143 = nn.BatchNorm2d(2048)
#         (입력 2048, 출력 2048)

        self.conv_151 = nn.Conv2d(2048, 512, kernel_size = 1, stride = 1, bias = False) #7x7x512, 47층
        self.bn_151 = nn.BatchNorm2d(512)
        self.conv_152 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False) #7x7x512, 48층
        self.bn_152 = nn.BatchNorm2d(512)
        self.conv_153 = nn.Conv2d(512, 2048, kernel_size = 1, stride = 1,bias = False) #7x7x2048, 49층
        self.bn_153 = nn.BatchNorm2d(2048)
#         (입력 2048, 출력 2048)
       
        self.linear = nn.Linear(2048, num_classes)# FC, 50층 
        
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

        #layer0 - 1
        out = self.conv_001(out) #1
        out = self.bn_001(out)
        out = F.relu(out)
        out = self.conv_002(out) #2
        out = self.bn_002(out)                                  
        out = F.relu(out)
        out = self.conv_003(out) #3 
        out = self.bn_003(out)
        shortcut =self.conv_short0(x)
        shortcut = self.bn_short0(shortcut)
        out = out + shortcut
        out = F.relu(out)
        #layer0 - 2
        out = self.conv_011(out) #1
        out = self.bn_011(out)
        out = F.relu(out)
        out = self.conv_012(out) #2
        out = self.bn_012(out)                                  
        out = F.relu(out)
        out = self.conv_013(out) #3 
        out = self.bn_013(out)
        #layer0 - 3
        out = self.conv_021(out) #1
        out = self.bn_021(out)
        out = F.relu(out)
        out = self.conv_022(out) #2
        out = self.bn_022(out)                                  
        out = F.relu(out)
        out = self.conv_023(out) #3 
        out = self.bn_023(out)
                               
        print("[layer 0]:",out.shape)
        
        #layer1 - 1
        out = self.conv_031(out) #1
        out = self.bn_031(out)
        out = F.relu(out)
        out = self.conv_032(out) #2
        out = self.bn_032(out)                                  
        out = F.relu(out)
        out = self.conv_033(out) #3 
        out = self.bn_033(out)
        shortcut =self.conv_short3(shortcut)
        shortcut = self.bn_short3(shortcut)
        out = out + shortcut
        out = F.relu(out)
        #layer1 - 2
        out = self.conv_041(out) #1
        out = self.bn_041(out)
        out = F.relu(out)
        out = self.conv_042(out) #2
        out = self.bn_042(out)                                  
        out = F.relu(out)
        out = self.conv_043(out) #3 
        out = self.bn_043(out)
        #layer1 - 3
        out = self.conv_051(out) #1
        out = self.bn_051(out)
        out = F.relu(out)
        out = self.conv_052(out) #2
        out = self.bn_052(out)                                  
        out = F.relu(out)
        out = self.conv_053(out) #3 
        out = self.bn_053(out)
        #layer1 - 4
        out = self.conv_061(out) #1
        out = self.bn_061(out)
        out = F.relu(out)
        out = self.conv_062(out) #2
        out = self.bn_062(out)                                  
        out = F.relu(out)
        out = self.conv_063(out) #3 
        out = self.bn_063(out)
        
        print("[layer 1]:",out.shape)
        
        #layer2 - 1
        out = self.conv_071(out) #1
        out = self.bn_071(out)
        out = F.relu(out)
        out = self.conv_072(out) #2
        out = self.bn_072(out)                                  
        out = F.relu(out)
        out = self.conv_073(out) #3 
        out = self.bn_073(out)
        shortcut =self.conv_short7(shortcut)
        shortcut = self.bn_short7(shortcut)
        out = out + shortcut
        out = F.relu(out)
        #layer2 - 2
        out = self.conv_081(out) #1
        out = self.bn_081(out)
        out = F.relu(out)
        out = self.conv_082(out) #2
        out = self.bn_082(out)                                  
        out = F.relu(out)
        out = self.conv_083(out) #3 
        out = self.bn_083(out)
        #layer2 - 3
        out = self.conv_091(out) #1
        out = self.bn_091(out)
        out = F.relu(out)
        out = self.conv_092(out) #2
        out = self.bn_092(out)                                  
        out = F.relu(out)
        out = self.conv_093(out) #3 
        out = self.bn_093(out)
        #layer2 - 4
        out = self.conv_101(out) #1
        out = self.bn_101(out)
        out = F.relu(out)
        out = self.conv_102(out) #2
        out = self.bn_102(out)                                  
        out = F.relu(out)
        out = self.conv_103(out) #3 
        out = self.bn_103(out)
        #layer2 - 5
        out = self.conv_111(out) #1
        out = self.bn_111(out)
        out = F.relu(out)
        out = self.conv_112(out) #2
        out = self.bn_112(out)                                  
        out = F.relu(out)
        out = self.conv_113(out) #3 
        out = self.bn_113(out)
        #layer2 - 6
        out = self.conv_121(out) #1
        out = self.bn_121(out)
        out = F.relu(out)
        out = self.conv_122(out) #2
        out = self.bn_122(out)                                  
        out = F.relu(out)
        out = self.conv_123(out) #3 
        out = self.bn_123(out)
        
        print("[layer 2]:",out.shape)
        
        #layer3 - 1
        out = self.conv_131(out) #1
        out = self.bn_131(out)
        out = F.relu(out)
        out = self.conv_132(out) #2
        out = self.bn_132(out)                                  
        out = F.relu(out)
        out = self.conv_133(out) #3 
        out = self.bn_133(out)
        shortcut =self.conv_short13(shortcut)
        shortcut = self.bn_short13(shortcut)
        out = out + shortcut
        out = F.relu(out)
        #layer3 - 2
        out = self.conv_141(out) #1
        out = self.bn_141(out)
        out = F.relu(out)
        out = self.conv_142(out) #2
        out = self.bn_142(out)                                  
        out = F.relu(out)
        out = self.conv_143(out) #3 
        out = self.bn_143(out)
        #layer3 - 3
        out = self.conv_151(out) #1
        out = self.bn_151(out)
        out = F.relu(out)
        out = self.conv_152(out) #2
        out = self.bn_152(out)                                  
        out = F.relu(out)
        out = self.conv_153(out) #3 
        out = self.bn_153(out)
        
        print("[layer 3]:",out.shape)
   
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        print("[layer FC]:",out.shape)
        return out
        
    
if __name__ == '__main__':
    
    model = ResNet50().cuda()
    input = torch.zeros(1, 3, 224, 224).cuda()
    
    pred = model(input)
    
        
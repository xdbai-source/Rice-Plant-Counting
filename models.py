import torch
from torch import nn
from torch.utils import model_zoo


class Model(nn.Module):
    def __init__(self, gap = False):
        super(Model, self).__init__()
        self.gap = gap                
        self.vgg = VGG()              
        self.load_vgg()               
        if self.gap:
            self.de = Detect()        
        else:
            self.amp = BackEnd()      
            self.dmp = BackEnd()

            self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
            self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        input = self.vgg(input)
        if self.gap:
            dis = self.de(*input)         
            return dis
        else:
            amp_out = self.amp(*input)    
            dmp_out = self.dmp(*input)

            amp_out = self.conv_att(amp_out)
            dmp_out = amp_out * dmp_out
            dmp_out = self.conv_out(dmp_out)

            return dmp_out, amp_out       

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):     
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = BaseConv(1024, 512, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv4 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv7 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)

        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv4(input)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv7(input)
        input = self.conv8(input)
        input = self.conv9(input)

        return input


class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()
        self.ac = nn.ReLU()
        
        self.conv1 = BaseConv(512, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(128, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(32, 1, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.adappool = nn.AdaptiveMaxPool2d((50,80))
        self.line1 = nn.Sequential(nn.Linear(4000,100), nn.Dropout(p=0.01), nn.ReLU())
        
        self.line2 = nn.Linear(100,1)
        self._initialize_weights()
    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        x = self.conv1(conv5_3)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adappool(x)
        x = x.view(x.size(0), -1)
        x = self.line1(x)
     
        
        x = self.line2(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)






























    
    
    
    
    
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)    
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400).cuda()
    model = Model().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())

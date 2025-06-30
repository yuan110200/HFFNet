import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
#from mobilenetv2 import mobilenet_v2
#from resnet import Backbone_ResNet34_in3
from mobileone import mobileone, reparameterize_model
from thop import profile
from mobilenetv2 import mobilenet_v2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out = self.conv1(max_out)
        return self.sigmoid(max_out) * x

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class DLDHblock(nn.Module):
    def __init__(self,channel = 128):
        super(DLDHblock, self).__init__()
        self.channel = channel
        self.stage1 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=1)
        self.stage2 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=2)
        self.stage3 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=3)
        self.stage4 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=4)
        self.filter1 = convbnrelu(channel // 2, channel // 4, k=1, s=1, p=0, relu=True)
        self.filter2 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)

    def forward(self, r, t):
        rs = torch.split(r, self.channel // 4, dim=1)
        ts = torch.split(t, self.channel // 4, dim=1)
        
        r1 = self.stage1(rs[0])
        r2 = self.stage2(r1 + rs[1])
        r3 = self.stage3(r2 + rs[2])
        r4 = self.stage4(r3 + rs[3])

        t1 = self.stage1(ts[0])
        t2 = self.stage2(t1 + ts[1])
        t3 = self.stage3(t2 + ts[2])
        t4 = self.stage4(t3 + ts[3])

        rt1 = self.filter1(torch.cat([r1, t1], dim = 1))
        rt2 = self.filter1(torch.cat([r2, t2], dim = 1))
        rt3 = self.filter1(torch.cat([r3, t3], dim = 1))
        rt4 = self.filter1(torch.cat([r4, t4], dim = 1))

        rt = self.filter2(torch.cat([rt4, rt3, rt2, rt1], dim = 1))

        fl = self.fuse(rt + r)

        return fl

class DLDblock(nn.Module):
    def __init__(self,channel = 128):
        super(DLDblock, self).__init__()
        self.channel = channel
        self.stage1 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=1)
        self.stage2 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=2)
        self.stage3 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=3)
        self.stage4 = DSConv3x3(channel // 4, channel // 4, stride=1, dilation=5)
        self.filter1 = convbnrelu(channel // 2, channel // 4, k=1, s=1, p=0, relu=True)
        self.filter2 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)

    def forward(self, r, t):
        rs = torch.split(r, self.channel // 4, dim=1)
        ts = torch.split(t, self.channel // 4, dim=1)
        
        r1 = self.stage1(rs[0])
        r2 = self.stage2(r1 + rs[1])
        r3 = self.stage3(r2 + rs[2])
        r4 = self.stage4(r3 + rs[3])

        t1 = self.stage1(ts[0])
        t2 = self.stage2(t1 + ts[1])
        t3 = self.stage3(t2 + ts[2])
        t4 = self.stage4(t3 + ts[3])

        rt1 = self.filter1(torch.cat([r1, t1], dim = 1))
        rt2 = self.filter1(torch.cat([r2, t2], dim = 1))
        rt3 = self.filter1(torch.cat([r3, t3], dim = 1))
        rt4 = self.filter1(torch.cat([r4, t4], dim = 1))

        rt = self.filter2(torch.cat([rt4, rt3, rt2, rt1], dim = 1))

        fl = self.fuse(rt + r)

        return fl

class prediction_decoder(nn.Module):
    def __init__(self, channel1=16, channel2=24, channel3=24, n_classes=9):
        super(prediction_decoder, self).__init__()
        self.decoder3 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                DSConv3x3(channel3, channel3, stride=1, dilation=3),
                DSConv3x3(channel3, channel2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        self.decoder2 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                DSConv3x3(channel2, channel2, stride=1, dilation=3),
                DSConv3x3(channel2, channel1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        # 240 320 -> 480 640
        self.decoder1 = nn.Sequential(
                nn.Dropout2d(p=0.1),
                DSConv3x3(channel1, channel1, stride=1, dilation=3),
                DSConv3x3(channel1, channel1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 480 640
                DSConv3x3(channel1, channel1),
                nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
                )

    def forward(self, x3, x2, x1):
        x3_decoder = self.decoder3(x3)
        x2_decoder = self.decoder2(x3_decoder + x2)
        semantic_pred = self.decoder1(x2_decoder + x1)

        return semantic_pred

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[32, 96, 320],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*3, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        #self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3 = inputs
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c3.shape

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        #x = self.linear_pred(x)

        return x


class light_rgbt(nn.Module):
    def __init__(self, n_classes):
        super(light_rgbt, self).__init__()   
        self.rgb_pretrained = mobilenet_v2()
        self.thermal_pretrained = mobilenet_v2()

        self.dld5 = DLDHblock(320)
        self.dld4 = DLDblock(96)
        self.dld3 = DLDblock(32)

        self.decoder1 = DecoderHead(in_channels=[32, 96, 320], num_classes=n_classes, norm_layer=nn.BatchNorm2d, embed_dim=160)
        self.decoder2 = DecoderHead(in_channels=[24, 32, 96], num_classes=n_classes, norm_layer=nn.BatchNorm2d, embed_dim=96)
        self.decoder = prediction_decoder(16, 96, 160, n_classes)


    def forward(self, rgb, depth):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        r1, r2, r3, r4, r5 = self.rgb_pretrained(x)
        t1, t2, t3, t4, t5 = self.thermal_pretrained(ir)    

        f5 = self.dld5(r5, t5)
        f4 = self.dld4(r4, t4)
        f3 = self.dld3(r3, t3)

        enc1_feats = [f3, f4, f5]
        enc2_feats = [r2, f3, f4]

        dec_out1 = self.decoder1(enc1_feats)
        dec_out2 = self.decoder2(enc2_feats)
        
        semantic = self.decoder(dec_out1, dec_out2, r1)

        return semantic

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    light_rgbt(9)
    image = torch.randn(1, 3, 480, 640)
    thermal = torch.randn(1, 3, 480, 640)
    model = light_rgbt(9)
    Flops, params = profile(model, inputs=(image, thermal)) 
    print('Flops: % .4fG'%(Flops / 1000000000))   
    print('params: % .4fM'% (params / 1000000))
    x1 = model(image, thermal)
    print(x1.shape)
    
    # for PST900 dataset
    # LASNet(5)

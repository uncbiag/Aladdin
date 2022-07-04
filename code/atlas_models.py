import torch.nn as nn
import torch
import torch.nn.functional as F
from atlas_utils import *


dim = 3
Conv = nn.Conv2d if dim == 2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
conv = F.conv2d if dim == 2 else F.conv3d


class conv_bn_rel(nn.Module):
    """
    conv + bn (optional) + relu

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, group=1, dilation=1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1, dilation=1)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1, dilation=1)

        self.bn = BatchNorm(out_channels) if bn else None #, eps=0.0001, momentum=0, affine=True
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class FcRel(nn.Module):
    """
    fc+ relu(option)
    """
    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x



class SVF_resid(nn.Module):
    def __init__(self, img_sz, args, bn=False):
        super(SVF_resid, self).__init__()
        self.int_steps = 7
        self.img_sz = img_sz
        self.scale = 1.0 / (2 ** self.int_steps)
        self.id_transform = gen_identity_map(self.img_sz, 1.0).cuda(args.gpu)
        self.bilinear = Bilinear(zero_boundary=True)
        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False, group=2)
        self.down_path_2_1 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False, group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False, group=2)
        self.down_path_2_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_1 = conv_bn_rel(32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(64, 128, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_2 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_1 = conv_bn_rel(128, 256, 3, stride=2, active_unit='relu', same_padding=True, bn=bn)
        self.down_path_16_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit='relu', same_padding=True, bn=bn)

        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(256, 128, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn, reverse=True)
        self.up_path_8_2 = conv_bn_rel(128 + 128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_1 = conv_bn_rel(128, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn, reverse=True)
        self.up_path_4_2 = conv_bn_rel(64 + 64, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=bn, reverse=True)

        self.up_path_2_2 = conv_bn_rel(32 + 32, 16, 3, stride=1, active_unit='None', same_padding=True)
        self.up_path_2_3 = conv_bn_rel(16, 16, 3, stride=1, active_unit='None', same_padding=True)
        self.up_path_1_1 = conv_bn_rel(16, 16, 2, stride=2, active_unit='None', same_padding=False, bn=bn, reverse=True)
        self.up_path_1_2 = conv_bn_rel(16, 3, 3, stride=1, active_unit='None', same_padding=True)

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2 + d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2

        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3, u8_1), 1))
        u8_3 = self.up_path_8_3(u8_2)
        u8_3 = u8_2 + u8_3
        u4_1 = self.up_path_4_1(u8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3, u4_1), 1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        output = self.up_path_2_3(u2_2)

        flow = self.up_path_1_2(self.up_path_1_1(output))

        pos_flow = flow * self.scale
        neg_flow = -flow * self.scale
        for _ in range(self.int_steps):
            pos_deform_field = pos_flow + self.id_transform
            neg_deform_field = neg_flow + self.id_transform
            pos_flow_1 = self.bilinear(pos_flow, pos_deform_field)
            neg_flow_1 = self.bilinear(neg_flow, neg_deform_field)
            pos_flow = pos_flow_1 + pos_flow
            neg_flow = neg_flow_1 + neg_flow

        return pos_flow, neg_flow


#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
import numpy as np
import math
# import cv2
import os
# from torchvision.transforms import CenterCrop, RandomCrop, RandomHorizontalFlip, Compose
user_path = os.path.expanduser('~')
EPS = 1e-16


class GrayCropFlip(nn.Module):
    def __init__(self, channel_input='bgr', size=None, random=False, skip_gray=False):
        super(GrayCropFlip, self).__init__()
        self.skip_gray = skip_gray
        if not self.skip_gray:
            self.channel2idx = {channel_input[i]: i for i in range(len(channel_input))}

    def forward(self, x, length=None):
        # x: [B, max_frames, 96, 96, 3]
        if not self.skip_gray:
            assert x.shape[-1] == 3, 'shape error: input must have r,g,b 3 channels, but got {}'.format(x.shape)
            x_split = x.split(1, dim=-1)
            # Gray = R*0.299 + G*0.587 + B*0.114
            gray_frames = 0.114 * x_split[self.channel2idx['b']] + 0.587 * x_split[
                self.channel2idx['g']] + 0.299 * x_split[self.channel2idx['r']]
            x = gray_frames.sum(dim=-1)
        # [B, max_frames, 88, 88]
        if hasattr(self, 'random'):
            print("1")
            x = self.train_transform(x) if self.training and self.random else self.eval_transform(x)
        return x, length


class VideoFrontend(nn.Module):
    def __init__(self, output_dim=256):
        super(VideoFrontend, self).__init__()
        self.video_frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=output_dim)

    def forward(self, x):
        B, T, _, _ = x.size()
        x = x.unsqueeze(1)
        x = self.video_frontend(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(B, T, -1)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x

class LSTM_Encoder(nn.Module):
    def __init__(self,feature_dim,hidden_size,num_layers):
        super(LSTM_Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack_rnn = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, batch_first=False, bidirectional=False, num_layers=1)

    def forward(self, cur_inputs, current_frame):
        packed_input = nn.utils.rnn.pack_padded_sequence(cur_inputs, current_frame)
        rnn_out, _ = self.stack_rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out) 
               
        return rnn_out

class CNN_Resnet(nn.Module):
    def __init__(self, output_dim=256):
        super(CNN_Resnet, self).__init__()
        self.video_frontend = VideoFrontend(output_dim)

    def forward(self, video_inputs):
        video_inputs = video_inputs.permute(1, 0, 2, 3)
        lip_inputs = self.video_frontend(video_inputs) #(B, T, 256)
        lip_inputs = lip_inputs.permute(1, 0, 2) # (T, B, 256)
        
        return lip_inputs

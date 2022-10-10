from __future__ import absolute_import

import math
import numpy as np

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.initializer import Normal, Constant, Assign
zeros_ = Constant(value=0)
ones_ = Constant(value=1)


def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
        conv_layer,
        nn.BatchNorm2D(out_planes),
        nn.ReLU(),
    )
    return block


class STNHead(nn.Layer):
    def __init__(self, in_planes, num_ctrlpoints, activation='none'):
        super(STNHead, self).__init__()

        self.in_planes = in_planes
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_planes, 32),  # 32*64
            nn.MaxPool2D(kernel_size=2, stride=2),
            conv3x3_block(32, 64),  # 16*32
            nn.MaxPool2D(kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2D(kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2D(kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2D(kernel_size=(1, 2), stride=(1, 2)),
            conv3x3_block(256, 256))  # 1*2

        self.stn_fc1 = nn.Sequential(
            nn.Linear(2 * 256, 512),
            nn.BatchNorm1D(512),
            nn.ReLU())
        self.stn_fc2 = nn.Linear(512, num_ctrlpoints * 2)

        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)

    def init_weights(self, module):
        for m in module.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(0, math.sqrt(2. / n))
                normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                normal_ = Normal(0, 0.001)
                normal_(m.weight)
                zeros_(m.bias)

    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        # print(ctrl_points.shape)
        if self.activation is 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        elif self.activation == 'relu':
            ctrl_points = F.relu(paddle.to_tensor(ctrl_points))
        zeros_(stn_fc2.weight)
        Assign(value=ctrl_points.reshape(-1))(stn_fc2.bias)

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.shape
        x = x.reshape([batch_size, -1])
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = paddle.nn.Sigmoid(x)
        if self.activation == 'relu':
            x = F.relu(x)
        x = x.reshape([-1, self.num_ctrlpoints, 2])
        return img_feat, x


if __name__ == "__main__":
    in_planes = 3
    num_ctrlpoints = 20
    activation = 'none'  # 'sigmoid'
    stn_head = STNHead(in_planes, num_ctrlpoints, activation)
    input = paddle.randn([10, 3, 32, 64])
    control_points = stn_head(input)
    print(control_points.shape)

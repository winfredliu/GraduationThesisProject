import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    # [B, C, H, W] -> [B, C, H, W]
    # BasicBlock -> Attention -> BasicBlock -> Attention- >  BottlrBlock

    def __init__(self, 
                 inplanes, 
                 planes, 
                 kernel_size=1, 
                 stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.conv_q_right = nn.Conv2d(
            self.inplanes,
            1,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)
        self.conv_v_right = nn.Conv2d(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)
        self.conv_up = nn.Conv2d(
            self.inter_planes,
            self.planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_q_left = nn.Conv2d(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)
        self.softmax_left = nn.Softmax(dim=2)


    def channel_attention(self, x):
        # x [B, C, H, w]
        input_x = self.conv_v_right(x) # [B, C1, H, W]
        batch, _, height, width = input_x.shape
        input_x = input_x.reshape(batch, self.inter_planes, height * width) # [B, C1, HxW]
        context_mask = self.conv_q_right(x) # [B, 1, H, W]
        context_mask = context_mask.reshape(batch, 1, height * width) # [B, 1, HxW]
        context_mask = self.softmax_right(context_mask) # [B, 1, HxW]
        context = torch.matmul(input_x, context_mask.permute(0, 2, 1)) # [B, C1, HxW]@[B, HxW, 1] = [B, C1, 1]
        context = context.unsqueeze(-1) # [B, C1, 1, 1]
        context = self.conv_up(context) # [B, C, 1, 1]
        mask_ch = self.sigmoid(context) # [B, C, 1, 1]
        out = x * mask_ch # [B, C, H, w]*[B, C, 1, 1] -> [B, C, H, w]*[B, C, H, W] 
        # e.g. [[3]]*[[2, 3], [1, 4]] -> [[3, 3], [3, 3]]*[[2, 3], [1, 4]] = [[6, 9], [3, 12]]
        return out

    def pixel_attention(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.shape
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.shape
        avg_x = avg_x.reshape(batch, channel, avg_x_h * avg_x_w)
        avg_x = avg_x.reshape(batch, avg_x_h * avg_x_w, channel)
        theta_x = self.conv_v_left(x).reshape(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.reshape(batch, 1, height, width) # x [B, C, H, W] x[:,:,i,j]
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        context_channel = self.channel_attention(x)
        context_spatial = self.pixel_attention(x)
        # torch.cat([context_channel, context_spatial], dim=1)
        out = context_spatial + context_channel
        return out
    
if __name__ == '__main__':
    x = torch.randn(1, 256, 64, 64)
    attention = Attention(256, 256)
    print(attention(x).shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.attention import Attention




class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.attention = Attention(256, 256)
        self.attention2 = Attention(256, 256)
        self.freeze_bn = freeze_bn
        
        pretrained = 'run/massroad/deeplab-resnet/miou_71.85.pth'
        pretrained_dict = torch.load(pretrained)['state_dict']
        # print(pretrained_dict.keys())
        self.load_state_dict(pretrained_dict, strict=False)

    def single_forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.attention2(x)
        low_level_feat = self.attention(low_level_feat)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=False)
        return x
    
    def forward(self, input):
        if self.training:
            scales = [1.0]
        else:
            scales = [0.75, 1.0, 1.25]
        H, W = input.shape[-2], input.shape[-1]
        final_logit = 0
        for scale in scales:
            h, w = int(H * scale + 0.5), int(H * scale + 0.5)
            input = F.interpolate(input, [h, w], mode='bilinear')
            logit = self.single_forward(input)
            logit = F.interpolate(logit, [H, W], mode='bilinear')
            
            final_logit += logit
        
        return final_logit


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())



#
# demo.py
#
import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import utils.connect as cn
from utils.metrics import Evaluator
from dataloaders import make_data_loader

def validation(model, evaluator, val_loader):
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label'] / 255
        image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()

        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        for n in range(pred.shape[0]):
            pred[n] = cn.cut_road_connection(np.uint8(pred[n]))
        evaluator.add_batch(target, pred)

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # parser.add_argument('--in-path', type=str, required=True, help='image to test')
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','massroad'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    # model = nn.DataParallel(model).cuda()
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    evaluator = Evaluator(nclass)
    validation(model,evaluator,val_loader)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))



if __name__ == "__main__":
   main()

# python demo.py --in-path your_file --out-path your_dst_file

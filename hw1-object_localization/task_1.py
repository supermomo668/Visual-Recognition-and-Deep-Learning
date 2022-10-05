import argparse
import os
import shutil
import time

import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchmetrics import JaccardIndex
from sklearn.preprocessing import minmax_scale
import wandb

import matplotlib.pyplot as plt
from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *
#
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp


USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=2,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=2,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=1e-2,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0
visual_interval = 15

def main():
    global args, best_prec1, dataset
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    #print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    #criterion = nn.MultiLabelSoftMarginLoss().cuda()   #
    # use binary cross entropy to evaluate difference between output and groundtruth labels (one-hot)
    criterion = nn.BCELoss().cuda()
    # optimizer with SGD with given parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                             weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    dataset = VOCDataset('trainval', top_n=10, image_size=512, data_dir='./data/VOCdevkit/VOC2007/')
    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    # fix seed
    torch.manual_seed(101)
    n = len(dataset)
    # split dataset 80/20 for train/validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(np.floor(n*0.8)), n-int(np.floor(n*0.8))])
    train_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))
    # loader for the relative dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=custom_collate_fn_VOC,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=custom_collate_fn_VOC,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    if USE_WANDB:
        wandb.init(project="vlr-hw1", reinit=True)
    # Ideally, use flags since wandb makes it harder to debug code.
    # GradCAM
    cam_extractor = SmoothGradCAMpp(model)
    for epoch in range(args.start_epoch, args.epochs):    
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, cam_extractor, wandb.Table(columns=["image", "heatmap", "GradCAM"]))

        # evaluate on validation set
        if epoch % args.eval_freq == 0:
            m1, m2 = validate(val_loader, model, criterion, epoch, cam_extractor, wandb.Table(columns=["image", "heatmap", "GradCAM"]))
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    wandb.finish()

# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, cam_extractor=None, vis_table=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    avg_m3 = AverageMeter()

    # switch to train mode
    model.train()
    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))
    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        input_im = data['image'].to('cuda')
        target = data['label']
                            
        # TODO (Q1.1): Get output from model
        if i==0: print("Forward pass")
        cls_out = model(input_im)
        # TODO (Q1.1): Perform any necessary functions on the output
        # Pooling + sigmoid is moved to within AlexNet.py
        # imoutput = nn.MaxPool2d(kernel_size=(conv_out.size(2), conv_out.size(3)))(conv_out)
        # imoutput = torch.sigmoid(imoutput.squeeze())
        
        if i==0: print(f"Output size:{model.feat_map.size()}")
        # upsample to match input size
        vis_heatmap = F.interpolate(model.feat_map, size=(input_im.shape[2],input_im.shape[3]), mode='nearest')
        if i==0: print(f"Heatmap output size:{vis_heatmap.shape}")
        
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(cls_out.cpu(), target)
        
        # measure metrics and record loss
        m1 = metric1(cls_out.cpu(), target)
        m2 = metric2(cls_out.cpu(), target)
        
        losses.update(loss.item(), len(data))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2)
                 )
        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
        c_map = plt.get_cmap('jet')
        if epoch <= 1 and i==0:
            for n, (im, out_heatmap) in enumerate(zip(input_im, vis_heatmap)):
                input_img = wandb.Image(im, boxes={
                    "predictions": {
                        "box_data": get_box_data(data['gt_classes'][n], data['gt_boxes'][n]),
                        "class_labels": class_id_to_label,       
                    },
                })
                # Log feature map from model.classifier for visualizing heampa
                att_map = vis_heatmap[n][data['gt_classes'][n][0]].cpu().detach().numpy()
                att_map = minmax_scale(np.abs(att_map).ravel(), feature_range=(0,1)).reshape(att_map.shape)
                
                # Retrieve the CAM by passing the class index and the model output
                if cam_extractor:
                    activation_map = cam_extractor(cls_out[n].squeeze(0).argmax().item(), cls_out[n])
                    gradcam_result = overlay_mask(to_pil_image(im), to_pil_image(activation_map[0][n], mode='F'), alpha=0.5)
                else:
                    gradcam_result = att_map
                #
                vis_table.add_data(input_img, wandb.Image(c_map(att_map)), wandb.Image(gradcam_result))
                if n==1: 
                    break
        wandb.log(
            {'train/loss':loss, 'train/metric1': m1,  'train/metric2': m2,}
        )
        # End of train()

def validate(val_loader, model, criterion, epoch=0, cam_extractor=None, vis_table=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))
    
    end = time.time()
    for i, (data) in enumerate(val_loader):
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        input_im = data['image'].to('cuda')
        target_class = data['label']
                            
        # TODO (Q1.1): Get output from model
        if i==0: print("Forward pass")
        cls_out = model(input_im)
        # TODO (Q1.1): Perform any necessary functions on the output
        # upsample to match input size
        vis_heatmap = F.interpolate(model.feat_map, size=(input_im.shape[2],input_im.shape[3]), mode='nearest')
        if i==0: print(f"Heatmap output size:{vis_heatmap.shape}")
        
        # TODO (Q1.1): Compute loss using ``criterion``
        # (same as above)
        loss = criterion(cls_out.to('cpu'), target_class)
        
        # measure metrics and record loss
        losses.update(loss.item(), len(data))
        m1 = metric1(cls_out.to('cpu'), target_class)
        m2 = metric2(cls_out.to('cpu'), target_class)
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize things as mentioned in handout
        # jet colormap
        c_map = plt.get_cmap('jet')

        # TODO (Q1.3): Visualize at appropriate intervals
        if (epoch==0 or epoch==1 ) and i==0:
            for n, (im, out_heatmap) in enumerate(zip(input_im, vis_heatmap)):
                input_img = wandb.Image(im, boxes={
                    "predictions": {
                        "box_data": get_box_data(data['gt_classes'][n], data['gt_boxes'][n]),
                        "class_labels": class_id_to_label,       
                    },
                })
                # Log feature map from model.classifier for visualizing heampa
                att_map = vis_heatmap[n][data['gt_classes'][n][0]].cpu().detach().numpy()
                att_map = minmax_scale(np.abs(att_map).ravel(), feature_range=(0,1)).reshape(att_map.shape)
                # Retrieve the CAM by passing the class index and the model output
                if cam_extractor:
                    activation_map = cam_extractor(cls_out[n].squeeze(0).argmax().item(), cls_out[n])
                    gradcam_result = overlay_mask(to_pil_image(im), to_pil_image(activation_map[0][n], mode='F'), alpha=0.5)
                else:
                    gradcam_result = att_amp
                #
                vis_table.add_data(input_img, wandb.Image(c_map(att_map)),wandb.Image(gradcam_result))
                if n==1: 
                    break
        wandb.log(
            {'val/loss':loss, 'val/metric1': m1,  'val/metric2': m2}
        )

        
    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric1(output, target):
    # TODO (Q1.5): compute metric1
    target = target.detach().numpy().astype('int')
    output = output.detach().numpy().astype('float')
    # get features that aren't all zero
    feat_considered = ~np.all(np.concatenate([target, output]), axis=0)
    mean_ap = sklearn.metrics.average_precision_score(target[:,feat_considered], output[:,feat_considered], average='samples')
    return mean_ap   #[0]
        
def metric2(output, target, thres=0.5):
    # TODO (Q1.5): compute metric2
    target = np.int32(target.detach().numpy())
    output = (output.detach().numpy()>thres).astype(int)
    feat_considered = ~np.all(np.concatenate([target, output]), axis=0)
    # threshold the sigmoid output and evaluate on recall where classes are present in the class
    recall = sklearn.metrics.recall_score(target[:, feat_considered], output[:, feat_considered], average='samples')
    #print(f"Metric 2:{recall}")
    return recall  #[0]

if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    description='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    description='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    description='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    description='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float,
    description='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    description='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int,
    description='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    description='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    description='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def metric1(output, target):
    # TODO (Q1.5): compute metric1
    target = target.detach().numpy().astype('int')
    output = output.detach().numpy().astype('float')
    # get features that aren't all zero
    feat_considered = ~np.all(np.concatenate([target, output]), axis=0)
    mean_ap = sklearn.metrics.average_precision_score(target[:,feat_considered], output[:,feat_considered], average='samples')
    return mean_ap    #[0]

def calculate_map(roi_bboxes, box_scores,  gt_classes, gt_bboxes, n_classes=20):
    """
    Calculate the mAP for classification.
    # roi_bboxes: (N=300,4)
    # box_scores: (N=300, 20) Output from model
    # gt_classes: (n_detected,)
    # gt_bboxes: (n_detected, 4)
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Using IOU to iterate each box for each class
    # Compare each iteratively and take Maximum IOU
    pred_boxes, gt_boxes = np.asarray(roi_bboxes), np.asarray(gt_bboxes)
    box_scores = np.asarray(box_scores)
    per_class_iou = np.empty(n_classes)   # ( 20)
    filtered_bboxes = []
    cls_labels = []
    
    for class_num in range(20):   # (per class)
        gt_class_idx = np.where(np.asarray(gt_classes)==class_num)[0]
        pred_class_idx = np.where(np.argmax(np.asarray(roi_bboxes), axis=1)==class_num)[0]
        if len(pred_class_idx)==0 or len(gt_class_idx)==0:
            continue
        # [ n_selected, 4];   # [n_selected, 4]
        pred_class_bbox, gt_class_bbox = pred_boxes[pred_class_idx], gt_boxes[gt_class_idx]
        bboxes, conf_scores = nms(pred_class_bbox, box_scores[pred_class_idx])
        # nms_idx = nms(torch.as_tensor(pred_class_bbox,dtype=torch.float32),
        #               torch.as_tensor(box_scores[pred_class_idx, class_num],dtype=torch.float32), 0.2)
        #bboxes, conf_scores = pred_class_bbox[nms_idx], box_scores[pred_class_idx, class_num][nms_idx]
        [filtered_bboxes.append(b) for b in  bboxes]
        cls_labels += [class_num]*len(bboxes)
        # iterate each bbox to gt bbox
        class_iou = []
        for i, pred_box in enumerate(pred_class_bbox):
            class_iou.append(np.max([iou(pred_box, gt_box) for gt_box in gt_class_bbox]))            
        per_class_iou[class_num] = np.array(class_iou).mean()
    if len(filtered_bboxes)>0:
        filtered_bboxes= np.stack(filtered_bboxes)
    #print(f"Number of bbox:{len(filtered_bboxes)},{filtered_bboxes}")
    return per_class_iou, (filtered_bboxes, cls_labels)


def test_model(model, val_loader=None, epoch=0,thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    test_loss = 0
    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))
    class_ap_table = wandb.Table(columns=dataset.CLASS_NAMES)
    vis_table = wandb.Table(columns=["image", "prediction", "mean AP (over all class)"])
    batch_class_ap = []
    #
    with torch.no_grad():
        batch_ap_by_class = []
        for i, data in enumerate(val_loader):
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class = data['gt_classes']
            if np.any([len(r)!= 300 for r in rois]):
                continue
            img_info = dict.fromkeys(np.arange(args.batch_size), {'pred_boxes':[],'pred_cls':[]})
            # TODO (Q2.3): perform forward pass, compute cls_probs
            box_prob = model(image, rois, target)   # (N, 300, 20)
            loss = model.build_loss(box_prob, target)
            test_loss += loss.item()
            
            # TODO (Q2.3): Iterate over each class (follow comments)
            #for n in range(args.batch_size):
            # Get NMS bboxes
            ap_by_class, (pred_boxes, labels) = calculate_map(rois[0], box_prob.detach().cpu(),  gt_class[0], gt_boxes[0])
            batch_class_ap.append(ap_by_class)
            running_mean_ap = np.nanmean(np.stack(batch_class_ap), axis=0)
            class_ap_table.add_data(*running_mean_ap)
            # TODO (Q2.3): visualize bounding box predictions when required
            if len(pred_boxes)>0:
                print(f"ap by class:{ap_by_class}")
                print(f"Pred boxes:{pred_boxes},{labels}")
                print(f"wandb:{get_box_data(labels, pred_boxes)}")

            if i%args.disp_interval==0:
                for n, im in enumerate(image):
                    gtbbox_img = wandb.Image(im, boxes={
                        "predictions": {
                        "box_data": get_box_data(data['gt_classes'][n], data['gt_boxes'][n]),
                        "class_labels": class_id_to_label,       
                        },
                    })
                    predbbox_img = wandb.Image(im, boxes={
                        "predictions": {
                        "box_data": get_box_data(labels, pred_boxes),
                        "class_labels": class_id_to_label,       
                        },
                    })
                    vis_table.add_data(gtbbox_img, predbbox_img, np.nanmean(running_mean_ap)) 
        wandb.log({f"val/{epoch}/Visuals": vis_table})
        wandb.log({f"val/{epoch}/Class mAP": class_ap_table})
    return np.nanmean(np.stack(batch_class_ap), axis=0)
            


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))
    class_ap_table = wandb.Table(columns=[str(c) for c in np.arange(len(dataset.CLASS_NAMES))])
    vis_table = wandb.Table(columns=["image", "class AP scores"])
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            debug=True if (epoch==0 and i==0) else False
            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class = data['gt_classes']
            if np.any([len(r)!= 300 for r in rois]):
                continue
            print(np.any([len(r)!= 300 for r in rois]), [len(r) for r in rois])
            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            if debug: print(f"Input shapes: Image {image.size()}; ROIs:{[len(r) for r in rois]}; targets:{target.size()}")
            cls_prob = model(image, rois, target)   # (N, 300, 20)
            if debug: print(f"Output shape: {cls_prob.size()}")
            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure metrics and record loss
            m1 = metric1(torch.sum(cls_prob,dim=1).cpu(), target.cpu())
            wandb.log(
                {'train/loss':loss, 'train/metric1': m1} #,  'train/metric2': m2,}
            )
            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if i % args.val_interval == 0 and iter != 0:
                print("Evaluating Model.")
                model.eval()
                ap = test_model(model, val_loader)
                model.train()
            
            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            if i % args.disp_interval==0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {train_loss:.4f}\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      train_loss=train_loss,
                      # ap=ap,
                  )
                )
                #logger.model_param_zhisto_summary(model=net, step=step)
                pass
                #
            if i%args.disp_interval:
                for n, im in enumerate(image):
                    input_img = wandb.Image(im, boxes={
                        "predictions": {
                            "box_data": get_box_data(data['gt_classes'][n], data['gt_boxes'][n]),
                            "class_labels": class_id_to_label,       
                        },
                    })
                    # predbbox_img = wandb.Image(im, boxes={
                    #     "predictions": {
                    #     "box_data": get_box_data(img_info[n]['pred_boxes'], img_info[n]['pred_boxes']),
                    #     "class_labels": class_id_to_label,       
                    #     },
                    # })
                    vis_table.add_data(input_img, ap)
                wandb.log({"train/Visuals": vis_table})
                # wandb.log({"AP by Class": class_ap_table})
    
    # TODO (Q2.4): Plot class-wise APs
    


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    global args
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    dataset = VOCDataset('trainval', top_n=10, image_size=512, data_dir='../data/VOCdevkit/VOC2007/')
    n = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(np.floor(n*0.8)), n-int(np.floor(n*0.8))])
    train_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))

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
    
    # Initialize wandb logger
    if args.use_wandb:
        wandb.init(project="vlr-hw1", reinit=True)
        
    # Create network and initialize
    net = WSDDN(classes=dataset.CLASS_NAMES)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in net.features.parameters():
        param.requires_grad = False
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    params = list(net.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)

    # Training
    print("Start train.")
    train_model(net, train_loader, val_loader, optimizer, args)

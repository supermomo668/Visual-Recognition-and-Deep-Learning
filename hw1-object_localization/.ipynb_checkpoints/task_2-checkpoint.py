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


def calculate_map():
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    pass


def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    with torch.no_grad():
        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            # TODO (Q2.3): perform forward pass, compute cls_probs
            
            
            # TODO (Q2.3): Iterate over each class (follow comments)
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                
                # use NMS to get boxes and scores
                pass

            # TODO (Q2.3): visualize bounding box predictions when required
            calculate_map()


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            
            # forward
            model(image, rois, gt_vec)
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Log to screen
            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration
                log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
                    step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
                log_print(log_text, color='green', attrs=['bold'])
                re_cnt = True

            #TODO: evaluate the model every N iterations (N defined in handout)
            if step % eval_interval == 0 and step>0:
                aps = test_net(name='test_weights', net = net, imdb = test_imdb, logger=logger, step=step, visualize=True, thresh = 0.0001)
                if firstEval:
                    vis.line(X = np.array([step]), Y = np.array([np.mean(aps)]), win="test/mAP", opts=dict(title='Test mAP'))
                    firstEval = 0
                else:
                    vis.line(X = np.array([step]), Y = np.array([np.mean(aps)]), win="test/mAP", update="append", opts=dict(title='Test mAP'))
                for i in range(len(aps)):
                    logger.scalar_summary(tag='test/AP/'+imdb._classes[i], value=aps[i], step=step)
            
    
            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if i % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            

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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(np.floor(n*0.8)), n-int(np.floor(n*0.8))])
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
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in model.parameters():
        param.requires_grad = False
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = None

    # Training
    train_model(net, train_loader, val_loader, optimizer, args)

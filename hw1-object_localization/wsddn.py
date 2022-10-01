import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align
#
from torch.autograd import Variable

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=20):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
         self.features  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #nn.ReLU()
            )

        self.roi_pool = RoIPool(pooled_height=6, pooled_width=6, spatial_scale=1.0/16)

        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True)
        )
        
        self.score_fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=20)
        )
        self.bbox_fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=20)
        )
        # loss
        self.criterion = nn.BCELoss(size_average=True).cuda()# None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        # TODO (Q2.1): Use image and rois as input
            # turn to channel first
        im_data = Variable(torch.from_numpy(im_data).type(torch.FloatTensor).permute(0, 3, 1, 2)
        rois = Variable(torch.from_numpy(rois).type(torch.FloatTensor)
        #TODO: Use im_data and rois as input
        # compute cls_prob which are N_roi X 20 scores
        # Checkout faster_rcnn.py for inspiration

        rpns = self.roi_pool(self.features(im_data), rois)
        rpns = rpns.view(rpns.size(0), 256 * 6 * 6)
        class_out = self.classifier(rpns)
        cls_score = F.softmax(self.score_cls(class_out), dim=1)   
        det_score = F.softmax(self.score_det(class_out), dim=0)
        # cls_score = F.softmax(cls_score, dim=-1)
        # det_score = F.softmax(det_score, dim=-2)
        
        
        # compute cls_prob which are N_roi X 20 scores
        cls_prob = cls_score * det_score

        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        loss = F.binary_cross_entropy(torch.clamp(torch.sum(cls_prob,dim=0), 0, 1) ,
                                      label_vec, size_average=False)
        return loss

class FC(nn.Module):
    def __init__(self, in_features, out_features, activate=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activate = activate
        if activate:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.activate:
            x = self.relu(x)
        return x
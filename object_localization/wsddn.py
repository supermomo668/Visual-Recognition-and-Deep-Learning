import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align, RoIPool
#
from torch.autograd import Variable

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=20, pretrained=True):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(f"Classes:{classes}")

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

        self.roi_pool = RoIPool((6, 6), spatial_scale=31.0)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True)
        )
        
        self.score_out = nn.Sequential(
            nn.Linear(in_features=4096, out_features=20)
        )
        self.bbox_out = nn.Sequential(
            nn.Linear(in_features=4096, out_features=20)
        )
        # loss
        self.criterion = nn.BCELoss(reduction='sum').cuda() # None
        #self.criterion = nn.BCEWithLogitsLoss(reduction='sum').cuda() # None
        # load weight
        if pretrained:
            load_weights = model_zoo.load_url(model_urls['alexnet'])
            for item_name in self.features.state_dict().keys():
                self.features.state_dict()[item_name] = load_weights['features.'+item_name]
        # Set require grad and initialize
        # for m in [self.classifier, self.score_out, self.bbox_out]:
        #     for layer in m:
        #         layer.requires_grad = True
        #         if hasattr(layer,'weight'):
        #             nn.init.xavier_uniform(layer.weight)

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        # TODO (Q2.1): Use image and rois as input
        im_data = image.type(torch.FloatTensor).cuda()
        rois = [roi.type(torch.FloatTensor).cuda() for roi in rois]
        #TODO: Use im_data and rois as input
        rois_pooled = self.roi_pool(self.features(im_data), rois) 
        # -> Tensor[K, C, output_size[0], output_size[1]]   # (N, 256, 6, 6)
        self.pooled_shape = rois_pooled.size()
        rois_pooled = rois_pooled.view(len(rois_pooled),-1)
        # (b=300, 9216)
        rois_feat = self.classifier(rois_pooled)   # (300, 4096)
        assert torch.sum(torch.isnan(rois_feat))==0, f"ROI feat problem: {rois_pooled.size()}"
        # score/bbox: out (300, 20)
        class_score = F.softmax(self.score_out(rois_feat), dim=0)     
        # (300 =300*1, 20)
        detect_score = F.softmax(self.bbox_out(rois_feat), dim=1)     # (300 =300*1, 20)
        # compute cls_prob which are N_roi X 20 scores
        try:
            box_prob = class_score * detect_score   # (N, 300 =300*1, 20)
        except:
            print(f"Debug size: rois:{[len(r) for r in rois]}:{pooled_shape} \nrois_feat:{rois_feat.size()}\nbox_prob:{box_prob.size()}")
        if self.training:
            self.cross_entropy = self.build_loss(box_prob, gt_vec.cuda())
        return box_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss
        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        cls_prob = torch.sum(cls_prob, dim=0).unsqueeze(0)    #(1, 20)
        cls_prob = torch.clamp(cls_prob, 0.0, 1.0)
        return self.criterion(cls_prob, label_vec.cuda())

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
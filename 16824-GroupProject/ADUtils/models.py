from numpy.lib.function_base import select

import torch, torchvision, torchmetrics
import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.callbacks import  GradientAccumulationScheduler
from torch.optim import lr_scheduler
from torchsummary import summary
from torch import nn

import segmentation_models_pytorch as smp
from torchvision.models import EfficientNet_B7_Weights, ResNeXt101_32X8D_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights


class FPNclassifier(smp.FPN):
    def __init__(self,backbone = 'timm-res2net50_14w_8s', n_classes=2, encoder_weights=None, **kwargs):
        # https://github.com/qubvel/segmentation_models.pytorch#encoders 
        assert backbone in ['timm-res2net50_14w_8s','efficientnet-b7', 'timm-mobilenetv3_large_minimal_100',
                            'timm-mobilenetv3_small_100']
        activation = 'softmax' if n_classes > 2 else 'sigmoid'
        super().__init__(backbone, in_channels=3, encoder_depth=5, decoder_merge_policy='cat', encoder_weights=encoder_weights,
                        aux_params={'classes':n_classes, 'pooling': "avg", 'activation':activation})
        # extract ffeatures only (*, 2048, 1, 1)
        self.feat_out = torch.nn.Sequential(*list(self.classification_head.children())[:1])
        activation = 'softmax' if n_classes > 2 else 'sigmoid'
        
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.feat_out(x)
        return x
    
     
class Linknetclassifier(smp.Linknet):
    def __init__(self, backbone = 'timm-res2net50_14w_8s', n_classes=2, encoder_weights=None, **kwargs):
        # https://github.com/qubvel/segmentation_models.pytorch#encoders 
        assert backbone in ['timm-res2net50_14w_8s','efficientnet-b7', 'timm-mobilenetv3_large_minimal_100',
                            'timm-mobilenetv3_small_100']
        activation = 'softmax' if n_classes > 2 else 'sigmoid'
        super().__init__(backbone, in_channels=3, encoder_depth=5, encoder_weights=encoder_weights,
                        aux_params={'classes':n_classes, 'pooling': "avg", 'activation':activation})
        # extract ffeatures only (*, 2048, 1, 1)
        self.feat_out = torch.nn.Sequential(*list(self.classification_head.children())[:1])
        activation = 'softmax' if n_classes > 2 else 'sigmoid'
        
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.feat_out(x)
        return x
    
class HEClassificationModel(pl.LightningModule):
    def __init__(self, model_name:str, n_classes:int=2, pretrain:bool=True,
                 input_size:tuple=(224,224),  log_metrics:bool=False, 
                 sync_dist=False, debug:bool=False):
        super().__init__()
        print(f"Using pre-trained head:{model_name}")
        avail_models =  ['mobilenetv3','resnext101','efficientnetb7','resnet50','Multiscale-Linknet','Multiscale-FPN']
        assert model_name in avail_models, f"Must be one of {avail_models}"
        self.debug = debug
        self.n_classes = n_classes
        self.sync_dist = sync_dist
        # Step 1: Initialize model with the weights
        if model_name in avail_models[:4]:
            if model_name == 'mobilenetv3':
                self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrain else None)
            elif model_name == 'resnext101':
                self.model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrain else None)
            elif model_name == 'efficientnetb7':
                self.model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrain else None)
            elif model_name =='resnet50':
                self.model = models.resnet50(pretrained=ResNet50_Weights.IMAGENET1K_V2 if pretrain else None)
            # replace/remove head
            removed = list(self.model.children())[:-1]
            self.model_base = torch.nn.Sequential(*removed) 
        else:
            # Model with a (*, n, 1,1) output
            if model_name == 'Multiscale-Linknet':
                self.model_base = Linknetclassifier(pretrained=pretrain)
            elif model_name == 'Multiscale-FPN':
                self.model_base = FPNclassifier(pretrained=pretrain)
         
        in_feats = self._get_output_feat(self.model_base, input_size)
        # head
        self.model_head = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_feats, out_features=self.n_classes),
                                        nn.LogSoftmax(dim=1) if n_classes>2 else nn.Sigmoid(),
                                       )
        self.model = torch.nn.Sequential(self.model_base, self.model_head)
            #self.model_head.to(device=META_ARGS.device)     
        # metrics
        self.log_metrics = log_metrics
        if log_metrics:
            self.metric_device = 'cpu'
            self.accuracy = torchmetrics.Accuracy().to(self.metric_device)
            self.recall = torchmetrics.Recall(average='macro', num_classes=2).to(self.metric_device)
            #self.ROC = torchmetrics.ROC(num_classes=n_classes)
            self.AUROC = torchmetrics.AUROC(num_classes=n_classes, pos_label=1).to(self.metric_device)

    def _get_output_feat(self, model, in_shape=(224,224)):
        x = torch.randn((3,)+in_shape)
        return model(x.unsqueeze(0)).flatten().size()[0]

    def _forward_feature_extract(self, x):
        return self.model_base(x)

    def forward(self, x):
        x = self.model(x)
        if self.debug: print(f"Num classes:{self.n_classes}\nModel classifier\n:{self.model_head}")
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1e5)
        return [optimizer], [lr_scheduler]

    def get_loss(self, y_hat, y):
        #loss = nn.CrossEntropyLoss()   # does softmax for you (no need in classifcation)
        #loss = nn.LogSoftmax()
        #loss = F.nll_loss
        if self.debug: print(y.size(), y.dtype, y_hat.size(), y_hat.dtype)
        return F.cross_entropy(y_hat,  y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        # training metrics
        
        # optimize (done under the hoood)
        if self.log_metrics:
            acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
            self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
        return loss
        #return self.get_loss(y, y_hat)

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        # compute metrics
        val_loss =self.get_loss(y_hat, y)
        if self.log_metrics:
            acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device).detach(), y.to(self.metric_device).detach())
            #auroc = self.AUROC(y_hat.to(self.metric_device), y.to(self.metric_device))
            #fpr, tpr, thresholds = self.ROC(y_hat, y)

            self.log("val_loss", val_loss)
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_dist)
            #self.AUROC.update(y_hat.cpu().detach(), y.cpu().detach())
            #self.log("validation_auc", self.AUROC, on_step=False, on_epoch=True, sync_dist=self.sync_dist)   # prog_bar=True,


class HEEnsembleModel(pl.LightningModule):
    def __init__(self, 
                 ensembles_settings:dict={'efficientnetb7':3, 'resnext101':2}, 
                 pretrain:bool=True,
                 n_classes:int=2,
                 input_shape=(224,224),
                 metrics={},
                 debug=False):
        super(HEEnsembleModel, self).__init__()
        self.debug = debug
        self.sync_dist = False
        models = []
        self.n_models = 0
        for name, number in ensembles_settings.items():
            [models.append(
                HEClassificationModel(model_name=name, 
                                      n_classes=2, 
                                      pretrain=pretrain,
                                      log_metrics=False,
                                     )
                         ) for i in range(number)
            ]
            self.n_models += number
        self.ensemble_model = torch.nn.ModuleList(models)
        self.classifier = torch.nn.Linear(self.n_models*n_classes, n_classes)
        #self.save_hyperparameters() # Uncomment to show error
        self.CEloss = nn.CrossEntropyLoss()
        # metrics
        self.metrics = metrics
        self.accuracy = torchmetrics.Accuracy()
        self.recall = torchmetrics.Recall(average='macro', num_classes=2)
        #self.AUROC = torchmetrics.AUROC(num_classes=n_classes, pos_label=1)

    def forward(self, x):
        output=[]
        for m in self.ensemble_model:
            output.append(m(x))
        combined = torch.concat(output,dim=1)
        x = self.classifier(combined)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-10)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def get_loss(self, y_hat, y):
        #loss = nn.CrossEntropyLoss()   # does softmax for you (no need in classifcation)
        #loss = nn.LogSoftmax()
        #loss = F.nll_loss
        if self.debug: print(y.size(), y.dtype, y_hat.size(), y_hat.dtype)
        return self.CEloss(y_hat,  y)

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        # training metrics
        # acc = self.accuracy(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        # rec = self.recall(torch.argmax(y_hat, dim=1).to(self.metric_device), y.to(self.metric_device))
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        rec = self.recall(torch.argmax(y_hat, dim=1), y)
        # optimize (done under the hoood)

        self.log('train_loss', loss, on_step=True, on_epoch=True,  sync_dist=self.sync_dist)
        self.log('train_acc', acc, on_epoch=True, sync_dist=self.sync_dist)
        self.log('train_rec', rec, on_epoch=True,  sync_dist=self.sync_dist)
        return loss
        #return self.get_loss(y, y_hat)

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        # compute metrics
        val_loss =self.get_loss(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        rec = self.recall(torch.argmax(y_hat, dim=1), y)
        if hasattr(self, 'AUROC'):
            auroc = self.AUROC(y_hat.to(self.metric_device), y.to(self.metric_device))
        #fpr, tpr, thresholds = self.ROC(y_hat, y)
        #
        self.log("val_loss", val_loss, on_step=True, sync_dist=self.sync_dist)
        self.log('val_acc', acc,  on_epoch=True, sync_dist=self.sync_dist)
        self.log('val_rec', rec,  on_epoch=True, sync_dist=self.sync_dist)
        if hasattr(self, 'AUROC'):
            self.AUROC.update(y_hat.to(self.metric_device), y.to(self.metric_device))
            self.log("validation_auc", auroc, on_step=False, on_epoch=True, sync_dist=self.sync_dist)  # prog_bar=True
            
    def test_step(self, batch, batch_idx=None):
        x, y = batch
        y_hat = self(x)
        # compute metrics
        test_loss =self.get_loss(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        rec = self.recall(torch.argmax(y_hat, dim=1), y)
        if hasattr(self, 'AUROC'):
            auroc = self.AUROC(y_hat.to(self.metric_device), y.to(self.metric_device))
        #fpr, tpr, thresholds = self.ROC(y_hat, y)
        #
        self.log("test_loss", test_loss, on_step=True, sync_dist=self.sync_dist)
        self.log('test_acc', acc,  on_epoch=True, sync_dist=self.sync_dist)
        self.log('test_rec', rec,  on_epoch=True, sync_dist=self.sync_dist)
        if hasattr(self, 'AUROC'):
            self.AUROC.update(y_hat.to(self.metric_device), y.to(self.metric_device))
            self.log("validation_auc", auroc, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
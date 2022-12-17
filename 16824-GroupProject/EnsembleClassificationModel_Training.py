#!/usr/bin/env python
# coding: utf-8

# Data from: http://zhao-nas.bio.cmu.edu:5000/fsdownload/aBDx29J7H/Ensemble%20learning%20data_shared

from pathlib import Path
import cv2 , os, numpy as np, torch, pandas as pd

# 
from torch import nn, optim
import torch.nn.functional as F
#import torchvision.transforms as T

from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
#
from ADUtils.data import *
from ADUtils.models import *
from ADUtils.callbacks import *
import pytorch_lightning as pl, torchmetrics
import os
#os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"  # nccl (not for windows)
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# https://pytorch.org/docs/stable/distributed.html
#     Rule of thumb
#         Use the NCCL backend for distributed GPU training
#         Use the Gloo backend for distributed CPU training


AVAIL_GPUS = torch.cuda.device_count()
AVAIL_CPUS = os.cpu_count()
print(f"GPUS:{AVAIL_GPUS}|CPUS:{AVAIL_CPUS}")

class PATH_ARGS:
    proj_path = Path('./').absolute()  # [CHANGE THIS for new environment]
    name = "cycleGAN_augmented-cls_model(pretrained)"
    model_path = proj_path/'model_chkpts'/name
    # data path
    #data_path = proj_path/'TestingData'   # Test path
    #data_path = proj_path/'Ensemble_learning data'      # [CONFIRM THIS for new environment]
    data_path = proj_path.parent
    # 2 types of images (HE  FISH)
    data_name = ['HE_RBG_Corp_images']
    dataindex_fn = data_name[0]+'_processed/dataIndex(ubuntu).csv'
    dataindex_path = data_path/dataindex_fn
    #data_name = ['HE images', 'HIPT_AGH_FluorescentImage_R1']
    # 2 groups to classify
    class_names = ['Responder','NonResponder']

print(PATH_ARGS.__dict__)
def mkdirifNE(p):
    if not os.path.exists(p): os.mkdir(p)

def load_img(img_paths: list, is_mask=False):
        """ load array from a list of image paths """
        if is_mask: flag = 0
        else: flag = -1
        return np.concatenate([np.expand_dims(cv2.imread(str(img_fp), flag), axis=0)
                               for img_fp in img_paths.tolist()])
def normalize(ratios):
    """normalize a list of ratios to sum to 1"""
    return [r/sum(ratios) for r in ratios]

class META_ARGS:
        RANDOM_SEED = 42
        INPUT_DIM = (224,224)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_normalize_attributes(data_index_df):
    x_imgs = load_img(data_index_df['x_img_path'])
    means, stds = np.mean(x_imgs, axis=((0,1,2))), np.std(x_imgs, axis=((0,1,2)))
    return means, stds


class MODEL_ARGS:
    n_classes = len(PATH_ARGS.class_names)

class TRAIN_ARGS:
    batch_size = 16 if AVAIL_GPUS else 512
    epochs = 100

def main(args):
    mkdirifNE(PATH_ARGS.model_path)
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    #
    index_cols = ['parent_path', 'type', 'tissue','x_img_path']
    data_index_df = pd.read_csv(args.dataindex_path, index_col=list(range(len(index_cols))))
    print(f"Index file index: {data_index_df.index.names}, columns: {data_index_df.columns}")
    # DEFAULT (ie: no accumulated grads)
    model_chkpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=PATH_ARGS.model_path,
                                                  filename='models-{epoch:02d}-{val_loss:.2f}', save_top_k=2, mode='min')
    cbs = [
        model_chkpt_cb,
        pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=8, mode="min"),
        #pl.callbacks.GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1}),
        #PRMetrics(),
    ]
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,   #"gpu",
        devices="auto" if args.n_devices is None else args.n_devices,   #2,
        logger=WandbLogger(project=args.wandb_project_name,  entity="3m-m", job_type='train'),
        max_epochs=TRAIN_ARGS.epochs, callbacks=cbs,
        #strategy='dp', # "horovod",  # dp
        #plugins=DDPPlugin(find_unused_parameters=True),
    )
    
    model = HEEnsembleModel(
        ensembles_settings={'resnext101':1},
        pretrain=args.pretrained,
        input_shape=(224,224),
        n_classes=MODEL_ARGS.n_classes,
        debug=False
    )
    
    # train
    datamodule = HEDataModule(batch_size=args.batchsize, dataindex_path=args.dataindex_path, patch_size=args.patchsize,
                              index_cols=index_cols, debug=False)
    datamodule.setup()
    trainer.fit(model=model, datamodule=datamodule) 
        # save with parameters
    # test
    print(f" Testing from best checkpoint:{model_chkpt_cb.best_model_path}")

    #result = trainer.validate(model, test_dataloader)
    result = trainer.test(model=model, datamodule=datamodule)
    


# ### Prediction/submission
#test_loader = DataLoader(HEdatasets['test'], shuffle=True, batch_size=TRAIN_args.test_batch_size)

if __name__ == "__main__":
    import datetime, argparse
    print(f"Starting run at:{datetime.datetime.now()}")
    def parse_arguments():
        # Command-line flags are defined here.
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataindex_path', dest='dataindex_path', type=str, required=True,
                            default=PATH_ARGS.dataindex_path, help="dataindex path")
        parser.add_argument('--pretrained', dest='pretrained', type=bool,
                            default=True, help="Loading pretrained weights")
        parser.add_argument('--batchsize', dest='batchsize', type=int,
                            default=TRAIN_ARGS.batch_size, help="batch size")
        parser.add_argument('--patchsize', dest='patchsize', type=int,
                            default=None, help="patch size")
        parser.add_argument('--accelerator', dest='accelerator', type=str,
                            default='gpu', help="type of accelerator")
        parser.add_argument('--wandb_project_name', dest='wandb_project_name',type=str,
                            default='VLR-FinalProject', help="wandb project name")
        parser.add_argument('--n_devices', dest='n_devices', type=int,
                            default=None, help="Number of devices")
        
        #parser_group = parser.add_mutually_exclusive_group(required=False)
        #parser_group.add_argument('--render', dest='render', action='store_true', help="Whether to render the environment.")
        return parser.parse_args()
    
    args = parse_arguments()
    main(args)
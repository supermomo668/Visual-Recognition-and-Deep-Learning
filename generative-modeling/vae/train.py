from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *

def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """
    z = model.encoder(x)
    loss = torch.square(model.decoder(z)-x).sum(dim=(1,2,3)).mean()
    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """
    def kl_divergence(z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        # 2. get the probabilities from the equation # kl
        kl = (q.log_prob(z) - p.log_prob(z)).sum(dim=-1)
        return kl
    # Not used
    def likelihood(x_recon, x):
        dist = torch.distributions.Normal(x_recon, torch.tensor([1.0]).cuda())
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))
    # compute sampled z
    mu, log_var = model.encoder(x)   # (*, z_dim)
    std = torch.exp(0.5*log_var)  # (*, z_dim)
    z = torch.distributions.Normal(mu, std).rsample()  # (*, z_dim)
    #
    #recon_loss = -1.0*likelihood(model.decoder(z), x).mean()  # x_recon vs x
    recon_loss = torch.square(model.decoder(z)-x).sum(dim=(1,2,3)).mean()
    kl_loss = kl_divergence(z, mu, std).mean()
    total_loss = recon_loss + beta*kl_loss
    return total_loss, OrderedDict(srecon_loss=recon_loss, kl_loss=kl_loss)

def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """
    def _helper(epoch):
        return torch.tensor(epoch/max_epochs)
    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)
    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    os.makedirs('vae/vae_data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vis_x = next(iter(val_loader))[0][:36]
    val_loss = {'srecon_loss':[], 'kl_loss':[]}
    #beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        #TODO : add plotting code for metrics (required for multiple parts)
        for k, m in val_loss.items():
            val_loss[k].append(val_metrics[k])
            save_plot(range(epoch+1), val_loss[k], k, 'epochs', f'{k} vs epochs', 'vae_data/'+log_dir+f'/{k}_vs_epoch')

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'vae/vae_data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'vae/vae_data/'+log_dir+ '/epoch_'+str(epoch))


if __name__ == '__main__':
    import argparse 
    def parse_a2c_arguments():
        # Command-line flags are defined here.
        parser = argparse.ArgumentParser()
        parser.add_argument('--latent_size', dest='latent_size', type=int,
                            default=1024, help="Size of latent space")   # 'LunarLander-v2'
        parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                            default=20, help="Size of latent space")   # 'LunarLander-v2'
        parser.add_argument('--loss_mode', dest='loss_mode', type=str,
                            default='vae', help="Size of latent space")   # 'LunarLander-v2'
        parser.add_argument('--log_dir', dest='log_dir', type=str,
                            default='ae_latent1024', help="directory")
        # ['ae_latent1024','vae_latent1024', 'vae_latent1024_beta_constant0.8','vae_latent1024_beta_linear1']
        parser.add_argument('--beta_mode', dest='beta_mode', type=str,
                            default='constant', help="directorye")   
        # ['constant', 'linear']
        parser.add_argument('--target_beta_val', dest='target_beta_val', type=float,
                            default=0.8, help="final beta")   # 
        # [0.8. 1]
        return parser.parse_known_args()[0]  #parser.parse_args()
    args = parse_a2c_arguments().__dict__
    #TODO: Experiments to run : 
    #2.1 - Auto-Encoder
    #Run for latent_sizes 16, 128 and 1024
    args['loss_mode'] = 'ae'
    exp_params = {
        "log_dir": ['ae_latent16','ae_latent128','ae_latent1024'],
        "latent_size": [16, 128, 1024]
    }
    for i in range(3):
        for p, v in exp_params.items():
            args[p] = v[i]
        main(**args)  
    #Q 2.2 - Variational Auto-Encoder
    
    args['loss_mode'] = 'vae'
    args['log_dir'] = 'vae_latent1024'
    main(**args)  
    #Q 2.3.1 - Beta-VAE (constant beta)
    #Run for beta values 0.8, 1.2
    args['beta_mode'] = 'constant'
    main(**args)  
    exp_params = {
        "log_dir": ['vae_latent1024_beta_constant0.8','vae_latent1024_beta_constant1','vae_latent1024_beta_constant1.2'],
        "target_beta_val": [0.8, 1, 1.2]
    }
    for i in range(3):
        for p, v in exp_params.items():
            args[p] = v[i]
        main(**args)  
    
    #Q 2.3.2 - VAE with annealed beta (linear schedule)
    args['beta_mode'] = 'linear'
    args['log_dir'] = 'vae_latent1024_beta_linear1'
    main(**args)  
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
        debug=False
    ):
        super(DiffusionModel, self).__init__()
        assert model.channels == model.out_dim
        assert not model.learned_sinusoidal_cond
        self.debug = debug
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.device = torch.cuda.current_device()

        self.num_timesteps = timesteps   #self.betas.shape[0]
        if sampling_timesteps:
            self.time_steps = np.asarray(list(range(1, self.num_timesteps, self.num_timesteps//sampling_timesteps))) + 1
            print(self.time_steps.shape)
        else:
            self.time_steps = np.arange(timesteps)
        
        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        alphas = 1. - self.betas
        # TODO (Q3.1): compute the cummulative products for current and previous timesteps
        
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # TODO (Q3.1): pre-compute the alphas needed for forward process
        # Hint: you should look at all the equations and see what you can pre-compute
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = self._get_posterior_variance()*ddim_sampling_eta
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # TODO (Q 3.1): compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = self.betas * self.sqrt_alphas_cumprod / (1. - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = 1/self.sqrt_recip_alphas*(1-self.alphas_cumprod_prev)/ (1-self.alphas_cumprod)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
    
    def _get_posterior_variance(self):
        # TODO (Q3.1): compute the variance of the posterior distribution
        # Hint: this is the \sigma_{t} in the DDPM section
        return self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def predict_start_image_from_noise(self, x_t, t, noise):
        # TODO (Q3.1): given a noised image x_t and noise tensor, predict x_0
        x_start = 1/self.sqrt_alphas_cumprod[t]*(x_t-self.sqrt_one_minus_alphas_cumprod[t]*noise)
        return x_start

    def get_posterior_parameters(self, x_start, x_t, t):
        # Compute the posterior mean and variance for x_{t-1} 
        # using the coefficients, x_t, and x_0
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t):
        # TODO (Q3.1): given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        if self.debug: print(f"model input:{x.size(), t.size()}")
        
        pred_noise = self.model(x, t)
        x_start = self.predict_start_image_from_noise(x, t[0], pred_noise)
        if self.debug: [print(f"model_predictions: {p.size()}") for p in [pred_noise, x_start]]
        return (pred_noise, x_start)

    def mean_variance_at_previous_timestep(self, x, t):
        # TODO (Q3.1): predict the mean and variance for the posterior (x_{t-1})
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?
        t = torch.full((x.size()[0],), t, device=torch.device(self.device), dtype=torch.long)
        pred_noise, x_start = self.model_predictions(x, t)
        x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.get_posterior_parameters(x_start, x, t)
        if self.debug: 
            for p in [model_mean, posterior_variance, posterior_log_variance, x_start]: 
                print(f"mean_variance_at_previous_timestep: {p.size()}")
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO (3.1): given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        model_mean, posterior_variance, posterior_log_variance, x_start = self.mean_variance_at_previous_timestep(x, t)
        pred_img = model_mean +  torch.sqrt(posterior_variance) * noise
        if self.debug:
            for p in [pred_img, x_start]:
                print(f"predict_denoised_at_prev_timestep: {p.size()}")
        return pred_img, x_start

    @torch.no_grad()
    def ddpm_sample(self, shape):
        # TODO (Q3.1): implement the DDPM sampling process.
        # Hint: while returning the final image, ensure it is scaled between [0,1].
        img = torch.randn(shape, device=torch.device(self.device))
        for t in range(0, self.num_timesteps)[::-1]:
            #t = torch.full((shape[0],), i, device=torch.device(self.device), dtype=torch.long)
            img, x_0 = self.predict_denoised_at_prev_timestep(img, torch.tensor(t).to(self.device))
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape):
        # TODO (Q3.2.1): implement the DDIM sampling process.
        # 'uniform':
        print(f"Using ddim sampling:{len(range(0, self.num_timesteps, self.num_timesteps//self.sampling_timesteps))} time steps.")
        img = torch.randn(shape, device=torch.device(self.device))
        #for t in range(0, self.num_timesteps, self.num_timesteps//self.ddim_sampling_eta)[::-1]:
        for t in range(0, self.num_timesteps)[::-1]:
            new_img, x_0 = self.predict_denoised_at_prev_timestep(img, torch.tensor(t).to(self.device))
            if t %self.sampling_timesteps==0:
                img = new_img
        img = unnormalize_to_zero_to_one(img)
        return img

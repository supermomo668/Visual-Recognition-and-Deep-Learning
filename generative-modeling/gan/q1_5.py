import os

import torch

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # compute gradient penalty
    def get_gradient(discrim_interp, interp):
        # interp = real * epsilon + fake * (1 - epsilon)
        # discrim_interp = disc(interp)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interp ,  # interpolated images,
            outputs=discrim_interp,   # interpolate images scores
            grad_outputs=torch.ones_like(discrim_interp), 
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient  # (*, 3, 64, 64)  # im shape
    gradient = get_gradient(discrim_interp, interp)
    # Penalty: mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient.view(len(gradient), -1).norm(2, dim=1) - 1)**2)
    # put together loss
    loss = torch.mean(discrim_fake) - torch.mean(discrim_real) + lamb * penalty
    return loss


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    gen_loss = -1. * torch.mean(discrim_fake)
    return gen_loss



if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=64,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )

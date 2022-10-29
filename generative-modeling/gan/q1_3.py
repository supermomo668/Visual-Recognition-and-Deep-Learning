import os

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.3.1: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    #criterion = torch.nn.BCEWithLogitsLoss()
    bs = len(discrim_real)
    disc_fake_loss = F.binary_cross_entropy_with_logits(discrim_fake.view(-1), torch.zeros_like(discrim_fake).view(-1))
    disc_real_loss = F.binary_cross_entropy_with_logits(discrim_real.view(-1), torch.ones_like(discrim_real).view(-1))
    return (disc_fake_loss + disc_real_loss)


def compute_generator_loss(discrim_fake):
    # TODO 1.3.1: Implement GAN loss for generator.
    #criterion = torch.nn.BCEWithLogitsLoss()
    bs = len(discrim_fake)
    gen_loss = F.binary_cross_entropy_with_logits(discrim_fake.view(-1), torch.ones_like(discrim_fake).view(-1))
    return gen_loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
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

from glob import glob
import os
import torch
import wandb
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import VisionDataset

import pathlib
proj_path = pathlib.Path('./')

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.
    lr = 2e-4
    disc = disc.cuda()
    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0, 0.9))
    scheduler_discriminator = torch.optim.lr_scheduler.LinearLR(
        optim_discriminator, start_factor=1, end_factor=0, total_iters=int(5e5), last_epoch=-1, verbose=False)
    optim_generator = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0, 0.9))
    scheduler_generator = scheduler_discriminator = torch.optim.lr_scheduler.LinearLR(
        optim_generator, start_factor=1, end_factor=0, total_iters=int(1e5), last_epoch=-1, verbose=False)
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    wandb_logging = False
):
    if wandb_logging:
        wandb.init(project="vlr-hw2", reinit=False)
    datadir_exist = os.path.exists("gan/datasets/CUB_200_2011_32")
    print(f"data dir exist:{datadir_exist}")
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="gan/datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    #
    scaler = torch.cuda.amp.GradScaler()
    #
    iters = 0
    fids_list = []
    iters_list = []

    vislogger_train = wandb.Table(columns=["Iter", "Generated Image",  "Interpolation"])
    while iters < num_iterations:
        print(f"Epoch after iters:{iters}")
        for i, train_batch in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                train_batch = train_batch.cuda()
                bs = len(train_batch)
                # TODO 1.2: compute generator outputs and discriminator outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                gen_batch = gen(n_samples=bs)   # [N, 3, 64, 64]
                discrim_fake = disc(gen_batch)
                discrim_real = disc(train_batch)

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data
                #discrim_interp = interpolate_latent_space(gen, prefix+f'interpolated-iter{iters}batch{i}.jpg')

                epsilon = torch.rand((bs,)+(1,)*3, device='cuda', requires_grad=True)    # (*, 1,1,1)
                interp = train_batch * epsilon + gen_batch * (1 - epsilon)
                discrim_interp = disc(interp)   
                discriminator_loss = disc_loss_fn(
                    discrim_real, discrim_fake, discrim_interp, interp, lamb
                )
                wandb.log({'train/disc. Loss': discriminator_loss})
            ## 
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward(retain_graph=True)
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast():
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    gen_val = gen(n_samples=batch_size)
                    discrim_fake_val = disc(gen_val)
                    generator_loss = gen_loss_fn(discrim_fake_val)
                wandb.log({'train/gen. Loss': generator_loss})
                ##
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                fid_bs = batch_size   # 256
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        generated_samples = torch.clamp(gen(n_samples=100), 0, 1)
                    #
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    torch.jit.save(gen, prefix + "generator.pt")
                    torch.jit.save(disc, prefix + "discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=fid_bs,
                        num_gen=10_000,
                    )
                    wandb.log({'val/FID': fid})
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )

                vislogger_train.add_data(iters, 
                                         wandb.Image(Image.open(prefix + "samples_{}.png".format(iters))),
                                         wandb.Image(Image.open(prefix + "interpolations_{}.png".format(iters)))
                                        )
            wandb.log({f"val/Visuals": vislogger_train})
            ##
            scaler.update()
            iters += 1
    print("Training Done.")
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=fid_bs,
        num_gen=50_000,
    )
    wandb.log({f"val/FinalFID": fid})
    print(f"Final FID (Full 50K): {fid}")

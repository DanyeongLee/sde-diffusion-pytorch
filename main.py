import os
from tqdm import tqdm

import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch_ema import ExponentialMovingAverage
from src.unet import Unet

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import wandb



def train(model, optimizer, ema, sde, dataloader, device):
    epoch_loss = 0.
    model.train()
    for x, _ in tqdm(dataloader):
        optimizer.zero_grad()
        x = x.to(device)
        loss = sde.score_matching_loss(model, x)
        loss.backward()
        optimizer.step()
        ema.update()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


@torch.no_grad()
def validation(model, sde, ema, dataloader, device):
    with ema.average_parameters():
        epoch_loss = 0.
        epoch_samples = 0
        model.eval()
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            loss = sde.score_matching_loss(model, x)
            epoch_loss += loss.item() * x.shape[0]
            epoch_samples += x.shape[0]
        epoch_loss = epoch_loss / epoch_samples

        samples = sde.predictor_corrector_sample(model, (8, 1, 28, 28), device)
        samples = torch.clamp(samples, 0., 1.)

    return epoch_loss, samples



@hydra.main(version_base='1.3', config_path='configs', config_name='main')
def main(cfg: DictConfig):
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, name=cfg.name, config=wandb_config)

    device = torch.device(cfg.device)

    model = Unet(
        dim=cfg.model.dim,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.channels
    ).to(device)
    sde = instantiate(cfg.sde).to(device)
    optimizer = instantiate(cfg.optimizer)(params=model.parameters())    
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.train.ema_decay)

    if cfg.data.dataset == 'mnist':
        train_data = MNIST(root=cfg.data.root, train=True, download=True, transform=transforms.ToTensor())
    elif cfg.data.dataset == 'cifar10':
        train_data = CIFAR10(root=cfg.data.root, train=True, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    train_data, val_data = random_split(train_data, [50000, 10000])
    train_dataloader = DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True, num_workers=cfg.data.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False, num_workers=cfg.data.num_workers)


    for epoch in range(cfg.train.n_epochs):
        print(f'Epoch {epoch}')
        train_loss = train(model, optimizer, ema, sde, train_dataloader, device)
        val_loss, samples = validation(model, sde, ema, val_dataloader, device)
        wandb.log({'train/loss': train_loss, 'val/loss': val_loss})
        wandb.log({'samples': [wandb.Image(sample) for sample in samples]})

    ckpt_dir = os.path.dirname(to_absolute_path(cfg.ckpt_path))
    os.makedirs(to_absolute_path(ckpt_dir), exist_ok=True)
    with ema.average_parameters():
        torch.save(model.state_dict(), cfg.ckpt_path)
    
if __name__ == '__main__':
    main()

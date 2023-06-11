import os
import argparse
from tqdm import tqdm
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from src.sde import VE_SDE, VP_SDE, SubVP_SDE
from src.unet import Unet


def main(args):
    os.makedirs('ckpts', exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    train_data = FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

    model = Unet(
        dim=28,
        dim_mults=(1, 2, 4),
        channels=1
    ).to(device)

    if args.sde == 've':
        sde = VE_SDE(rescale=True).to(device)
    elif args.sde == 'vp':
        sde = VP_SDE(rescale=True).to(device)
    elif args.sde == 'subvp':
        sde = SubVP_SDE(rescale=True).to(device)
    else:
        raise ValueError('Invalid SDE type')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-12)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    for epoch in range(50):
        print(f'Epoch {epoch}')
        epoch_loss = 0.
        for x, _ in tqdm(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            loss = sde.score_matching_loss(model, x)
            loss.backward()
            optimizer.step()
            ema.update()
            epoch_loss += loss.item()
        
        print(f'Loss: {epoch_loss / len(train_dataloader)}')

    with ema.average_parameters():
        torch.save(model.state_dict(), f'ckpts/{args.sde}.ckpt')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sde', type=str, default='subvp', choices=['ve', 'vp', 'subvp'])
    args = parser.parse_args()
    main(args)


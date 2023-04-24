from tqdm import tqdm
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from src.sde import VE_SDE, VP_SDE, SubVP_SDE
from src.unet import Unet



train_data = MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
#train_data = CIFAR10(root='../../data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

device = torch.device('cuda:1')

model = Unet(
    dim=28,
    dim_mults=(1, 2, 4),
    channels=1
).to(device)
sde = SubVP_SDE(rescale=True).to(device)
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
    torch.save(model.state_dict(), 'ckpts/sub_vp_sde_mnist.pth')
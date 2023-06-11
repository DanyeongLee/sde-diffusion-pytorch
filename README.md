# sde-diffusion-pytorch
A minimal PyTorch implementation of SDE-based diffusion models.

# Usage
## Training
```python
import torch
from src.unet import Unet
from src.sde import VE_SDE, VP_SDE, SubVP_SDE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = ...
train_loader = DataLoader(train_data, ...)

sde = VE_SDE().to(device)
model = Unet(...).to(device)
optimizer = torch.optim.Adam(...)

for epoch in range(10):
    for x, _ in train_loader:
        optimizer.zero_grad()
        loss = sde.score_matching_loss(model, x.to(device))
        loss.backward()
        optimizer.step()
```

## Sampling
```python

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sde = VE_SDE().to(device)
model = Unet(...).to(device)
model.load_state_dict(...)

shape = (32, 1, 28, 28)  # generate 32 samples
samples = sde.predictor_corrector_sample(model, shape, device)
```

# Generated samples (example)
![samples](/samples.png?raw=true)

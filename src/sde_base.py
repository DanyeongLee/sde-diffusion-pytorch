from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn


class SDEBase(nn.Module):
    def __init__(self, train_eps=1e-5, sample_eps=1e-3, rescale=False):
        super().__init__()
        self.eps = train_eps
        self.sample_eps = sample_eps
        self.rescale = rescale

    def drift_coef(self, x, t):
        pass
    
    def diffusion_coef(self, t):
        pass
    
    def x0_coef(self, t):
        pass
    
    def sigma_t(self, t):
        pass

    def match_dim(self, x, y):
        while len(x.shape) < len(y.shape):
            x = x.unsqueeze(-1)
        return x

    def forward(self, model, x, t):
        score = model(x, t)
        if self.rescale:
            sigma = self.sigma_t(t)
            while len(sigma.shape) < len(x.shape):
                sigma = sigma.unsqueeze(-1)
            score = score / sigma
        
        return score
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        s_t = self.sigma_t(t)
        x0_coef = self.x0_coef(t)
        s_t = self.match_dim(s_t, x_0)
        x0_coef = self.match_dim(x0_coef, x_0)
        
        return x0_coef * x_0 + s_t * noise
    
    def sample_time_batch(self, batch_size, device=None):
        return (self.eps - 1) * torch.rand(batch_size, device=device) + 1
    
    def score_matching_loss(self, model, x):
        time = self.sample_time_batch(x.shape[0], x.device)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, time, noise)
        score = self.forward(model, x_t, time)
        sigma = self.sigma_t(time)
        sigma = self.match_dim(sigma, x)
        loss = torch.mean(torch.sum((sigma * score + noise) ** 2, dim=(1, 2, 3)))

        return loss
    
    @torch.no_grad()
    def euler_maruyama_step(self, model, x, t, delta_t):
        score = self.forward(model, x, t)
        g = self.diffusion_coef(t)
        g = self.match_dim(g, x)
        
        if t[0] == self.sample_eps:
            z = 0
        else:
            z = torch.randn_like(x)

        drift = self.drift_coef(x, t)
        x_mean = x - (drift - (g**2) * score) * delta_t
        x_new = x_mean + g * np.sqrt(delta_t) * z

        return x_new

    @torch.no_grad()
    def euler_maruyama_sample(self, model, shape, device, n_steps=500):
        x_t = torch.randn(shape).to(device)
        time_steps = np.linspace(1, self.sample_eps, n_steps)
        delta_t = time_steps[0] - time_steps[1]
        for t in tqdm(time_steps):
            time_batch = torch.ones(shape[0], device=device) * t
            x_t = self.euler_maruyama_step(model, x_t, time_batch, delta_t)
        
        return x_t
    
    @torch.no_grad()
    def langevin_step(self, model, x, t, snr=0.16):
        score = self.forward(model, x, t)
        noise = torch.randn_like(x)
        score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        step_size = 2 * (snr * noise_norm / score_norm)**2

        x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        return x
    
    @torch.no_grad()
    def predictor_corrector_step(self, model, x, t, delta_t, n_lang_steps=1, snr=0.16):
        for i in range(n_lang_steps):
            x = self.langevin_step(model, x, t, snr)

        x = self.euler_maruyama_step(model, x, t, delta_t)
        
        return x
    
    @torch.no_grad()
    def predictor_corrector_sample(self, model, shape, device, n_steps=500, n_lang_steps=1, snr=0.16):
        x_t = torch.randn(shape).to(device)
        time_steps = np.linspace(1, self.sample_eps, n_steps)
        delta_t = time_steps[0] - time_steps[1]
        for t in tqdm(time_steps):
            time_batch = torch.ones(shape[0], device=device) * t
            x_t = self.predictor_corrector_step(model, x_t, time_batch, delta_t, n_lang_steps, snr)

        return x_t
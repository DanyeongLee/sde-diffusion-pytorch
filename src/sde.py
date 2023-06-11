import numpy as np
import torch
from src.sde_base import SDEBase



class VE_SDE(SDEBase):
    '''
    An SDE version of NCSN
    '''
    def __init__(self, sigma_min=0.01, sigma_max=1., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def drift_coef(self, x, t):
        return torch.zeros_like(x)

    def sigma_t(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def diffusion_coef(self, t):
        s_t = self.sigma_t(t)
        return s_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
    
    def x0_coef(self, t):
        return torch.ones_like(t)


class VP_SDE(SDEBase):
    '''
    An SDE version of DDPM.
    '''
    def __init__(self, beta_min=0.1, beta_max=20., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift_coef(self, x, t):
        drift = self.beta_t(t)
        drift = self.match_dim(drift, x)
        drift = - drift * x / 2
        return drift
    
    def diffusion_coef(self, t):
        return torch.sqrt(self.beta_t(t))
    
    def x0_coef(self, t):
        x = - t**2 * (self.beta_max - self.beta_min) / 4
        x = x - t * self.beta_min / 2
        return torch.exp(x)
    
    def sigma_t(self, t):
        x = self.x0_coef(t)
        return torch.sqrt(1 - x**2)
    


class SubVP_SDE(SDEBase):
    def __init__(self, beta_min=0.1, beta_max=20., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def beta_t_integrated(self, t):
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2

    def drift_coef(self, x, t):
        drift = self.beta_t(t)
        drift = self.match_dim(drift, x)
        drift = - drift * x / 2
        return drift
    
    def diffusion_coef(self, t):
        coef = self.beta_t(t)
        coef *= 1 - torch.exp(-2 * self.beta_t_integrated(t))
        return torch.sqrt(coef)
    
    def x0_coef(self, t):
        x = - t**2 * (self.beta_max - self.beta_min) / 4
        x = x - t * self.beta_min / 2
        return torch.exp(x)
    
    def sigma_t(self, t):
        x = self.x0_coef(t)
        return 1 - x**2
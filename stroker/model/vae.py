# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


TTensor = torch.Tensor

class VAEData:

    def __init__(self, recons: TTensor, latent: TTensor, mean: TTensor, log_dev: TTensor):
        self.recons = recons
        self.latent = latent
        self.mean = mean
        self.log_dev = log_dev

        self._loss = None
        self._rc_loss = None
        self._kl_loss = None
    
    def calc_loss(self, target: TTensor, coef=1.0):
        rc_loss = nn.functional.binary_cross_entropy(self.recons, target, reduction="sum")
        kl_loss = -1/2 * torch.sum(1 + self.log_dev - self.mean ** 2 - self.log_dev.exp())
        self._loss = loss = rc_loss + kl_loss * coef
        self._kl_loss = kl_loss
        self._rc_loss = rc_loss
        return loss
    
    @property
    def loss(self):
        return self._loss.item()

    @property
    def kl_loss(self):
        return self._kl_loss.item()
    
    @property
    def rc_loss(self):
        return self._rc_loss.item()


class VAE(nn.Module):

    def __init__(
            self,
            image_shape: tuple[int] | list[int] | np.ndarray,
            num_layer: int,
            base_channel: int,
            latent_size: int = 50,
    ):
        super().__init__()
        self.encoder = Encoder(image_shape, num_layer, base_channel, latent_size)
        self.decoder = Decoder(self.encoder)
    
    def forward(self, x):
        latent, mean, log_dev = self.encoder(x)
        recons = self.decoder(latent)
        data = VAEData(recons, latent, mean, log_dev)
        return data
    
    def encode(self, images):
        if not torch.is_tensor(images):
            images = torch.from_numpy(images).float()
        if images.ndim == 2:
            images = torch.unsqueeze(images, 0)
        if images.ndim == 3:
            images = torch.unsqueeze(images, 0)
        latent, _, _ = self.encoder(images)
        return latent.detach().numpy()
    
    def decode(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.from_numpy(latents)
        if latents.ndim == 1:
            latents = latents.unsqueeze(latents, 0)
        recons = self.decoder(latents)
        return recons.detach().numpy()

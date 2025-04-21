# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List


class Encoder(nn.Module):

    def __init__(
            self,
            image_shape: Tuple[int] | List[int] | np.ndarray,
            num_layer: int,
            base_channel: int,
            latent_size: int = 50,
    ) -> None:
        super().__init__()
        self.image_shape = np.asarray(image_shape)
        self.num_layer = num_layer
        ch = np.power(2, np.arange(num_layer)) * base_channel
        ch = np.insert(ch, 0, image_shape[0])
        self.ch = ch

        self.shape_after_conv = shape_after_conv = (self.image_shape[1:] / (2 ** num_layer))
        self.size_after_conv = size_after_conv = int(shape_after_conv[0] * shape_after_conv[1] * ch[-1])
        self.latent_size = latent_size

        self.convs = nn.Sequential(*[
            self._make_layer(ich, och)
            for ich, och in zip(ch[:-1], ch[1:])
        ])

        self.layer_flt = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_after_conv, 100),
            nn.ReLU(),
        )
        self.layer_ave = nn.Linear(100, latent_size)
        self.layer_dev = nn.Linear(100, latent_size)
    
    def _make_layer(self, in_ch, out_ch=None):
        out_ch = 2 * in_ch if out_ch is None else out_ch
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = self.layer_flt(x)

        mean = self.layer_ave(x)
        log_dev = self.layer_dev(x)

        eps = torch.rand_like(mean)
        feat = mean + torch.exp(log_dev / 2) * eps
        return feat, mean, log_dev

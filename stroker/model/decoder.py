# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn

from .encoder import Encoder


class Decoder(nn.Module):

    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        ch = encoder.ch[::-1]
        ch[-1] = ch[-2]
        self.ch = ch

        self.shape_resize_to = np.asarray([ch[0], *encoder.shape_after_conv]).astype(int)

        self.layer_latent = nn.Sequential(
            nn.Linear(encoder.latent_size, 100),
            nn.ReLU(),
            nn.Linear(100, encoder.size_after_conv),
            nn.ReLU()
        )
        self.convs = nn.Sequential(*[
            self._make_layer(ich, och)
            for ich, och in zip(ch[:-1], ch[1:])
        ])
        self.layer_recons = nn.Sequential(
            nn.Conv2d(ch[-1], encoder.image_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_ch, out_ch=None):
        if out_ch is None:
            out_ch = in_ch // 2
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def forward(self, x):
        feat = self.layer_latent(x)
        feat = feat.view(x.shape[0], *self.shape_resize_to)
        
        xcv = self.convs(feat)
        recons = self.layer_recons(xcv)

        return recons

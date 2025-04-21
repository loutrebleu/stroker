# -*- coding: utf-8 -*-

import torch.utils as utils
import torchvision.datasets as tvdataset
import torchvision.transforms as tvtransforms



def new_dataset(cf):
    transform = tvtransforms.Compose([
        tvtransforms.RandomCrop(cf.transforms.crop),
        tvtransforms.Resize(cf.transforms.resize),
        tvtransforms.Grayscale(),
        tvtransforms.ToTensor()
    ])
    ds = tvdataset.ImageFolder(cf.path, transform)
    return ds


def new_dataloader(cf):
    dataset = new_dataset(cf.dataset)
    return utils.data.DataLoader(dataset, batch_size=cf.loader.batch_size, shuffle=cf.loader.shuffle)

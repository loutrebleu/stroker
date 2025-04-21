# -*- coding: utf-8 -*-

import os
import time
import xtools as xt
import xsim
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

from stroker.tools.dataloader import new_dataloader
from stroker.model.vae import VAE



def build_vae_parameter(cf: xt.Config):
    dcf = cf.model
    return dcf._cf


def build_result_df(logger: xsim.Logger) -> pd.DataFrame:
    ret = xsim.Retriever(logger)
    res = pd.DataFrame(dict(
        step=ret.step(),
        loss=ret.loss(),
        kl_loss=ret.kl_loss(),
        rc_loss=ret.rc_loss(),
    ))
    return res


def draw_and_save_fig_of_losses(result: pd.DataFrame, filename: str = None, show: bool = False):
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(10, 6))
    result.plot(x="step", y="loss", ax=axes[0])
    result.plot(x="step", y="kl_loss", ax=axes[1])
    result.plot(x="step", y="rc_loss", ax=axes[2])
    if filename is not None:
        fig.savefig(filename)
    if show:
        plt.show()

    return fig, axes



def train(args):
    # config
    config_file = args.config
    cf = xt.Config(config_file)

    # data loader
    loader = new_dataloader(cf)

    # model
    param = build_vae_parameter(cf)
    vae = VAE(**param)

    # trainer
    opt = optim.Adam(vae.parameters(), cf.train.lr)

    # logger
    logger = xsim.Logger()
    result_root_path = "./result"
    result_path = xt.make_dirs_current_time(result_root_path)
    cf.dump(xt.join(result_path, "config.yaml"))
    
    print("[info] start training")
    # for step in tqdm(range(1, cf.train.epoch+1)):
    time_avons = time.time()
    for step in range(1, cf.train.epoch+1):
        ep_loss = 0
        ep_kl_loss = 0
        ep_rc_loss = 0
        for images, labels in loader:
            data = vae(images)
            loss = data.calc_loss(images, coef=cf.train.kl_coef)

            size_batch = len(images)
            ep_loss += data.loss * size_batch
            ep_kl_loss += data.kl_loss * size_batch
            ep_rc_loss += data.rc_loss * size_batch

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        num_data = len(loader)
        ep_loss /= num_data
        ep_kl_loss /= num_data
        ep_rc_loss /= num_data
        logger.store(
            step=step,
            loss=ep_loss,
            kl_loss=ep_kl_loss,
            rc_loss=ep_rc_loss,
        ).flush()

        if step % cf.train.show_interval == 0:
            time_apres = time.time()
            time_epoch = (time_apres - time_avons) / cf.train.show_interval
            print(
                f"[TRAIN] step={step: 6d}; loss={ep_loss: 10.3f}; kl={ep_kl_loss: 10.3f};  rc={ep_rc_loss: 10.3f};  epoch_time={time_epoch: 6.3f}"
            )
            res = build_result_df(logger)
            draw_and_save_fig_of_losses(res, filename=xt.join(result_path, f"losses.png"))
            time_avons = time.time()

        if step % cf.train.save_interval == 0:
            print("[info] save model and figure")
            torch.save(vae.state_dict(), xt.join(result_path, f"model_step={step:06d}.pth"))
            # res = build_result_df(logger)
            # draw_and_save_fig_of_losses(res, filename=xt.join(result_path, f"losses.png"))

    res = build_result_df(logger)
    res.to_csv(xt.join(result_path, "data.csv"), index=False)

    draw_and_save_fig_of_losses(res, filename=xt.join(result_path, "losses.png"), show=False)


def test(args):
    # config
    config_file = args.config
    cf = xt.Config(config_file)
    cf.loader.batch_size = 1
    cf.loader.shuffle = False

    # data loader
    loader = new_dataloader(cf)

    # model
    param = build_vae_parameter(cf)
    vae = VAE(**param)

    # load weight
    state_dict = torch.load(cf.validate.model_path, weights_only=True)
    vae.load_state_dict(state_dict)
    
    np_images = []
    np_images_line = []
    for image, label in loader:
        # print(image.shape)
        np_image = image.detach().numpy()[0, 0,...]
        data = vae(image)
        np_recons = data.recons.detach().numpy()[0, 0, ...]
        np_show_image = np.vstack([np_image, np_recons])
        np_images_line.append(np_show_image)
        if len(np_images_line) == 10:
            np_images.append(np.hstack(np_images_line))
            np_images_line = []
        if len(np_images) >= 4:
            break
    
    np_images = np.vstack(np_images)
    plt.imshow(np_images, cmap="Greys_r")
    plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-v", "--validate", action="store_true")
    # parser.add_argument("-l", "--latent", action="store_true")
    # parser.add_argument("-r", "--reconstruct", action="store_true")

    args = parser.parse_args()
    # if args.train:
    #     args.config = "config_train.yaml"
    # if args.validate:
    #     args.config = "config_validate.yaml"

    print("[info] arguments:", args)


    if args.train:
        train(args)
        exit(0)
    
    if args.validate:
        test(args)
        exit(0)

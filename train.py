import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from src.model import VIDEO_CONTRASTIVE_LEARNING
from src.data import Vdatamodule


config = {
    "img_rep_dim": 512,
    "video_rep_dim": 768,
    "n_layer": 6,
    "bidirectional": False,
    "peak_lr": 1e-4,
    "last_lr": 1e-6,
    "beta_1": 0.9,
    "beta_2": 0.95,
    "weight_decay": 0.1,
    "eps": 1e-08,
    "lr_warmup_perc": 0.1,
    "temperature": 0.1,
    # "debugging": True,
    "train_batch_size": 2,
    "valid_batch_size": 2,
    "num_workers": 2
}

def main():

    L.seed_everything(49, workers=True)

    torch.autograd.set_detect_anomaly(True)

    torch.set_float32_matmul_precision("medium")

    # load model
    model = VIDEO_CONTRASTIVE_LEARNING(**config)

    # load datamodule
    data_module = Vdatamodule(**config)

    trainer = L.Trainer(
        accelerator="gpu",
        check_val_every_n_epoch=5,
        max_epochs=50,
        gradient_clip_val=1.0
    )

    trainer.fit(model, datamodule=data_module)

    # trainer.test(model, datamodule=data_module)

    # trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()

import pytorch_lightning as pl
import argparse
import os
from glob import glob
import BMIC_Utils as bmu

from datamod.dataset import LitDataModule
from model.model_pl import LitModel


def main(config,ckpt):
    dataset = bmu.autoArgs(config.data, LitDataModule)
    model = LitAEModel.load_from_checkpoint(ckpt)
    
    trainer = pl.Trainer(logger=False)
    pred_out = trainer.predict(model, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--cfg", default="checkpoints/", type=str)
    parser.add_argument("--ckpt", default="checkpoints/CT_AE/aaskn1l0/", type=str)

    argv = parser.parse_args()

    config = bmu.yaml_read(argv.cfg + "/config.yaml")
    if os.path.isdir(argv.ckpt):
        ckpt = glob(argv.cfg + "/checkpoints/*")
        ckpt = ckpt[0]

    argv = parser.parse_args()

    config = bmu.yaml_read(argv.cfg)
    main(config,ckpt)

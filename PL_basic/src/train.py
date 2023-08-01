import sys

import pytorch_lightning as pl

from pytorch_lightning.cli import LightningCLI
import torch
from datamod.data_pl import LitDataModule
from model.model_pl import LitModel
import os




def cli_main():
    cli = LightningCLI(
        LitDataModule,
        LitModel,
    )

if __name__ == "__main__":
    cli_main()
    print('Done')

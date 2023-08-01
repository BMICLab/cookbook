import pytorch_lightning as pl
from torch.utils.data import DataLoader,random_split
import BMIC_Utils as bmu

from PL_basic.datamod.dataset import ImgDataset

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self, img_fld, gt_fld, batch_size=1, crop=512, n_sample=-1, workers=6
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        data_full = bmu.autoArgs(self.hparams, ImgDataset)

        train_size, val_size = [int(x * len(data_full)) for x in [0.8, 0.2]]
        test_size = len(data_full) - (train_size + val_size)
        self.train, self.val, self.test = random_split(
            data_full, [train_size, val_size, test_size]
        )
        
        self.val.dataset.is_val=True
        self.test.dataset.is_val=True

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )
        
if __name__ == "__main__":
    dataset = LitDataModule(
        img_fld='data/imgs',
        gt_fld='data/labels',
        batch_size=3
    )
    dataset.setup('train')

    print(len(dataset.train_dataloader()))
from typing import Any
import pytorch_lightning as pl
import BMIC_Utils as bmu
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.arc import ANN

class LitModel(pl.LightningModule):
    def __init__(self, lr,ce_weight,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        
        self.model = bmu.autoArgs(self.hparams,ANN)
        
        self.criterion = # TODO: Loss
        self.validation_step_outputs = []
        
    def forward(self, batch):
        self.model.eval()
        return self.model(batch)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x,y = batch
        pred = self.model(x)
        return pred
        
    
    def training_step(self, batch) -> STEP_OUTPUT:
        x,y = batch
        
        output = self.model(x)
        
        loss = self.criterion(output,y)
        
        self.log('train_loss', loss, prog_bar=True,sync_dist=True)
        return loss
        
    def validation_step(self, batch,idx) -> STEP_OUTPUT:
        x,y = batch
        
        output = self.model(x)
        
        loss = self.criterion(output,y)
        
        self.log('val_loss', loss, prog_bar=True,sync_dist=True)
        
        if idx==0: # Save stuff for end of validation report
            self.validation_step_outputs.append(
                {
                    'img':  x.detach().cpu().numpy().squeeze(),
                    'pred': output.detach().cpu().numpy().squeeze(),
                    'gt':   y.detach().cpu().numpy().squeeze(),
                }
            )
        return loss
            
    def on_validation_end(self) -> None:
        img = self.validation_step_outputs[0]['img']
        pred = self.validation_step_outputs[0]['pred']
        gt = self.validation_step_outputs[0]['gt']
                
        self.logger.log_image(key='Predict', images=[img],masks=[{
                "predictions": {
                    "mask_data": pred.argmax(0), # For segmentation
                    "class_labels": {0:'BG',1:'Artery'}
                },
                "ground_truth": {
                    "mask_data": gt,
                    "class_labels": {0:'BG',1:'Artery'}
                },
                }])

        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        ]
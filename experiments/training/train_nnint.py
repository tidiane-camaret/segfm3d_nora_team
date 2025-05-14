import os
import scipy as sp
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch.nn as nn
from torch.utils.data import DataLoader

from src.methods.nninteractivecore import nnInteractiveCorePredictor
from src.config import config
from src.training.dataloading import SimpleInteractiveSegmentationDataset #, custom_collate_fn
from src.training.loss import DiceLoss
from src.eval_metrics import (  # TODO : Use the competition repo as source instead
    compute_multi_class_dsc,
    compute_multi_class_nsd,
)
class InteractiveSegmentationModel(pl.LightningModule):
    def __init__(self, checkpoint_path, learning_rate=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained model
        predictor = nnInteractiveCorePredictor(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),  # Lightning will handle device placement
            verbose=False,
        )
        self.network = predictor.network
        del predictor
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        # Process each item in the batch separately
        inputs = batch['input']
        targets = batch['gt']
        
        total_loss = 0
        for i in range(len(inputs)):
            # Forward pass
            outputs = self.network(inputs[i].unsqueeze(0))

            # Compute loss
            loss = self.criterion(outputs, targets[i].unsqueeze(0))
            
            # Track loss
            total_loss += loss
            
            # Log per-sample loss
            self.log(f"train_sample_{i}_loss", loss, prog_bar=False)
        
        # Calculate average loss
        avg_loss = total_loss / len(inputs)
        
        # Log metrics
        self.log("train_loss", avg_loss, prog_bar=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        # Process each item in the batch separately
        inputs = batch['input']
        targets = batch['gt']
        spacing = batch['spacing']
        
        total_loss = 0
        total_dsc = 0
        total_nsd = 0
        for i in range(len(inputs)):
            # Forward pass
            outputs = self.network(inputs[i].unsqueeze(0))
            # Compute loss
            loss = self.criterion(outputs, targets[i].unsqueeze(0))
            output_masks = outputs.argmax(dim=1).cpu().numpy().squeeze(0)
            dsc = compute_multi_class_dsc(output_masks, targets[i].cpu().numpy())
            # compute nsd
            if dsc > 0.2:
                # only compute nsd when dice > 0.2 because NSD is also low when dice is too low

                nsd = compute_multi_class_nsd(output_masks, targets[i].cpu().numpy(), spacing[i].cpu().numpy())
            else:
                nsd = 0.0  # Assume model performs poor on this sample
            
            # Track loss
            total_loss += loss
            total_dsc += dsc
            total_nsd += nsd
        
        # Calculate average loss
        avg_loss = total_loss / len(inputs)
        avg_dsc = total_dsc / len(inputs)
        avg_nsd = total_nsd / len(inputs)
        
        # Log metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_dsc", avg_dsc, prog_bar=True)
        self.log("val_nsd", avg_nsd, prog_bar=True)
        
        return avg_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


class InteractiveSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, gt_dir, batch_size=2, num_workers=4):
        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # Create train and validation datasets
        self.train_dataset = SimpleInteractiveSegmentationDataset(
            img_dir=self.img_dir,
            gt_dir=self.gt_dir,
            split='train'
        )
        
        self.val_dataset = SimpleInteractiveSegmentationDataset(
            img_dir=self.img_dir,
            gt_dir=self.gt_dir,
            split='val'
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            #collate_fn=custom_collate_fn,
            pin_memory=True
        )


def train_model():
    # Set up paths
    checkpoint_path = os.path.join(config["NNINT_CKPT_DIR"], "nnInteractive_v1.0")
    img_dir = os.path.join(config["DATA_DIR"], "3D_val_npz")
    gt_dir = os.path.join(config["DATA_DIR"], "3D_val_gt_interactive_seg")
    
    # Initialize data module
    data_module = InteractiveSegmentationDataModule(
        img_dir=img_dir,
        gt_dir=gt_dir,
        batch_size=2,
        num_workers=4
    )
    
    # Initialize model
    model = InteractiveSegmentationModel(
        checkpoint_path=checkpoint_path,
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # Set up WandbLogger
    wandb_logger = WandbLogger(
        project="interactive-segmentation",
        name="nninteractive-finetune",
        log_model=True
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="/nfs/norasys/notebooks/camaret/model_checkpoints/nnint/finetuned",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices="auto",  # Use all available GPUs
        strategy=DDPStrategy(find_unused_parameters=True),
        #logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,  # Use mixed precision for faster training
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train_model()
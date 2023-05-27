from src.utils.utils import common_load_dataset
from src.methods.abstract_method import AbstractMethod
from src.methods.dirty.utils.DirtyDataset import DirtyDataset
from src.methods.dirty.utils import util
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader
import _jsonnet
from src.methods.dirty.model.model import TypeReconstructionModel
import pytorch_lightning as pl
import json
import torch


class DirtyMethod(AbstractMethod):
    def __init__(self, train_set, dev_set, test_set, config, dirty_config, extra_config = None, ckpt = None):
        super().__init__(train_set, dev_set, test_set, config)
        dirty_config = json.loads(_jsonnet.evaluate_file(dirty_config))
        self.dirty_config = dirty_config
        if extra_config:
            extra_config = json.loads(extra_config)
            dirty_config = util.update(dirty_config, extra_config)

        batch_size = dirty_config["train"]["batch_size"]
        train_set = DirtyDataset(self.train_data,
                                 dirty_config["data"], percent=float(config['percent'])
                                 )
        dev_set = DirtyDataset(self.val_data, dirty_config["data"])
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            collate_fn=DirtyDataset.collate_fn,
            num_workers=16,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            dev_set,
            batch_size=batch_size,
            collate_fn=DirtyDataset.collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        wandb_logger = WandbLogger(name=config['exp_name'], project="dire", log_model=True)
        wandb_logger.log_hyperparams(config)
        # model
        self.model = TypeReconstructionModel(dirty_config)
        self.ckpt = ckpt

        resume_from_checkpoint = ckpt
        if resume_from_checkpoint == "":
            resume_from_checkpoint = None
        self.trainer = pl.Trainer(
            max_epochs=dirty_config["train"]["max_epoch"],
            logger=wandb_logger,
            gpus=1 if config['cuda'] else None,
            auto_select_gpus=True,
            gradient_clip_val=1,
            callbacks=[
                EarlyStopping(
                    monitor="val_retype_acc"
                    if dirty_config["data"]["retype"]
                    else "val_rename_acc",
                    mode="max",
                    patience=dirty_config["train"]["patience"],
                )
            ],
            check_val_every_n_epoch=dirty_config["train"]["check_val_every_n_epoch"],
            progress_bar_refresh_rate=10,
            accumulate_grad_batches=dirty_config["train"]["grad_accum_step"],
            resume_from_checkpoint=resume_from_checkpoint,
            limit_test_batches=dirty_config["test"]["limit"] if "limit" in dirty_config["test"] else 1.0
        )

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def evaluate(self):
        test_set = DirtyDataset(self.test_data, self.dirty_config["data"])
        test_loader = DataLoader(
            test_set,
            batch_size=self.dirty_config["test"]["batch_size"],
            collate_fn=DirtyDataset.collate_fn,
            num_workers=8,
            pin_memory=True,
        )
        ret = self.trainer.test(self.model, test_dataloaders=test_loader, ckpt_path=self.ckpt)
        json.dump(ret[0], open("test_result.json", "w"))

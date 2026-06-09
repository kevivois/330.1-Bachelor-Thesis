from lightning.pytorch.callbacks import Callback
class LossHistory(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(
                trainer.callback_metrics["train_loss"].item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(
                trainer.callback_metrics["val_loss"].item()
            )
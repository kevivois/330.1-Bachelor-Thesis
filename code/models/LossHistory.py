from lightning.pytorch.callbacks import Callback

'''
Class used to receive in callback the validation and train loss in a training of darts model and therefore used to plot
This code has been developped with the help of Gemini
'''
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
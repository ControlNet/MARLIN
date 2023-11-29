from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
import re


class EarlyStoppingLR(Callback):

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True

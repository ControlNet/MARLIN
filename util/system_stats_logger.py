from pytorch_lightning import Callback, Trainer, LightningModule


class SystemStatsLogger(Callback):
    """Log system stats for each training epoch"""

    def __init__(self):
        try:
            import psutil
        except ImportError:
            raise ImportError("psutil is required to use SystemStatsLogger")
        self.psutil = psutil

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        cpu_usage = self.psutil.cpu_percent()
        memory_usage = self.psutil.virtual_memory().percent
        logged_info = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }
        pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
        pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)

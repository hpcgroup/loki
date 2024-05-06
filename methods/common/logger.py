from methods.common.configure_model import get_config_dict
import wandb
import torch
import os


class WandbLogger:
    def __init__(self, args):
        self.rank = os.environ.get("RANK")
        if self.rank == '0':
            self.config = get_config_dict(args)
            self.run = wandb.init(project='PCA-TopK', config=self.config)
    
    def update_config(self, kwargs):
        if self.rank == '0':
            self.run.config.update(kwargs)

    def log(self, kwargs):
      if self.rank == '0':
          self.run.log(kwargs)
    
    def finish(self):
      if self.rank == '0':
          self.run.finish()

class NoOpLogger:
    def __init__(self, args):
        pass

    def update_config(self, kwargs):
        pass

    def log(self, kwargs):
        pass
    
    def finish(self):
        pass
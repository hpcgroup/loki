from methods.common.configure_model import get_config_dict
import wandb
import torch
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

class WandbLogger:
    def __init__(self, args):
        self.rank = os.environ.get("RANK")
        if self.rank == '0':
            self.config = get_config_dict(args)
            jobid = os.environ.get("JOBID", '0')
            if args.lm_harness_eval:
                groupid = "lm_harness"
            else:
                groupid = "ppl"
            self.run = wandb.init(project='Loki', config=self.config, name=jobid, 
                                  group=groupid, job_type='eval', tags=[groupid])
            wandb.define_metric("compression_ratio", summary="mean")
    
    def update_config(self, kwargs):
        if self.rank == '0':
            self.run.config.update(kwargs)

    def log(self, kwargs):
      if self.rank == '0':
          self.run.log(kwargs)
    
    def log_ppl(self, ppl):
      if self.rank == '0':
          self.run.log({'perplexity': ppl})
    
    def log_lm_harness_results(self, tasks, results):
      if self.rank == '0':
          assert results is not None
          for task in tasks.keys():
              metric = tasks[task]
              result = results[task]
              if metric in result.keys():
                  # Replace ,none with empty string
                  metric_name = metric.replace(",none", "")
                  self.run.log({task + "_" + metric_name: result[metric]})
    
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

    def log_ppl(self, ppl):
        pass
    
    def finish(self):
        pass
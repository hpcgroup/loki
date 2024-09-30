from .common.saver import TensorSaver as TensorSaver
from .common.logger import WandbLogger as WandbLogger
from .common.logger import NoOpLogger as NoOpLogger

G_TENSOR_SAVER = None
LOGGER = None
G_TIMERS = None

def init_tensor_saver(tensor_dir):
    global G_TENSOR_SAVER 
    G_TENSOR_SAVER = TensorSaver(tensor_dir)

def init_logger(args):
    global LOGGER
    if args.use_wandb:
        LOGGER = WandbLogger(args)
    else:
        LOGGER = NoOpLogger(args)

def finish_logger():
    global LOGGER
    if LOGGER is not None:
        LOGGER.finish()




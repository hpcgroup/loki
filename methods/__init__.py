#from .baselines.h2o.modify_llama import make_llama_attention_h2o as make_llama_attention_h2o
#from .baselines.h2o_hf_opt.modify_llama import make_llama_attention_h2o as make_llama_attention_h2o
#from .baselines.h2o.modify_opt import make_opt_attention_h2o as make_opt_attention_h2o
#from .baselines.h2o.modify_mistral import make_mistral_attention_h2o as make_mistral_attention_h2o
#
#from .baselines.topk.modify_llama import make_llama_attention_top_k as make_llama_attention_top_k
#from .baselines.topk.modify_opt import make_opt_attention_top_k as make_opt_attention_top_k
#from .baselines.topk.modify_mistral import make_mistral_attention_top_k as make_mistral_attention_top_k
#from .baselines.topk.modify_gptneox import make_gptneox_attention_top_k as make_gptneox_attention_top_k
#
#from .baselines.sparq.modify_llama import make_llama_attention_sparq as make_llama_attention_sparq
#
#
#from .spark.modify_llama import make_llama_attention_spark as make_llama_attention_spark
#from .sparhat.modify_llama import make_llama_attention_sparhat as make_llama_attention_sparhat
#from .sparhat.cache_utils import SparHatCache as SparHatCache
#
#from .pca.modify_llama import make_llama_attention_pca as make_llama_attention_pca
#from .pca.modify_mistral import make_mistral_attention_pca as make_mistral_attention_pca
#
#from .pca_topk.modify_llama import make_llama_attention_pca_topk as make_llama_attention_pca_topk
#from .pca_topk.modify_mistral import make_mistral_attention_pca_topk as make_mistral_attention_pca_topk

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




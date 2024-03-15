#from .baselines.h2o.modify_llama import make_llama_attention_h2o as make_llama_attention_h2o
from .baselines.h2o_hf_opt.modify_llama import make_llama_attention_h2o as make_llama_attention_h2o
from .baselines.h2o.modify_opt import make_opt_attention_h2o as make_opt_attention_h2o
from .baselines.h2o.modify_mistral import make_mistral_attention_h2o as make_mistral_attention_h2o

from .baselines.topk.modify_llama import make_llama_attention_top_k as make_llama_attention_top_k
from .baselines.topk.modify_opt import make_opt_attention_top_k as make_opt_attention_top_k
from .baselines.topk.modify_mistral import make_mistral_attention_top_k as make_mistral_attention_top_k

from .baselines.sparq.modify_llama import make_llama_attention_sparq as make_llama_attention_sparq


from .spark.modify_llama import make_llama_attention_spark as make_llama_attention_spark
from .sparhat.modify_llama import make_llama_attention_sparhat as make_llama_attention_sparhat
from .sparhat.cache_utils import SparHatCache as SparHatCache

from .pca.modify_llama import make_llama_attention_pca as make_llama_attention_pca

from .common.saver import TensorSaver as TensorSaver

G_TENSOR_SAVER = None

def init_tensor_saver(tensor_dir):
    global G_TENSOR_SAVER 
    G_TENSOR_SAVER = TensorSaver(tensor_dir)



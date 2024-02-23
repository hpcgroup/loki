from .baselines.h2o.modify_llama import make_llama_attention_h2o as make_llama_attention_h2o
from .baselines.h2o.modify_opt import make_opt_attention_h2o as make_opt_attention_h2o

from .baselines.topk.modify_llama import make_llama_attention_top_k as make_llama_attention_top_k
from .baselines.topk.modify_opt import make_opt_attention_top_k as make_opt_attention_top_k

from .baselines.sparq.modify_llama import make_llama_attention_sparq as make_llama_attention_sparq


from .spark.modify_llama import make_llama_attention_spark as make_llama_attention_spark
from .sparhat.modify_llama import make_llama_attention_sparhat as make_llama_attention_sparhat
from .sparhat.cache_utils import SparHatCache as SparHatCache
from .histh2o.modify_llama import make_llama_attention_histh2o as make_llama_attention_histh2o
#from .common.saver import TensorSaver as TensorSaver


#tensor_saver = None

#def init_tensor_saver(output_dir):
#    global tensor_saver
#    tensor_saver = TensorSaver(output_dir)

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.opt.modeling_opt import OPTAttention


from .modify_gpt_neox import get_top_k_forward_gpt_neox
from .modify_opt import get_top_k_forward_opt


def make_attention_top_k(top_k):
    OPTAttention.forward = get_top_k_forward_opt(top_k)
    GPTNeoXAttention.forward = get_top_k_forward_gpt_neox(top_k)
